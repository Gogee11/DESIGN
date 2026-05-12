"""LightGCN + LLM-Augmented Graph Regularization (LightGCN_AGR).

Active loss components (each gated by its own weight in the YAML config; set the
weight to 0 to ablate):
    - L_bpr   : BPR loss on the user-item training graph (cf view)
    - L_reg   : L2 regularization on user/item embeddings (reg_weight)
    - L_prf   : InfoNCE between cf embeddings and LLM-projected semantic
                embeddings (prf_weight)
    - L_str   : InfoNCE between cf view and LLM-augmented graph view
                (str_weight) - "augmented graph" comes from data_handler.aug_torch_adj
                which is built by adding top-K LLM-similarity edges to the
                training bipartite graph
    - L_recon : Reconstruct LLM semantic vector from a masked cf embedding
                (recon_weight) - masked autoencoder style self-supervision
    - L_ib    : HSIC information-bottleneck between cf and aug views (beta)

Total loss = L_bpr + L_reg + L_prf + L_str + L_recon + L_ib.

Each weight maps to exactly one term so a hyperparameter sweep is interpretable.
"""

import torch as t
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.aug_utils import NodeMask
from models.base_model import BaseModel
from models.loss_utils import cal_infonce_loss, ssl_con_loss
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


def kernel_matrix(x, sigma):
    return t.exp((t.matmul(x, x.transpose(0, 1)) - 1) / sigma)


def hsic(Kx, Ky, m):
    if m < 2:
        return Kx.new_tensor(0.0)
    Kxy = t.mm(Kx, Ky)
    h = t.trace(Kxy) / m ** 2 + t.mean(Kx) * t.mean(Ky) - 2 * t.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2


class LightGCN_AGR(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_AGR, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        if not hasattr(data_handler, 'aug_torch_adj'):
            raise RuntimeError(
                "LightGCN_AGR requires data_handler.aug_torch_adj. "
                "Make sure the dataset has user_embedding/item_embedding pkl files "
                "and the model name contains 'agr' so the data handler builds it."
            )
        self.aug_adj = data_handler.aug_torch_adj

        self.dataset = configs['data']['name']
        self._init_dataset_config()

        # Shared user/item embeddings: contrasts come from different graphs (cf vs aug),
        # not from a separate parameter set.
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        # Frozen LLM semantic embeddings (precomputed via SentenceTransformer).
        device = configs['device']
        self.usrprf_embeds = t.tensor(configs['user_embedding']).float().to(device)
        self.itmprf_embeds = t.tensor(configs['item_embedding']).float().to(device)
        self.prf_embeds = t.cat([self.usrprf_embeds, self.itmprf_embeds], dim=0)
        prf_dim = self.usrprf_embeds.shape[1]

        hidden_dim = (prf_dim + self.embedding_size) // 2
        self.mlp = nn.Sequential(
            nn.Linear(prf_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.embedding_size),
        )
        self.gen_mlp = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, prf_dim),
        )
        for module in (self.mlp, self.gen_mlp):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    init(layer.weight)

        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.edge_dropper = SpAdjEdgeDrop()

        self.is_training = False
        self.final_embeds = None
        self.batch_size = 1

    def _init_dataset_config(self):
        cfg = configs['model']
        ds_cfg = cfg.get(self.dataset, {})

        def _get(key, default):
            if key in ds_cfg:
                return ds_cfg[key]
            return cfg.get(key, default)

        self.layer_num = int(_get('layer_num', 3))
        self.reg_weight = float(_get('reg_weight', 1e-7))
        self.keep_rate = float(_get('keep_rate', 0.8))

        self.mask_ratio = float(_get('mask_ratio', 0.15))
        self.recon_weight = float(_get('recon_weight', 0.0))
        self.re_temperature = float(_get('re_temperature', 0.2))

        self.kd_temperature = float(_get('kd_temperature', 0.2))

        self.prf_weight = float(_get('prf_weight', 0.0))
        self.str_weight = float(_get('str_weight', 0.0))
        self.beta = float(_get('beta', 0.0))
        self.sigma = float(_get('sigma', 0.25))

    def _mask(self):
        embeds = t.cat([self.user_embeds, self.item_embeds], dim=0)
        masked_embeds, seeds = self.masker(embeds)
        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds

    def _propagate(self, adj, embeds_in):
        return t.spmm(adj, embeds_in)

    def forward(self, adj=None, keep_rate=1.0, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj

        # Cache only the un-masked, full-keep_rate inference pass on the cf graph.
        cache_eligible = (
            (not self.is_training)
            and adj is self.adj
            and masked_user_embeds is None
            and masked_item_embeds is None
            and keep_rate == 1.0
        )
        if cache_eligible and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = t.cat([self.user_embeds, self.item_embeds], dim=0)
        else:
            embeds = t.cat([masked_user_embeds, masked_item_embeds], dim=0)

        if self.is_training and keep_rate < 1.0:
            adj = self.edge_dropper(adj, keep_rate)

        out = embeds
        embeds_list = [embeds]
        for _ in range(self.layer_num):
            out = self._propagate(adj, out)
            embeds_list.append(out)
        out = sum(embeds_list)

        if cache_eligible:
            self.final_embeds = out
        return out[:self.user_num], out[self.user_num:]

    def _pick(self, ue, ie, batch_data):
        ancs, poss, negs = batch_data
        return ue[ancs], ie[poss], ie[negs]

    def _bpr_loss(self, ue, ie, batch_data):
        ancs, poss, negs = batch_data
        u, p, n = ue[ancs], ie[poss], ie[negs]
        pos = (u * p).sum(-1)
        neg = (u * n).sum(-1)
        return -t.log(t.sigmoid(pos - neg) + 1e-9).mean()

    def _l2_reg(self, ancs, poss, negs):
        u = self.user_embeds[ancs]
        p = self.item_embeds[poss]
        n = self.item_embeds[negs]
        return 0.5 * (u.norm(2).pow(2) + p.norm(2).pow(2) + n.norm(2).pow(2)) / max(len(ancs), 1)

    def _hsic_term(self, users, pos_items, ue_cf, ie_cf, ue_aug, ie_aug):
        m = self.batch_size
        users_u = t.unique(users)
        x = F.normalize(ue_cf[users_u], p=2, dim=1)
        y = F.normalize(ue_aug[users_u], p=2, dim=1)
        Kx = kernel_matrix(x, self.sigma)
        Ky = kernel_matrix(y, self.sigma)
        loss_u = hsic(Kx, Ky, m)

        items_u = t.unique(pos_items)
        xi = F.normalize(ie_cf[items_u], p=2, dim=1)
        yi = F.normalize(ie_aug[items_u], p=2, dim=1)
        Ki = kernel_matrix(xi, self.sigma)
        Kj = kernel_matrix(yi, self.sigma)
        loss_i = hsic(Ki, Kj, m)

        return loss_u + loss_i

    def cal_loss(self, batch_data):
        self.is_training = True
        self.batch_size = len(batch_data[0])
        ancs, poss, negs = batch_data
        device = self.user_embeds.device
        zero = t.zeros((), device=device)

        # Optional masking for reconstruction self-supervision.
        if self.recon_weight > 0 and self.mask_ratio > 0:
            masked_u, masked_i, seeds = self._mask()
        else:
            masked_u, masked_i, seeds = None, None, None

        # Two views: cf graph and LLM-augmented graph.
        ue_cf, ie_cf = self.forward(self.adj, self.keep_rate, masked_u, masked_i)
        if self.str_weight > 0 or self.beta > 0:
            ue_aug, ie_aug = self.forward(self.aug_adj, self.keep_rate, masked_u, masked_i)
        else:
            ue_aug, ie_aug = None, None

        bpr_loss = self._bpr_loss(ue_cf, ie_cf, batch_data)
        reg_loss = self.reg_weight * self._l2_reg(ancs, poss, negs)

        if self.prf_weight > 0:
            user_prf = self.mlp(self.usrprf_embeds)
            item_prf = self.mlp(self.itmprf_embeds)
            anc_e, pos_e, neg_e = self._pick(ue_cf, ie_cf, batch_data)
            anc_p, pos_p, neg_p = self._pick(user_prf, item_prf, batch_data)
            prf_loss = (
                cal_infonce_loss(anc_e, anc_p, user_prf, self.kd_temperature)
                + cal_infonce_loss(pos_e, pos_p, item_prf, self.kd_temperature)
                + cal_infonce_loss(neg_e, neg_p, item_prf, self.kd_temperature)
            ) / max(len(ancs), 1)
            prf_loss = prf_loss * self.prf_weight
        else:
            prf_loss = zero

        if self.str_weight > 0 and ue_aug is not None:
            anc_c, pos_c, neg_c = self._pick(ue_cf, ie_cf, batch_data)
            anc_a, pos_a, neg_a = self._pick(ue_aug, ie_aug, batch_data)
            str_loss = (
                cal_infonce_loss(anc_c, anc_a, ue_aug, self.kd_temperature)
                + cal_infonce_loss(pos_c, pos_a, ie_aug, self.kd_temperature)
                + cal_infonce_loss(neg_c, neg_a, ie_aug, self.kd_temperature)
            ) / max(len(ancs), 1)
            str_loss = str_loss * self.str_weight
        else:
            str_loss = zero

        if self.recon_weight > 0 and seeds is not None:
            combined = t.cat([ue_cf, ie_cf], dim=0)
            recon = self.gen_mlp(combined[seeds])
            target = self.prf_embeds[seeds]
            recon_loss = ssl_con_loss(recon, target, self.re_temperature) * self.recon_weight
        else:
            recon_loss = zero

        if self.beta > 0 and ue_aug is not None:
            ib_loss = self._hsic_term(ancs, poss, ue_cf, ie_cf, ue_aug, ie_aug) * self.beta
        else:
            ib_loss = zero

        loss = bpr_loss + reg_loss + prf_loss + str_loss + recon_loss + ib_loss
        losses = {
            'bpr_loss': bpr_loss.detach(),
            'reg_loss': reg_loss.detach() if isinstance(reg_loss, t.Tensor) else zero,
            'prf_loss': prf_loss.detach() if isinstance(prf_loss, t.Tensor) else zero,
            'str_loss': str_loss.detach() if isinstance(str_loss, t.Tensor) else zero,
            'recon_loss': recon_loss.detach() if isinstance(recon_loss, t.Tensor) else zero,
            'ib_loss': ib_loss.detach() if isinstance(ib_loss, t.Tensor) else zero,
        }
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        self.final_embeds = None  # invalidate stale cache
        ue, ie = self.forward(self.adj, 1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        full_preds = ue[pck_users] @ ie.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

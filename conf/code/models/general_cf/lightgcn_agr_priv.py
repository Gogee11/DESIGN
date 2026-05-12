"""LightGCN_AGR with explicit privacy hooks (Plan B variant).

Identical to LightGCN_AGR except:
    1. Embeddings are L2-normalized at inference and during BPR scoring.
       This bounds attack-feature magnitudes (cosine + score now identical scale)
       and is known to reduce score-based MIA.
    2. Inference embeddings get a small zero-mean Gaussian perturbation
       (priv_noise_std). Training is unchanged so utility loss is bounded; the
       perturbation only affects the user-facing predict path and the MIA's view
       of the model output. Set priv_noise_std=0 to disable.

Switch by setting `--model lightgcn_agr_priv` and adding the priv_noise_std
field to your YAML (or pass via CLI override).
"""
import torch as t
import torch.nn.functional as F

from config.configurator import configs
from models.general_cf.lightgcn_agr import LightGCN_AGR


class LightGCN_AGR_Priv(LightGCN_AGR):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        cfg = configs['model']
        ds_cfg = cfg.get(self.dataset, {})
        self.priv_noise_std = float(ds_cfg.get('priv_noise_std', cfg.get('priv_noise_std', 0.0)))
        self.normalize_at_predict = bool(ds_cfg.get('normalize_at_predict',
                                                    cfg.get('normalize_at_predict', True)))

    def full_predict(self, batch_data):
        self.is_training = False
        self.final_embeds = None
        ue, ie = self.forward(self.adj, 1.0)
        if self.normalize_at_predict:
            ue = F.normalize(ue, p=2, dim=-1)
            ie = F.normalize(ie, p=2, dim=-1)
        if self.priv_noise_std > 0:
            ue = ue + t.randn_like(ue) * self.priv_noise_std
            ie = ie + t.randn_like(ie) * self.priv_noise_std
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        full_preds = ue[pck_users] @ ie.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

    def forward(self, adj=None, keep_rate=1.0, masked_user_embeds=None, masked_item_embeds=None):
        ue, ie = super().forward(adj=adj, keep_rate=keep_rate,
                                  masked_user_embeds=masked_user_embeds,
                                  masked_item_embeds=masked_item_embeds)
        # Inference path (no grad, MIA-visible) gets normalization + noise.
        if (not self.is_training) and self.normalize_at_predict:
            ue = F.normalize(ue, p=2, dim=-1)
            ie = F.normalize(ie, p=2, dim=-1)
            if self.priv_noise_std > 0:
                ue = ue + t.randn_like(ue) * self.priv_noise_std
                ie = ie + t.randn_like(ie) * self.priv_noise_std
        return ue, ie

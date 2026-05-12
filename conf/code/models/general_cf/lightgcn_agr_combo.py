"""LightGCN_AGR_Combo: jointly enables M1 (confidence regularization) and
M4 (adversarial training with gradient reversal). Both losses are added on top
of the parent LightGCN_AGR loss; either knob can be set to 0 to ablate.

This is a stretch experiment to test whether the two defenses combine
super-additively.
"""
import torch as t
from torch import nn

from config.configurator import configs
from models.general_cf.lightgcn_agr import LightGCN_AGR
from models.general_cf.lightgcn_agr_adv import grad_reverse


class LightGCN_AGR_Combo(LightGCN_AGR):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        cfg = configs['model']
        ds_cfg = cfg.get(self.dataset, {})

        def _get(key, default):
            if key in ds_cfg:
                return ds_cfg[key]
            return cfg.get(key, default)

        # M1 confidence-reg knobs
        self.conf_weight = float(_get('conf_weight', 0.0))
        self.conf_target = float(_get('conf_target', 0.7))
        self.conf_neg_target = float(_get('conf_neg_target', 0.3))
        # M4 adversarial knobs
        self.adv_weight = float(_get('adv_weight', 0.0))
        self.adv_lambda = float(_get('adv_lambda', 1.0))

        d = self.embedding_size * 4
        self.disc = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _disc_features(self, u, i):
        return t.cat([u, i, (u - i).abs(), u * i], dim=-1)

    def cal_loss(self, batch_data):
        loss, losses = super().cal_loss(batch_data)
        ancs, poss, negs = batch_data
        ue, ie = self.forward(self.adj, self.keep_rate)

        # M1 confidence loss
        if self.conf_weight > 0:
            u, ip, in_ = ue[ancs], ie[poss], ie[negs]
            pos_score = (u * ip).sum(-1); neg_score = (u * in_).sum(-1)
            pos_prob = t.sigmoid(pos_score); neg_prob = t.sigmoid(neg_score)
            conf_loss = ((pos_prob - self.conf_target) ** 2).mean() \
                        + ((neg_prob - self.conf_neg_target) ** 2).mean()
            conf_loss = conf_loss * self.conf_weight
            loss = loss + conf_loss
            losses['conf_loss'] = conf_loss.detach()

        # M4 adversarial loss
        if self.adv_weight > 0:
            u_m, i_m = ue[ancs], ie[poss]
            u_n, i_n = ue[ancs], ie[negs]
            feat_m = grad_reverse(self._disc_features(u_m, i_m), self.adv_lambda)
            feat_n = grad_reverse(self._disc_features(u_n, i_n), self.adv_lambda)
            logit_m = self.disc(feat_m).squeeze(-1)
            logit_n = self.disc(feat_n).squeeze(-1)
            bce = t.nn.functional.binary_cross_entropy_with_logits
            adv_loss = bce(logit_m, t.ones_like(logit_m)) \
                       + bce(logit_n, t.zeros_like(logit_n))
            adv_loss = adv_loss * self.adv_weight
            loss = loss + adv_loss
            losses['adv_loss'] = adv_loss.detach()

        return loss, losses

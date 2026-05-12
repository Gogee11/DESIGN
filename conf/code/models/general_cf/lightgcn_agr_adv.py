"""LightGCN_AGR with adversarial MIA training (Plan D).

A small MLP discriminator is trained jointly with the recommender. Its job is
to predict, given (u_emb, i_emb) features, whether (u, i) is a member of the
training set. Through a gradient-reversal layer (GRL), the encoder receives
NEGATED gradients from the discriminator, pushing the embeddings toward
configurations where members and non-members are indistinguishable.

This is a minimax / GAN-style defense: at convergence the discriminator should
be unable to do better than chance on member vs non-member, which directly
translates to a lower MIA AUC for any downstream attacker that uses similar
features.

Knobs (overridable via CLI):
    adv_weight  : multiplier on the adversarial encoder loss (default 0)
    adv_lambda  : GRL gradient scale; controls how strongly encoder is pushed
                  to fool the disc (default 1.0)
"""
import torch as t
from torch import nn

from config.configurator import configs
from models.general_cf.lightgcn_agr import LightGCN_AGR


class _GradReverse(t.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return _GradReverse.apply(x, lambda_)


class LightGCN_AGR_Adv(LightGCN_AGR):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        cfg = configs['model']
        ds_cfg = cfg.get(self.dataset, {})

        def _get(key, default):
            if key in ds_cfg:
                return ds_cfg[key]
            return cfg.get(key, default)

        self.adv_weight = float(_get('adv_weight', 0.0))
        self.adv_lambda = float(_get('adv_lambda', 1.0))

        d = self.embedding_size * 4  # [u, i, |u-i|, u*i]
        self.disc = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _disc_features(self, u, i):
        return t.cat([u, i, (u - i).abs(), u * i], dim=-1)

    def cal_loss(self, batch_data):
        loss, losses = super().cal_loss(batch_data)
        if self.adv_weight <= 0:
            return loss, losses
        ancs, poss, negs = batch_data
        ue_cf, ie_cf = self.forward(self.adj, self.keep_rate)

        # Members: actual training pairs (ancs, poss).
        u_m, i_m = ue_cf[ancs], ie_cf[poss]
        # Non-members: random user shuffle vs random items (likely not training).
        # negs already provides random non-positive items per anc; use them.
        u_n, i_n = ue_cf[ancs], ie_cf[negs]

        # Apply gradient reversal so that disc gets normal gradient (learns to
        # discriminate) but encoder gets *negated* gradient (learns to fool).
        feat_m = grad_reverse(self._disc_features(u_m, i_m), self.adv_lambda)
        feat_n = grad_reverse(self._disc_features(u_n, i_n), self.adv_lambda)

        logit_m = self.disc(feat_m).squeeze(-1)
        logit_n = self.disc(feat_n).squeeze(-1)
        # Disc wants to label members=1, non-members=0
        bce = t.nn.functional.binary_cross_entropy_with_logits
        adv_loss = bce(logit_m, t.ones_like(logit_m)) \
                   + bce(logit_n, t.zeros_like(logit_n))
        adv_loss = adv_loss * self.adv_weight
        loss = loss + adv_loss
        losses['adv_loss'] = adv_loss.detach()
        return loss, losses

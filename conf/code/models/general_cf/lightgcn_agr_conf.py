"""LightGCN_AGR with confidence regularization (Plan C variant).

Key idea (MemGuard-style): MIA exploits the model's over-confidence on training
pairs. Standard BPR pushes pos_score - neg_score -> +inf, which is exactly the
signal the attacker uses. We add a soft target that *bounds* the sigmoid
confidence on pos pairs:

    L_conf = (sigmoid(pos_score) - conf_target)^2 averaged over batch

With conf_target ~= 0.7, the model learns to keep predictions within a
reasonable range while still preserving the *relative* order between pos/neg
pairs (BPR keeps ranking; conf reg keeps magnitudes bounded).

Knobs (all in YAML, all forwardable through CLI overrides):
    conf_weight   : multiplier on L_conf (default 0 = no defense)
    conf_target   : sigmoid target on pos training pairs (default 0.7)
    conf_neg_target: sigmoid target on neg pairs (default 0.5)
"""
import torch as t
import torch.nn.functional as F

from config.configurator import configs
from models.general_cf.lightgcn_agr import LightGCN_AGR


class LightGCN_AGR_Conf(LightGCN_AGR):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        cfg = configs['model']
        ds_cfg = cfg.get(self.dataset, {})

        def _get(key, default):
            if key in ds_cfg:
                return ds_cfg[key]
            return cfg.get(key, default)

        self.conf_weight = float(_get('conf_weight', 0.0))
        self.conf_target = float(_get('conf_target', 0.7))
        self.conf_neg_target = float(_get('conf_neg_target', 0.5))
        # LLM-aware confidence target (Phase 10 stretch):
        # if alpha > 0, target = clamp(base_target + alpha * cos(u_LLM, i_LLM), 0.05, 0.95)
        # so semantically-similar pairs allow higher confidence and dissimilar pairs
        # are penalized for confidence. Set alpha=0 to disable (default).
        self.conf_target_llm_alpha = float(_get('conf_target_llm_alpha', 0.0))

    def _llm_aware_target(self, base_target, u_idx, i_idx):
        """target_uvi = clamp(base + alpha * cos(u_LLM, i_LLM), 0.05, 0.95)."""
        if self.conf_target_llm_alpha <= 0:
            return base_target
        u_l = self.usrprf_embeds[u_idx]
        i_l = self.itmprf_embeds[i_idx]
        u_n = u_l / (u_l.norm(dim=-1, keepdim=True) + 1e-8)
        i_n = i_l / (i_l.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (u_n * i_n).sum(-1)  # [batch], in [-1, 1]
        target = base_target + self.conf_target_llm_alpha * cos
        return target.clamp(0.05, 0.95)

    def cal_loss(self, batch_data):
        loss, losses = super().cal_loss(batch_data)
        if self.conf_weight <= 0:
            return loss, losses
        ancs, poss, negs = batch_data
        ue_cf, ie_cf = self.forward(self.adj, self.keep_rate)
        u = ue_cf[ancs]
        ip = ie_cf[poss]
        in_ = ie_cf[negs]
        pos_score = (u * ip).sum(-1)
        neg_score = (u * in_).sum(-1)
        pos_prob = t.sigmoid(pos_score)
        neg_prob = t.sigmoid(neg_score)
        pos_target = self._llm_aware_target(self.conf_target, ancs, poss)
        neg_target = self._llm_aware_target(self.conf_neg_target, ancs, negs)
        conf_loss = ((pos_prob - pos_target) ** 2).mean() \
                    + ((neg_prob - neg_target) ** 2).mean()
        conf_loss = conf_loss * self.conf_weight
        loss = loss + conf_loss
        losses['conf_loss'] = conf_loss.detach()
        return loss, losses

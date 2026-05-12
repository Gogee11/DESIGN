"""Load an existing checkpoint, evaluate test-set recall under inference-time
perturbation (priv noise std + score temperature), and run MIA. Produces a
single jsonl record joining recall and MIA AUC under the same defense.

Used to fairly compare test-time-only defenses where we don't retrain.

Usage:
    python -m scripts.eval_only \
        --src_model lightgcn_agr --ckpt checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025.pth \
        --priv_noise_std 0.2 --score_temperature 1.0 \
        --tag agr_eval_n0.2 --out_jsonl results/phase4d_eval_only.jsonl
"""
import argparse
import json
import os
import sys

import torch

from config.configurator import configs
from load_data.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.metrics import Metric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--src_model', type=str, default='lightgcn_agr',
                   help='Model class to instantiate via build_model. The ckpt state_dict must match.')
    p.add_argument('--priv_noise_std', type=float, default=0.0)
    p.add_argument('--score_temperature', type=float, default=1.0)
    p.add_argument('--normalize_at_predict', type=int, default=1)
    p.add_argument('--tag', type=str, default='eval')
    p.add_argument('--out_jsonl', type=str, default='results/eval_only.jsonl')
    p.add_argument('--skip_mia', action='store_true')
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    # Coerce model name in configs to src_model so build_model picks the right class.
    configs['model']['name'] = args.src_model.lower()

    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs['device'])

    state = torch.load(args.ckpt, map_location=configs['device'], weights_only=False)
    model.load_state_dict(state, strict=False)
    print(f"[eval_only] loaded ckpt {args.ckpt} into {args.src_model}")

    # Patch the forward path to apply test-time normalization + noise + scaling.
    orig_forward = model.forward
    do_normalize = bool(args.normalize_at_predict)
    noise_std = args.priv_noise_std
    T = args.score_temperature

    # Patched forward must keep the parent's named keyword args so MIA's signature
    # introspection works. We accept the same kwargs as LightGCN/AGR forward.
    def patched_forward(adj=None, keep_rate=1.0, **kw):
        ue, ie = orig_forward(adj=adj, keep_rate=keep_rate, **kw)
        if not getattr(model, 'is_training', False):
            if do_normalize:
                ue = torch.nn.functional.normalize(ue, p=2, dim=-1)
                ie = torch.nn.functional.normalize(ie, p=2, dim=-1)
            if noise_std > 0:
                ue = ue + torch.randn_like(ue) * noise_std
                ie = ie + torch.randn_like(ie) * noise_std
            if T != 1.0:
                ue = ue / T
                ie = ie / T
        return ue, ie

    model.forward = patched_forward
    model.eval()
    if hasattr(model, 'is_training'):
        model.is_training = False
    if hasattr(model, 'final_embeds'):
        model.final_embeds = None

    metric = Metric()
    test_result = metric.eval(model, data_handler.test_dataloader)

    record = {
        'kind': 'eval_only',
        'model': args.src_model,
        'dataset': configs['data']['name'],
        'seed': configs['train']['seed'],
        'tag': args.tag,
        'priv_noise_std': args.priv_noise_std,
        'score_temperature': args.score_temperature,
        'normalize_at_predict': bool(args.normalize_at_predict),
        'k_list': list(configs['test'].get('k', [5, 10, 20])),
    }
    ks = record['k_list']
    for metric_name, vals in test_result.items():
        if hasattr(vals, 'tolist'):
            vals = vals.tolist()
        for i, k in enumerate(ks):
            record[f'{metric_name}@{k}'] = float(vals[i])

    print(f"[eval_only] recall={test_result.get('recall')}  ndcg={test_result.get('ndcg')}")

    # Also run MIA on the perturbed embeddings (using forward-patched model).
    if not args.skip_mia:
        try:
            from attack.MIA import (build_attack_edges, build_features,
                                     get_inference_embeddings, train_attacker)
            class _AttackArgs:
                pass
            aa = _AttackArgs()
            aa.member_limit = 50000
            aa.nonmember_limit = 50000
            aa.nonmember_source = 'mixed'
            aa.test_size = 0.3
            aa.attack_seed = 42
            aa.attacker = 'lr'
            (mu, mi), (nu, ni) = build_attack_edges(data_handler, aa)
            ue, ie = get_inference_embeddings(model)
            f_m = build_features(ue, ie, mu, mi, data_handler.trn_mat)
            f_n = build_features(ue, ie, nu, ni, data_handler.trn_mat)
            import numpy as np
            feats = np.concatenate([f_m, f_n], axis=0)
            labels = np.concatenate([np.ones(len(f_m), np.int64),
                                      np.zeros(len(f_n), np.int64)], axis=0)
            metrics = train_attacker(feats, labels, aa)
            for k in ('auc', 'acc', 'precision', 'recall', 'f1'):
                record['mia_' + k] = metrics.get(k)
            print(f"[eval_only] MIA AUC={metrics['auc']:.4f}")
        except Exception as e:
            print(f"[eval_only] MIA failed: {e}")
            record['mia_error'] = str(e)

    os.makedirs(os.path.dirname(args.out_jsonl) or '.', exist_ok=True)
    with open(args.out_jsonl, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"[eval_only] appended -> {args.out_jsonl}")


if __name__ == '__main__':
    main()

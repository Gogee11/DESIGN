"""Run a single experiment cell: train -> save ckpt -> run MIA -> append metrics jsonl.

This is the unit invoked by scripts/sweep.py for each grid point. It can also be
invoked directly. Each run is fully self-contained (re-imports configurator), so
multiple instances can run concurrently on the same GPU.

Example:
    python -m scripts.run_experiment \\
        --model lightgcn_agr --dataset digital_music --seed 2025 \\
        --tag b4_p025 --beta 4.0 --prf_weight 0.025 \\
        --results_jsonl results/sweep.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--tag', type=str, default='default',
                   help="Tag distinguishes runs of the same model+dataset+seed")
    p.add_argument('--cuda', type=str, default='0')
    p.add_argument('--results_jsonl', type=str, default='results/results.jsonl')
    # forwarded HP overrides
    for k in ('prf_weight', 'str_weight', 'recon_weight', 'beta', 'sigma',
             'reg_weight', 'keep_rate', 'mask_ratio', 'kd_temperature',
             'priv_noise_std', 'conf_weight', 'conf_target', 'conf_neg_target',
             'conf_target_llm_alpha'):
        p.add_argument(f'--{k}', type=float, default=None)
    p.add_argument('--aug_top_k', type=int, default=None)
    p.add_argument('--normalize_at_predict', type=int, default=None)
    p.add_argument('--embedding_size', type=int, default=None)
    p.add_argument('--adv_weight', type=float, default=None)
    p.add_argument('--adv_lambda', type=float, default=None)
    p.add_argument('--epoch', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--skip_train', action='store_true',
                   help="Use existing ckpt, only run MIA")
    p.add_argument('--skip_mia', action='store_true')
    p.add_argument('--attacker', type=str, choices=['lr', 'mlp'], default='lr')
    p.add_argument('--use_dp', action='store_true',
                   help="Train via main_dp.py (DP-SGD) instead of main.py")
    return p.parse_args()


def hp_args(args):
    out = []
    for k in ('prf_weight', 'str_weight', 'recon_weight', 'beta', 'sigma',
             'reg_weight', 'keep_rate', 'mask_ratio', 'kd_temperature',
             'aug_top_k', 'epoch', 'batch_size', 'patience',
             'priv_noise_std', 'normalize_at_predict', 'embedding_size',
             'conf_weight', 'conf_target', 'conf_neg_target', 'conf_target_llm_alpha',
             'adv_weight', 'adv_lambda'):
        v = getattr(args, k, None)
        if v is not None:
            out += [f'--{k}', str(v)]
    return out


def ckpt_path_for(args):
    suffix = f"-{args.tag}" if args.tag and args.tag != 'default' else ''
    return os.path.join(
        'checkpoint', args.model,
        f"{args.model}-{args.dataset}-{args.seed}{suffix}.pth")


def run_train(args):
    entry = 'main_dp.py' if args.use_dp else 'main.py'
    cmd = [sys.executable, entry,
           '--model', args.model,
           '--dataset', args.dataset,
           '--seed', str(args.seed),
           '--cuda', args.cuda,
           '--tag', args.tag] + hp_args(args)
    print('[run_experiment] TRAIN:', ' '.join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, env=os.environ.copy())
    dt = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"train failed rc={res.returncode}")
    print(f"[run_experiment] train done in {dt:.1f}s")
    return dt


def run_mia(args):
    ckpt = ckpt_path_for(args)
    if not os.path.exists(ckpt):
        # try variants for mlp re-runs (strip _mlp suffix from tag) and un-tagged
        candidates = []
        base_tag = args.tag.replace('_mlp', '') if args.tag.endswith('_mlp') else args.tag
        if base_tag and base_tag != 'default':
            candidates.append(os.path.join('checkpoint', args.model,
                                            f"{args.model}-{args.dataset}-{args.seed}-{base_tag}.pth"))
        candidates.append(os.path.join('checkpoint', args.model,
                                        f"{args.model}-{args.dataset}-{args.seed}.pth"))
        for c in candidates:
            if os.path.exists(c):
                ckpt = c
                break
    cmd = [sys.executable, '-m', 'attack.MIA',
           '--model', args.model,
           '--dataset', args.dataset,
           '--seed', str(args.seed),
           '--cuda', args.cuda,
           '--checkpoint_path', ckpt,
           '--out_json', args.results_jsonl,
           '--attacker', args.attacker,
           '--tag', args.tag] + hp_args(args)
    print('[run_experiment] MIA:', ' '.join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, env=os.environ.copy())
    dt = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"MIA failed rc={res.returncode}")
    print(f"[run_experiment] MIA done in {dt:.1f}s")
    return dt


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.results_jsonl) or '.', exist_ok=True)
    rec = {'model': args.model, 'dataset': args.dataset, 'seed': args.seed,
           'tag': args.tag, 'start': time.time()}
    try:
        if not args.skip_train:
            rec['train_seconds'] = run_train(args)
        if not args.skip_mia:
            rec['mia_seconds'] = run_mia(args)
        rec['status'] = 'ok'
    except Exception as e:
        rec['status'] = 'error'
        rec['error'] = repr(e)
        with open(args.results_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        raise
    print(f"[run_experiment] OK {args.model}/{args.dataset}/{args.tag}")


if __name__ == '__main__':
    main()

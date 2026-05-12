"""Run a hyperparameter sweep with bounded parallelism on a single GPU.

Each grid point spawns a subprocess invocation of scripts.run_experiment, which
itself spawns main.py + attack.MIA. We use ThreadPoolExecutor with N workers
(default 3) -- threads here are fine because each worker waits on a subprocess.

Edit the GRIDS dict below to change what is swept; or pass --grid <preset>.

Output: results/<run_name>.jsonl  (one record per (train, MIA) pair)
"""
import argparse
import concurrent.futures as cf
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', type=str, required=True,
                   choices=['baseline', 'sweep_beta', 'sweep_prf', 'sweep_str',
                            'pareto', 'all_baselines', 'smoke',
                            'multiseed_best', 'aug_topk', 'dp_ref',
                            'attacker_mlp_rerun'])
    p.add_argument('--datasets', type=str, nargs='+',
                   default=['digital_music'])
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--cuda', type=str, default='0')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--out_jsonl', type=str, default=None)
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--epoch', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    return p.parse_args()


def cells_for_grid(grid_name, datasets, seed):
    """Returns list of dicts, each describing a run."""
    out = []
    if grid_name == 'baseline' or grid_name == 'all_baselines':
        for ds in datasets:
            out.append({'model': 'lightgcn', 'dataset': ds, 'seed': seed,
                        'tag': 'baseline'})
            out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': seed,
                        'tag': 'default'})
        return out
    if grid_name == 'smoke':
        # one tiny run per ds: 100 epoch limit, keep_rate stays 1.0
        for ds in datasets:
            out.append({'model': 'lightgcn', 'dataset': ds, 'seed': seed,
                        'tag': 'smoke', '_extra_args': ['--epoch', '60', '--patience', '3']})
            out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': seed,
                        'tag': 'smoke', '_extra_args': ['--epoch', '60', '--patience', '3']})
        return out
    # Defaults (digital_music): beta=1.0, prf=0.002, str=0.05.
    # Sweeps span 0x .. ~10x default.
    if grid_name == 'sweep_beta':
        for ds in datasets:
            for beta in [0.0, 0.5, 1.0, 4.0]:
                out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': seed,
                            'tag': f'beta{beta}', 'beta': beta})
        return out
    if grid_name == 'sweep_prf':
        for ds in datasets:
            for prf in [0.0, 0.001, 0.002, 0.01]:
                out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': seed,
                            'tag': f'prf{prf}', 'prf_weight': prf})
        return out
    if grid_name == 'sweep_str':
        for ds in datasets:
            for sw in [0.0, 0.02, 0.05, 0.2]:
                out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': seed,
                            'tag': f'str{sw}', 'str_weight': sw})
        return out
    if grid_name == 'pareto':
        # 3x3 grid of (beta, str_weight) at default prf -- the two strongest knobs
        for ds in datasets:
            for beta, sw in itertools.product(
                    [0.0, 1.0, 4.0], [0.0, 0.05, 0.2]):
                out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': seed,
                            'tag': f'b{beta}_s{sw}', 'beta': beta, 'str_weight': sw})
        return out
    if grid_name == 'multiseed_best':
        # Re-run the Pareto-best point (PRF-only AGR) with extra seeds for variance
        for ds in datasets:
            for sd in [2026, 2027]:
                out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': sd,
                            'tag': 'b0.0_s0.0', 'beta': 0.0, 'str_weight': 0.0})
            # also extra seeds for LightGCN baseline and AGR default for fair multi-seed
            for sd in [2026, 2027]:
                out.append({'model': 'lightgcn', 'dataset': ds, 'seed': sd,
                            'tag': 'baseline'})
                out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': sd,
                            'tag': 'default'})
        return out
    if grid_name == 'aug_topk':
        # Ablation of LLM kNN augmented graph: vary aug_top_k from 0 (cf==aug) up
        for ds in datasets:
            for k in [0, 5, 20]:
                out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': seed,
                            'tag': f'augk{k}', 'aug_top_k': k})
        return out
    if grid_name == 'dp_ref':
        # DP-SGD reference (the failed-innovation baseline). Single noise level.
        for ds in datasets:
            out.append({'model': 'lightgcn_agr', 'dataset': ds, 'seed': seed,
                        'tag': 'dpref_n1.0', '_dp': True,
                        '_extra_args': []})
        return out
    if grid_name == 'attacker_mlp_rerun':
        # Re-run MIA on the existing ckpts with an MLP attacker (no retrain)
        for ds in datasets:
            for cell in [('lightgcn', 'baseline'),
                          ('lightgcn_agr', 'default'),
                          ('lightgcn_agr', 'b0.0_s0.0'),
                          ('lightgcn_agr', 'b1.0_s0.0'),
                          ('lightgcn_agr', 'b4.0_s0.2')]:
                out.append({'model': cell[0], 'dataset': ds, 'seed': seed,
                            'tag': cell[1], '_mlp_only': True})
        return out
    raise ValueError(f"Unknown grid {grid_name}")


def cell_to_cmd(cell, args, results_jsonl):
    cmd = [sys.executable, '-m', 'scripts.run_experiment',
           '--model', cell['model'],
           '--dataset', cell['dataset'],
           '--seed', str(cell['seed']),
           '--cuda', args.cuda,
           '--tag', cell['tag'],
           '--results_jsonl', results_jsonl]
    for k in ('beta', 'prf_weight', 'str_weight', 'recon_weight', 'sigma',
              'reg_weight', 'keep_rate', 'mask_ratio', 'kd_temperature',
              'aug_top_k', 'priv_noise_std', 'normalize_at_predict'):
        if k in cell:
            cmd += [f'--{k}', str(cell[k])]
    if args.epoch is not None:
        cmd += ['--epoch', str(args.epoch)]
    if args.patience is not None:
        cmd += ['--patience', str(args.patience)]
    if cell.get('_dp'):
        cmd += ['--use_dp']
    if cell.get('_mlp_only'):
        cmd += ['--skip_train', '--attacker', 'mlp']
        # rewrite tag to reflect mlp attacker variant for distinct jsonl record
        for i, t in enumerate(cmd):
            if t == '--tag' and i + 1 < len(cmd):
                cmd[i + 1] = cmd[i + 1] + '_mlp'
                break
    if '_extra_args' in cell:
        cmd += cell['_extra_args']
    return cmd


def run_one(cmd):
    print(f"[sweep] start: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    rc = res.returncode
    tail_stdout = '\n'.join((res.stdout or '').splitlines()[-30:])
    tail_stderr = '\n'.join((res.stderr or '').splitlines()[-30:])
    print(f"[sweep] done rc={rc} dt={dt:.1f}s :: {' '.join(cmd[-6:])}")
    if rc != 0:
        print(f"[sweep] STDOUT tail:\n{tail_stdout}")
        print(f"[sweep] STDERR tail:\n{tail_stderr}")
    return {'cmd': cmd, 'returncode': rc, 'seconds': dt,
            'stderr_tail': tail_stderr if rc != 0 else ''}


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.grid}_{datetime.now().strftime('%m%d_%H%M')}"
    out_jsonl = args.out_jsonl or os.path.join('results', f'{run_name}.jsonl')
    os.makedirs(os.path.dirname(out_jsonl) or '.', exist_ok=True)

    cells = cells_for_grid(args.grid, args.datasets, args.seed)
    cmds = [cell_to_cmd(c, args, out_jsonl) for c in cells]
    print(f"[sweep] grid={args.grid} datasets={args.datasets} cells={len(cells)} workers={args.workers}")
    print(f"[sweep] results -> {out_jsonl}")

    summary = []
    summary_path = out_jsonl.replace('.jsonl', '.summary.jsonl')
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, c) for c in cmds]
        for fut in cf.as_completed(futures):
            r = fut.result()
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({k: v for k, v in r.items() if k != 'stderr_tail'},
                                   ensure_ascii=False) + '\n')
            summary.append(r)
    n_ok = sum(1 for r in summary if r['returncode'] == 0)
    n_fail = len(summary) - n_ok
    print(f"\n[sweep] OK {n_ok}/{len(summary)}, FAIL {n_fail}")
    print(f"[sweep] details: {summary_path}")
    print(f"[sweep] metrics: {out_jsonl}")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

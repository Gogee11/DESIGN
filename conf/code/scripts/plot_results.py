"""Read results jsonl and write summary tables + Pareto plots.

Usage:
    python -m scripts.plot_results --in results/sweep.jsonl --out_dir results/figs
"""
import argparse
import json
import os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='results/figs')
    return p.parse_args()


def load_records(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('auc') is None:
                continue
            rows.append(r)
    return rows


def write_table(rows, out_path):
    keys = ['dataset', 'model', 'tag', 'seed', 'auc', 'acc', 'f1',
            'precision', 'recall']
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(keys) + '\n')
        for r in rows:
            f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')


def maybe_plot(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}; skipping plots")
        return
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r['dataset']].append(r)
    os.makedirs(out_dir, exist_ok=True)
    for ds, items in by_ds.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in items:
            label = f"{r['model']}/{r.get('tag', '')}"
            ax.scatter(r['auc'], r.get('acc', 0), s=40, label=label)
        ax.set_xlabel('MIA AUC (higher = more privacy leakage)')
        ax.set_ylabel('MIA ACC')
        ax.set_title(f'Privacy-utility raw: {ds}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best', ncol=2)
        out = os.path.join(out_dir, f'mia_{ds}.png')
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot] saved {out}")


def main():
    args = parse_args()
    rows = load_records(args.in_path)
    print(f"[plot] loaded {len(rows)} attack records from {args.in_path}")
    if not rows:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    table_path = os.path.join(args.out_dir, 'summary.tsv')
    write_table(rows, table_path)
    print(f"[plot] wrote {table_path}")
    maybe_plot(rows, args.out_dir)


if __name__ == '__main__':
    main()

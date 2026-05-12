"""Read the combined results.jsonl and emit a Markdown report + Pareto plot.

Output:
    results/REPORT.md       - human-readable summary
    results/figs/pareto_<dataset>.png  - scatter MIA AUC vs recall@20
    results/figs/sweep_<which>.png     - line plots of per-axis sweeps
"""
import argparse
import json
import os
from collections import defaultdict


def load_records(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows.append(r)
    return rows


def join_utility_to_mia(rows):
    """Join MIA records with their utility records (same model+dataset+seed+tag)."""
    util = {}
    mia = []
    for r in rows:
        if r.get('kind') == 'utility' or 'recall@20' in r and 'auc' not in r:
            key = (r['model'], r['dataset'], r['seed'], r.get('tag', 'default'))
            util[key] = r
        elif r.get('kind') == 'mia' or 'auc' in r:
            mia.append(r)
    joined = []
    for r in mia:
        key = (r['model'], r['dataset'], r['seed'], r.get('tag', 'default'))
        u = util.get(key, {})
        merged = dict(r)
        for k, v in u.items():
            if k not in merged:
                merged[k] = v
        joined.append(merged)
    return joined


def fmt(v, prec=4):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.{prec}f}'
    return str(v)


def write_report(rows, out_md):
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r['dataset']].append(r)
    lines = []
    lines.append('# LLM-AGR Privacy-Utility Study\n')
    lines.append('Auto-generated from results jsonl.\n')
    for ds, items in by_ds.items():
        lines.append(f'\n## Dataset: `{ds}` ({len(items)} runs)\n')
        items_sorted = sorted(items, key=lambda r: (r['model'], r.get('tag', '')))
        # Pull baseline as a reference if present
        lgcn = [r for r in items_sorted if r['model'] == 'lightgcn']
        agr_default = [r for r in items_sorted if r['model'] == 'lightgcn_agr' and r.get('tag', '') in ('default',)]
        ref_lgcn = lgcn[0] if lgcn else None
        ref_agr = agr_default[0] if agr_default else None

        if ref_lgcn or ref_agr:
            lines.append('### Baseline comparison')
            lines.append('| Model | tag | recall@20 | ndcg@20 | MIA AUC | MIA ACC |')
            lines.append('|---|---|---|---|---|---|')
            for r in [ref_lgcn, ref_agr]:
                if r:
                    lines.append('| {} | {} | {} | {} | {} | {} |'.format(
                        r['model'], r.get('tag', ''),
                        fmt(r.get('recall@20')), fmt(r.get('ndcg@20')),
                        fmt(r.get('auc')), fmt(r.get('acc'))))
            if ref_lgcn and ref_agr:
                d_recall = (ref_agr.get('recall@20', 0) or 0) - (ref_lgcn.get('recall@20', 0) or 0)
                d_auc = (ref_agr.get('auc', 0) or 0) - (ref_lgcn.get('auc', 0) or 0)
                if abs(d_recall) > 1e-6 or abs(d_auc) > 1e-6:
                    lines.append('')
                    lines.append('AGR vs LightGCN: '
                                 f'Δrecall@20 = {d_recall:+.4f}, '
                                 f'ΔMIA_AUC = {d_auc:+.4f} (negative = better privacy)\n')

        lines.append('### All runs')
        lines.append('| Model | tag | recall@20 | ndcg@20 | MIA AUC | best_epoch |')
        lines.append('|---|---|---|---|---|---|')
        for r in items_sorted:
            lines.append('| {} | {} | {} | {} | {} | {} |'.format(
                r['model'], r.get('tag', ''),
                fmt(r.get('recall@20')), fmt(r.get('ndcg@20')),
                fmt(r.get('auc')),
                fmt(r.get('best_epoch'), prec=0)))

    md = '\n'.join(lines) + '\n'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'[analyze] wrote {out_md}')


def maybe_plot(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f'[analyze] matplotlib not available: {e}')
        return
    os.makedirs(out_dir, exist_ok=True)
    by_ds = defaultdict(list)
    for r in rows:
        if r.get('auc') is None or r.get('recall@20') is None:
            continue
        by_ds[r['dataset']].append(r)

    for ds, items in by_ds.items():
        # ---- Pareto scatter ----
        fig, ax = plt.subplots(figsize=(9, 6))
        for r in items:
            tag = r.get('tag', '')
            color = 'tab:blue' if r['model'] == 'lightgcn' else \
                    ('tab:green' if 'priv' in r['model'] else 'tab:red')
            marker = 'o'
            size = 100
            if 'baseline' in (tag or ''):
                marker, size = 'D', 140
            elif tag == 'default':
                marker, size = 's', 130
            ax.scatter(r['auc'], r['recall@20'], s=size, c=color, marker=marker,
                       edgecolors='black', linewidths=0.6, alpha=0.85)
            ax.annotate(f"{r['model'].replace('lightgcn_','').replace('lightgcn','LGCN')}/{tag}",
                        (r['auc'], r['recall@20']), fontsize=7,
                        textcoords='offset points', xytext=(5, 5))
        ax.set_xlabel('MIA AUC (lower = better privacy)')
        ax.set_ylabel('Test recall@20 (higher = better recommendation)')
        ax.set_title(f'Privacy-utility map: {ds}\n(top-left = Pareto-best)')
        ax.grid(True, alpha=0.3)
        # Don't invert -- keep natural reading: left = lower AUC = better privacy
        out = os.path.join(out_dir, f'pareto_{ds}.png')
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f'[analyze] saved {out}')

        # ---- Heatmap of AGR sweep cells ----
        agr_cells = [r for r in items if r['model'] == 'lightgcn_agr'
                      and r.get('tag', '').startswith('b')]
        # Parse tag like "b1.0_s0.05" -> beta=1.0, str=0.05
        def parse(tag):
            try:
                a, b = tag.split('_')
                return float(a[1:]), float(b[1:])
            except Exception:
                return None, None

        if len(agr_cells) >= 4:
            betas = sorted({parse(r['tag'])[0] for r in agr_cells if parse(r['tag'])[0] is not None})
            strs = sorted({parse(r['tag'])[1] for r in agr_cells if parse(r['tag'])[1] is not None})
            for metric in ('auc', 'recall@20'):
                grid = np.full((len(betas), len(strs)), np.nan)
                for r in agr_cells:
                    b, s = parse(r['tag'])
                    if b is None or s is None or r.get(metric) is None:
                        continue
                    i = betas.index(b); j = strs.index(s)
                    grid[i, j] = r[metric]
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(grid, cmap='viridis', aspect='auto')
                ax.set_xticks(range(len(strs)))
                ax.set_xticklabels([f'{s:g}' for s in strs])
                ax.set_yticks(range(len(betas)))
                ax.set_yticklabels([f'{b:g}' for b in betas])
                ax.set_xlabel('str_weight')
                ax.set_ylabel('beta (HSIC)')
                ax.set_title(f'{metric} on {ds} (LightGCN_AGR sweep)')
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        v = grid[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                                    color='white' if v < (np.nanmin(grid)+np.nanmax(grid))/2 else 'black',
                                    fontsize=9)
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                metric_safe = metric.replace('@', 'at')
                out = os.path.join(out_dir, f'heatmap_{ds}_{metric_safe}.png')
                fig.savefig(out, dpi=140)
                plt.close(fig)
                print(f'[analyze] saved {out}')


def normalize_record(r):
    """Promote alternate field names to canonical (auc, recall@20)."""
    if 'mia_auc' in r and r.get('auc') is None:
        r['auc'] = r['mia_auc']
    if 'mia_acc' in r and r.get('acc') is None:
        r['acc'] = r['mia_acc']
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_paths', type=str, nargs='+', default=None,
                   help='one or more results jsonl files; if omitted scans results/*.jsonl')
    p.add_argument('--out_md', type=str, default='results/REPORT.md')
    p.add_argument('--out_dir', type=str, default='results/figs')
    args = p.parse_args()

    all_rows = []
    paths = args.in_paths
    if not paths:
        # auto-discover phaseN_*.jsonl + phase*.jsonl in results/
        paths = []
        if os.path.isdir('results'):
            for fn in sorted(os.listdir('results')):
                if fn.endswith('.jsonl') and not fn.endswith('.summary.jsonl') \
                        and 'smoke' not in fn:
                    paths.append(os.path.join('results', fn))
    for p_in in paths:
        for r in load_records(p_in):
            all_rows.append(normalize_record(r))
    util_dir = 'results/utility'
    if os.path.isdir(util_dir):
        for fn in os.listdir(util_dir):
            if fn.endswith('.json'):
                with open(os.path.join(util_dir, fn), 'r', encoding='utf-8') as f:
                    try:
                        all_rows.append(normalize_record(json.load(f)))
                    except Exception:
                        pass
    print(f'[analyze] loaded {len(all_rows)} raw records from {len(paths)} files')
    joined = join_utility_to_mia(all_rows)
    # Also include eval_only-style records that are self-contained (have both R and AUC).
    for r in all_rows:
        if r.get('kind') == 'eval_only' and r.get('recall@20') is not None and r.get('auc') is not None:
            joined.append(r)
    print(f'[analyze] {len(joined)} runs with both utility+MIA')

    write_report(joined, args.out_md)
    maybe_plot(joined, args.out_dir)


if __name__ == '__main__':
    main()

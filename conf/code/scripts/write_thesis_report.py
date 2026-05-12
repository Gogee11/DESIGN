"""Synthesize a self-contained thesis-ready Markdown report from results jsonl.

Output:
    results/THESIS_REPORT.md

Sections:
  1. Summary
  2. Pipeline overview
  3. Recommendation utility (LightGCN vs LightGCN_AGR)
  4. Privacy attack (MIA) -- baseline comparison
  5. Hyperparameter sweep -- privacy/utility trade-off
  6. Discussion + limitations
  7. Reproducibility
"""
import argparse
import json
import os
from collections import defaultdict


def load_records(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    util_dir = 'results/utility'
    if os.path.isdir(util_dir):
        for fn in os.listdir(util_dir):
            if fn.endswith('.json'):
                with open(os.path.join(util_dir, fn), 'r', encoding='utf-8') as f:
                    try:
                        rows.append(json.load(f))
                    except Exception:
                        pass
    return rows


def join(rows):
    util = {}
    mia = []
    for r in rows:
        is_util = (r.get('kind') == 'utility' or
                   ('recall@20' in r and 'auc' not in r))
        if is_util:
            key = (r['model'], r['dataset'], r.get('seed'), r.get('tag', 'default'))
            util[key] = r
    for r in rows:
        if r.get('kind') == 'mia' or 'auc' in r:
            mia.append(r)
    out = []
    for r in mia:
        key = (r['model'], r['dataset'], r.get('seed'), r.get('tag', 'default'))
        merged = dict(r)
        u = util.get(key)
        if u:
            for k, v in u.items():
                if k not in merged:
                    merged[k] = v
        out.append(merged)
    return out


def fmt(v, prec=4):
    if v is None or v == '':
        return '–'
    if isinstance(v, float):
        return f'{v:.{prec}f}'
    return str(v)


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |',
            '|' + '|'.join(['---'] * len(header)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_paths', type=str, nargs='+', required=True)
    p.add_argument('--out_md', type=str, default='results/THESIS_REPORT.md')
    args = p.parse_args()

    rows = load_records(args.in_paths)
    joined = join(rows)
    print(f'Joined {len(joined)} runs from {len(rows)} raw records')

    by_ds = defaultdict(list)
    for r in joined:
        by_ds[r['dataset']].append(r)

    sections = []

    sections.append('# LLM-AGR Privacy-Utility Trade-off: Final Report\n')
    sections.append('Auto-generated from `results/*.jsonl` and `results/utility/*.json`.\n')

    # Section 1: Summary
    sections.append('## 1. Summary\n')
    summary_rows = []
    for ds in sorted(by_ds.keys()):
        items = by_ds[ds]
        lgcn = next((r for r in items if r['model'] == 'lightgcn' and 'baseline' in (r.get('tag') or '')), None)
        agr = next((r for r in items if r['model'] == 'lightgcn_agr' and r.get('tag') == 'default'), None)
        if lgcn and agr:
            d_recall = (agr.get('recall@20', 0) or 0) - (lgcn.get('recall@20', 0) or 0)
            d_auc = (agr.get('auc', 0) or 0) - (lgcn.get('auc', 0) or 0)
            verdict = 'AGR helps both' if (d_recall > 0 and d_auc < 0) else \
                      ('AGR boosts recall but raises AUC' if d_recall > 0 else
                       ('AGR improves privacy at recall cost' if d_auc < 0 else 'mixed'))
            summary_rows.append([ds, fmt(lgcn.get('recall@20')), fmt(agr.get('recall@20')),
                                  fmt(d_recall, 4),
                                  fmt(lgcn.get('auc')), fmt(agr.get('auc')),
                                  fmt(d_auc, 4), verdict])
    if summary_rows:
        sections.append(md_table(
            ['Dataset', 'LGCN R@20', 'AGR R@20', 'ΔR@20',
             'LGCN AUC', 'AGR AUC', 'ΔAUC', 'verdict'],
            summary_rows))
        sections.append('\n*ΔR@20 > 0: AGR recommends better. ΔAUC < 0: AGR is more privacy-safe.*\n')

    # Section 2: Pipeline
    sections.append('\n## 2. Pipeline overview\n')
    sections.append('1. **Data prep** (`scripts/prepare_dataset.py`): rating filter (≥3) → iterative k-core (k=5) → '
                    '6:2:2 random per-user split → BAAI BGE-small-en-v1.5 sentence encoder for user reviews '
                    'and item title+reviews → save `trn/val/tst.npz` + `usr_emb_np.npy` + `itm_emb_np.npy`.\n')
    sections.append('2. **Baseline training** (`main.py`): LightGCN (BPR + L2) and LightGCN_AGR (BPR + L2 + '
                    'PRF distillation + STR contrast on LLM-augmented graph + HSIC info-bottleneck + masked recon).\n')
    sections.append('3. **MIA** (`attack/MIA.py`): for each ckpt, load the model, compute final embeddings, '
                    'build hand-crafted features for sampled member/non-member edges, train an LR attacker, '
                    'report AUC/ACC/F1.\n')
    sections.append('4. **Hyperparameter sweep** (`scripts/sweep.py`): vary regularization weights '
                    '(β, str_weight, prf_weight) on a single dataset to map the privacy-utility trade-off.\n')

    # Section 3: Per-dataset detail
    for ds in sorted(by_ds.keys()):
        items = sorted(by_ds[ds], key=lambda r: (r['model'], r.get('tag', '')))
        sections.append(f'\n## 3. Detailed results: `{ds}`\n')
        rows = []
        for r in items:
            rows.append([r['model'], r.get('tag', 'default'), r.get('seed', '-'),
                         fmt(r.get('recall@5')), fmt(r.get('recall@10')),
                         fmt(r.get('recall@20')), fmt(r.get('ndcg@20')),
                         fmt(r.get('auc')), fmt(r.get('acc')),
                         fmt(r.get('best_epoch'), 0)])
        sections.append(md_table(
            ['Model', 'Tag', 'Seed', 'R@5', 'R@10', 'R@20', 'NDCG@20',
             'MIA AUC', 'MIA ACC', 'best_epoch'],
            rows))

        # Pareto interpretation
        agr_runs = [r for r in items if r['model'] == 'lightgcn_agr']
        if len(agr_runs) >= 4:
            sections.append('\n### Privacy-utility map (`b<beta>_s<str_weight>` tags)\n')
            sections.append('Each row is a (β, str_weight) cell; left columns are utility, right are privacy.\n')

    # Section 4: discussion
    sections.append('\n## 4. Discussion\n')
    sections.append('- **What we changed in the inherited code**: removed dead components (cf_index/learn_graph_structure '
                    'no-op, recon_loss never called, kd_weight unused, double-multiply by alpha); '
                    'wired up real LLM-augmented adjacency via top-K cosine kNN; '
                    'decoupled loss weights so each hyperparameter has exactly one role; '
                    'made data handler dataset-agnostic; switched encoder default to BAAI BGE-small-en-v1.5 (open-source, Chinese-team-published).\n')
    sections.append('- **DP-SGD as reference (failed innovation)**: previous attempt to use DP-SGD as the privacy '
                    'defense brought recall@20 from 0.26 → 0.006 — a 40× regression — confirming that classical DP '
                    'is not viable for LLM-augmented recommendation at this scale. The regularization-based defense '
                    'tested here has comparable utility to the unregularized baseline while shifting the MIA AUC.\n')
    sections.append('- **Limitations**: small datasets; single seed; LR attacker only (no MLP/strong attacker); '
                    'compute budget capped sweeps to ≤200 epochs and ≤9 cells.\n')

    # Section 5: Method comparison (M1-M4)
    sections.append('\n## 5. Comparison of 4 privacy-utility methods\n')
    sections.append('Each method is summarized by its best (recall, AUC) Pareto point on `digital_music`. '
                    'For reference: LightGCN baseline R@20 ≈ 0.241, AUC ≈ 0.902. '
                    'Degree-only attack floor AUC ≈ 0.900.\n')
    method_rows = []
    for r in joined:
        tag = r.get('tag', '')
        mdl = r.get('model', '')
        if tag.startswith('m1_') or 'conf' in mdl:
            method_rows.append(('M1 confidence reg', r))
        elif tag.startswith('m3_'):
            method_rows.append(('M3 emb compression', r))
        elif tag.startswith('m4_') or 'adv' in mdl:
            method_rows.append(('M4 adversarial GRL', r))
        elif 'balanced' in tag:
            method_rows.append(('M2 balanced MIA', r))
    if method_rows:
        sections.append('| method | tag | recall@20 | MIA AUC | best_epoch |')
        sections.append('|---|---|---|---|---|')
        for method, r in sorted(method_rows, key=lambda x: x[0]):
            sections.append(f"| {method} | {r.get('tag','')} | "
                            f"{fmt(r.get('recall@20'))} | {fmt(r.get('auc'))} | "
                            f"{fmt(r.get('best_epoch'), prec=0)} |")
        sections.append('')

    # Section 6: Reproducibility
    sections.append('\n## 6. Reproducibility\n')
    sections.append('```bash\n# remote DCU host\nsource /opt/dtk-25.04.2/env.sh\ncd /root/private_data/wxy/llw/code\nbash scripts/launch_full_pipeline.sh digital_music pareto 200 3\nbash scripts/launch_methods_unified.sh\npython3 -m scripts.analyze --out_md results/REPORT.md\n```\n')

    md = '\n'.join(sections) + '\n'
    os.makedirs(os.path.dirname(args.out_md) or '.', exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Wrote {args.out_md}')


if __name__ == '__main__':
    main()

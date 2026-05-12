# LLM-AGR Privacy-Utility Study (毕设项目)

This codebase trains LightGCN-based recommenders on Amazon datasets and
evaluates their susceptibility to **Membership Inference Attack (MIA)**, with
the central thesis that LightGCN-AGR's regularization knobs (HSIC IB, semantic
distillation, structural contrast on an LLM-augmented graph) provide a tunable
**privacy-utility trade-off** that is more practical than DP-SGD.

## Where things live

```
code/
├── main.py                  # Train a single model
├── main_dp.py               # Train with hand-rolled DP-SGD (failed-baseline reference)
├── attack/MIA.py            # Membership inference attack with LR/MLP classifier
├── config/
│   ├── configurator.py      # Parses CLI args and YAML, exposes `configs` global
│   └── models_config/
│       ├── lightgcn.yml             # Plain LightGCN
│       ├── lightgcn_agr.yml         # LLM-augmented + regularized (main innovation)
│       └── lightgcn_agr_priv.yml    # Plan B: AGR + explicit privacy hooks
├── load_data/data_handler_general_cf.py   # builds torch_adj + aug_torch_adj (LLM kNN)
├── models/general_cf/
│   ├── lightgcn.py
│   ├── lightgcn_agr.py
│   └── lightgcn_agr_priv.py
├── trainer/{trainer.py,dp_trainer.py,logger.py,...}
├── scripts/
│   ├── prepare_dataset.py     # raw jsonl -> trn/val/tst.npz + usr/itm_emb.npy
│   ├── download_amazon.py     # HF / McAuley download
│   ├── run_experiment.py      # 1 cell: train + MIA + jsonl record
│   ├── sweep.py               # parallel grid driver
│   ├── analyze.py             # produce report + plots from jsonl
│   ├── write_thesis_report.py # Markdown report
│   ├── launch_full_pipeline.sh # phase 1 (baselines) + phase 2 (sweep) + phase 3 (Plan B) + phase 4 (analyze)
│   └── launch_planb.sh         # standalone Plan B
└── data/<dataset>/             # trn/val/tst.npz, usr/itm_emb_np.npy/.pkl, *_ids.pkl
```

## Models compared

| name             | what it does                                                 |
|------------------|--------------------------------------------------------------|
| `lightgcn`       | Plain LightGCN with BPR + L2.                                |
| `lightgcn_agr`   | + LLM kNN-augmented adjacency, semantic distillation, structural contrast, HSIC IB. |
| `lightgcn_agr_priv` | + L2-normalized inference embeddings + Gaussian noise on inference. **Plan B if AGR doesn't beat baseline privacy.** |

## Loss components in `lightgcn_agr.py`

```
total = bpr_loss
      + reg_weight  * L2_norm
      + prf_weight  * InfoNCE(cf_emb,  LLM_mlp(prf_emb))   # semantic distillation
      + str_weight  * InfoNCE(cf_emb,  aug_emb)             # structural contrast
      + recon_weight * SSL_recon(masked_cf_emb, prf_emb)    # masked autoencoder (off by default)
      + beta        * HSIC(cf_emb, aug_emb)                 # information bottleneck
```
Each weight gates **one** loss term — set to 0 to ablate.

## Inherited bugs we removed

| issue | what was wrong | fix |
|---|---|---|
| `cf_index` no-op | `learn_graph_structure` returned `self.adj` because `cf_index` was never set | removed entire `learn_graph_structure` path |
| `aug_adj == adj` | `data_handler.aug_torch_adj` was never built; "structural" view fell back to the cf graph | added `_build_aug_bipartite()` using top-K cosine over LLM embeddings |
| dead `_reconstruction` | method was defined but never called from `cal_loss` | wired into `cal_loss` gated by `recon_weight` |
| dead `kd_weight` | YAML key was unused in code | dropped from YAML; renamed actual gate to `prf_weight` (already used) |
| double weighting | `(prf_loss + str_loss) * alpha` then alpha-style multiplier shadowed prf/str weights | flat sum: each weight gates exactly one term |
| Windows paths | `D:/models/all-MiniLM-L6-v2` hardcoded in `pre/` | `--model` CLI arg + default to BAAI/bge-small-en-v1.5 |
| pickle cross-version | `usr_emb_np.pkl` produced by numpy 2.x couldn't load on numpy 1.x | re-saved as `.npy`, configurator prefers npy then falls back to pkl |

## Datasets

| name            | source                                              | num_users | num_items | trn edges |
|-----------------|-----------------------------------------------------|-----------|-----------|-----------|
| `digital_music` | Amazon Digital_Music_5 (legacy 2014)                | 5,541     | 3,568     | 37,045    |

Adding a new dataset:
```bash
python -m scripts.download_amazon --category Industrial_and_Scientific --dataset industrial_scientific
python -m scripts.prepare_dataset --dataset industrial_scientific \
    --raw data/industrial_scientific/industrial_scientific.jsonl \
    --model BAAI/bge-small-en-v1.5
```

## Running on the SCNet DCU host

```bash
ssh -p 10587 root@ssh.zzai2.scnet.cn   # password authenticated
cd /root/private_data/wxy/llw/code
source /opt/dtk-25.04.2/env.sh         # makes torch.cuda work (Hygon DCU)

# end-to-end
bash scripts/launch_full_pipeline.sh digital_music pareto 200 3
# -> results/phase1_baselines.jsonl, phase2_pareto.jsonl, phase3_planb.jsonl
# -> results/all_combined.jsonl, results/THESIS_REPORT.md, results/figs/*.png
```

Single cell:
```bash
python -m scripts.run_experiment \
    --model lightgcn_agr --dataset digital_music --seed 2025 --cuda 0 \
    --tag b1_s05 --beta 1.0 --str_weight 0.05 \
    --results_jsonl results/probe.jsonl
```

## Reading the results

Each row in `results/*.jsonl` has the recommendation utility (`recall@K`, `ndcg@K`)
joined with the MIA result (`auc`, `acc`, `precision`, `recall`, `f1`).

**Lower MIA `auc`** → harder for an attacker to guess membership → better privacy.

**Higher `recall@20`** → better recommendation utility.

The Pareto frontier on `pareto_<dataset>.png` shows the points that dominate
all others. Heatmaps (`heatmap_<dataset>_*.png`) show the effect of β / str_weight
on each metric.

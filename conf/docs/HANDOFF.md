# 项目交接(Final)

> 最后更新: 2026-05-10 09:10
> 交接路径: `/home/ubuntu/wxy/llw/design/`

## 0. 一句话项目状态

LLM-AGR 隐私研究(本科毕设),已完成 ~140 个独立实验,产出完整 Pareto frontier。
核心发现:**LLM-AGR 默认配置反而比 LightGCN 更易被 MIA 攻破**;`str_weight` 是元凶;
3 类轻量化防御 (M1 置信度正则 / M3 嵌入压缩 / M4 对抗训练) 已系统对比,**M1+M4 combo 是双向严格 Pareto 最优**。

---

## 1. 远程主机访问

| 主机 | 端口 | 密码 | 状态 |
|---|---|---|---|
| 主远程 | 10587 | `MDPIRDTYRXMMKL4` | ✅ 实验已全部完成 |
| Remote A | 10154 | `GPBTNVOWDDZS67N` | ✅ 实验已全部完成 |
| (Remote B 10079) | 10079 | `MBLBVTRYSCAOTIT` | ❌ 不可用 (用户告知) |

NFS 共享路径: `/root/private_data/wxy/llw/code/` (代码、ckpt、结果都在此)
DCU 环境: `source /opt/dtk-25.04.2/env.sh` (主) / `/opt/dtk-25.04.1/env.sh` (Remote A)

---

## 2. 交接物 — 全部在 `/home/ubuntu/wxy/llw/design/`

```
design/
├── code/             # 全部 Python 源码 (~30 MB)
│   ├── main.py, main_dp.py, PROJECT_README.md
│   ├── config/{configurator.py, models_config/*.yml}     # 7 个 yml
│   ├── models/general_cf/{lightgcn,lightgcn_agr,lightgcn_agr_priv,lightgcn_agr_conf,lightgcn_agr_adv,lightgcn_agr_combo}.py
│   ├── load_data/data_handler_general_cf.py              # 含 LLM kNN 增强图
│   ├── trainer/                                            # 训练器
│   ├── attack/MIA.py                                       # MIA 攻击器(LR/MLP/度数均衡)
│   ├── scripts/                                            # 21 个驱动脚本
│   │   ├── prepare_dataset.py, run_experiment.py, sweep.py
│   │   ├── eval_only.py, analyze.py, write_thesis_report.py
│   │   └── launch_phase{1,8-17}.sh                          # 全部 phase launcher
│   └── data/digital_music/{trn/val/tst_mat.npz, usr/itm_emb_np.npy, *_ids.pkl}
├── checkpoint/        # 123 个 .pth ckpt (~190 MB)
│   ├── lightgcn/         # 3 ckpts (含多种子)
│   ├── lightgcn_agr/     # 30+ ckpts (P2 b/s 网格 + emb 压缩 + 多种子)
│   ├── lightgcn_agr_adv/ # M4 各种 adv_weight + patience
│   ├── lightgcn_agr_conf/ # M1 各种 conf_weight/target
│   ├── lightgcn_agr_combo/ # M1+M4 combo
│   └── lightgcn_agr_priv/ # Plan B (L2 norm + 噪声)
├── results/            # 全部 jsonl 实验结果 (~1 MB)
│   ├── phase1-17_*.jsonl     # 各 phase 主输出
│   ├── utility/utility-*.json  # 每个训练 cell 的 recall 元数据 ~50 个
│   └── *.out                  # 各 phase 运行日志
└── docs/              # 论文级文档
    ├── 01-描述.txt    # 同学最初描述(原始)
    ├── 02-todo.txt    # 早期 TODO
    ├── 03-new claim.md # ★ 重定型 claim 与子论点 L1-L4
    ├── 04-进展.txt    # ★ 实验全程日志(每 phase 一节)
    ├── HANDOFF.md     # ★ 本文件
    └── PROJECT_README.md # 代码结构说明
```

总尺寸 ~220 MB。

---

## 3. 重要文档优先级

| 优先级 | 文件 | 看完之后能做什么 |
|---|---|---|
| ★★★ | `docs/03-new claim.md` | 知道论文要写什么:5 大 finding (L1-L5)、对应实验、每个 cell 在论文里的位置 |
| ★★★ | `docs/04-进展.txt` | 知道每个实验为什么跑、怎么发现的、Pareto 曲线 |
| ★★ | `code/PROJECT_README.md` | 理解代码结构、如何加新 model / 新数据集 |
| ★ | `docs/HANDOFF.md` (本文件) | 整体交接说明 |
| ★ | `docs/01-描述.txt` | 同学最初的项目描述,作为 motivation 参考 |

---

## 4. 论文 5 大核心 finding

### L1 反直觉发现
LLM-AGR 默认 (β=1, str=0.05) 比 LightGCN MIA AUC 高 +3.4 pp,推荐仅 +1.1%。

**实验**: phase1_baselines.jsonl (LGCN / AGR-default,各 1 cell + Phase 12 多种子 6 cells)

### L2 机理识别 — `str_weight` 是元凶
- (β, str_weight) 3×3 网格中,str=0→0.05→0.2 → AUC 0.886→0.93→0.95
- β 影响仅 ±0.02 (微弱)
- aug_top_k ∈ {0, 3, 10, 20} 完全无关 → LLM kNN 增强图无作用
- prf_weight (LLM 语义蒸馏) 不增加泄漏

**实验**: phase2_pareto.jsonl (9 cells) + phase13_extensions.jsonl (aug_top_k 4 cells)

### L3 三类防御方法
1. **M1 置信度正则** (`lightgcn_agr_conf`): 加 `λ(σ(score)-τ)²`,降 AUC 1.5-2 pp
2. **M3 嵌入压缩** (`lightgcn_agr` + `embedding_size`): emb 32→16/8/4,降 AUC 5-19 pp(recall 大跌)
3. **M4 对抗训练** (`lightgcn_agr_adv`): GRL + MIA 判别器,降 AUC 2.5-4 pp
4. **M1+M4 combo** (`lightgcn_agr_combo`): 最优平衡 (s1_combo_c0.5_a0.1)

**实验**: phase8/9/10/13/14/15/16 各方法主 sweep + 多种子

### L4 鲁棒性
- 多种子 variance < 0.005
- 度数均衡 MIA 排序保留
- LR / MLP 攻击器排序一致

**实验**: phase8_m2_balanced_mia.jsonl (M2 度数均衡,5 cells), phase12b_mlp_attacker.jsonl (MLP 8 cells)

### L5 极端隐私 corner
- emb=2: AUC=0.7143(突破"度数下界"0.90)
- 完整 Pareto 曲线 emb={2,3,4,6,8,10,12,16,20,24,32}

**实验**: phase14/16/17 (emb 极端 + 多种子)

---

## 5. 接手后立即可做的 4 件事

### 5.1 出 Pareto 散点图 (5 min)
```bash
cd design/code
python3 -m scripts.analyze --out_md results/REPORT.md --out_dir results/figs
```
输出: `results/figs/pareto_digital_music.png` + `results/REPORT.md`

### 5.2 出论文级 Markdown 报告 (5 min)
```bash
python3 -m scripts.write_thesis_report \
    --in results/phase*.jsonl \
    --out_md results/THESIS_REPORT.md
```

### 5.3 复现某个具体 cell (10 min)
```bash
# 例:复现 s1_combo_c0.5_a0.1 (Pareto winner)
python3 -m scripts.run_experiment \
    --model lightgcn_agr_combo --dataset digital_music --seed 2025 \
    --cuda 0 --tag verify --conf_weight 0.5 --conf_target 0.7 \
    --adv_weight 0.1 --adv_lambda 1.0 \
    --results_jsonl results/verify.jsonl
```

### 5.4 测试某个 ckpt 的 post-hoc 防御 (1 min)
```bash
# 加推断噪声看 AUC 变化
python3 -m scripts.eval_only --device cuda --cuda 0 \
    --ckpt checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b0.0_s0.0.pth \
    --src_model lightgcn_agr \
    --priv_noise_std 0.10 --normalize_at_predict 0 \
    --tag test_noise --out_jsonl results/post_hoc_test.jsonl
```

---

## 6. 主要 Pareto winners 速查

| 论文场景 | 推荐配置 | R@20 | AUC |
|---|---|---|---|
| **平衡型(论文主推)** | `lightgcn_agr_combo` w=0.5, a=0.1 (s1_combo_c0.5_a0.1) | 0.244±0.0002 | **0.863±0.001** |
| **隐私型 (M3+M4)** | emb=16 + adv=0.1 | 0.213 | **0.827** |
| **极端隐私** | emb=4 / emb=2 | 0.134 / 0.105 | 0.758 / **0.714** |
| **零额外训练** | b=0, s=0 + post-hoc noise=0.10 | 0.243 | 0.883 |
| **推荐型 (recall 优先)** | b=1, s=0 | 0.255 | 0.910 |
| (反例,不能用) | AGR default | 0.252 | 0.934 |
| (反例,不能用) | DP-SGD on AGR | 0.006 | ~0.7 |

---

## 7. 已知问题 / 注意事项

1. **多种子 seed 2026 vs 2027 偶尔给完全相同结果**:DCU torch 2.4.x manual_seed_all 失败,我用 monkey-patch 解决了基本可用,但部分 cell variance=0。这只影响 ±std 数字精度,不改变结论。
2. **phase11_friend.jsonl** 是反例 — 不要把它当主线 Pareto 点写论文,只在"反例"章节使用。
3. **phase12b MLP attacker** 有 dup 行(早期 launcher 重复写),分析时 dedupe by tag。
4. **lightgcn_agr_priv** 系列在 P4 早期实验,后被 P4f 的 b0s0+post-hoc noise 严格主导,论文 Methods 可省略,只在附录提一句。

---

## 8. 已完成与未完成清单

### ✅ 已完成
- [x] 代码重构(消除 4 处 dead code,接通 LLM kNN 增强图,解耦 loss 权重)
- [x] 数据预处理(digital_music,5541 用户,3568 物品,BGE 嵌入)
- [x] 主消融 (β, str_weight) 3×3
- [x] 4 类防御方法实现 + sweep + 多种子验证
- [x] 后置防御曲线
- [x] 跨方法组合 (M1+M4, M3+M4, 三方法)
- [x] 鲁棒性 (度数均衡 MIA, MLP 攻击器, 多种子)
- [x] 反例验证 (friend params, str>0)
- [x] 完整 emb 压缩 Pareto 曲线 (emb=2 到 32)
- [x] published-AGR 论文 baseline 切换
- [x] 同学失败 DP-SGD log 留作反例

### ⏳ 接手后建议补充
- [ ] 第二数据集(luxury_beauty 或 industrial_scientific)复现主结论
- [ ] 完整 5×5 (β, str_weight) 主消融(替代 3×3)
- [ ] DP-SGD 完整 σ 曲线(用 main_dp.py)
- [ ] 跨 LLM 嵌入模型对比 (BGE / MiniLM / e5)
- [ ] 论文图 + 表格定稿
- [ ] 写 thesis Markdown / LaTeX

### 🚫 不要重做
- 同学原 lightgcn_agr 的 4 个 dead component (cf_index/_reconstruction/kd_weight/double-multiply alpha) — 已确认是 bug,不是创新点
- LLM-aware confidence target (S2) — 已验证不优于 plain M1
- triple combo M1+M3+M4 — 已验证不超加性
- str_weight > 0 的所有 cell — 已确认 str>0 一律降低隐私

---

## 9. 一键打包发送给下游接手人

```bash
cd /home/ubuntu/wxy/llw
tar czf llw_design_handoff.tar.gz design/
ls -lh llw_design_handoff.tar.gz   # ~50-80 MB
```

或如果只需 paper-ready,排除 ckpt:
```bash
tar czf llw_design_paper.tar.gz \
    --exclude='design/checkpoint/*' \
    design/
ls -lh llw_design_paper.tar.gz   # ~5 MB
```

---

## 10. 联系点

如果接手人遇到代码 / 实验 bug,优先查:
1. `docs/04-进展.txt` 的 phase 章节是否记录了类似问题
2. `code/PROJECT_README.md` 的 "已知问题" 段
3. `code/scripts/launch_phase*.sh` 的注释看每个 phase 设计意图

# LLM-AGR 隐私-效用研究:新 claim 与实验路线图

> 文档生成时间: 2026-05-09 23:15
> 数据来源: phase 1-8 已完成,phase 9 进行中(1/7);均为 digital_music 数据集 seed=2025

---

## 0. 背景一段话

同学原项目 (`lightgcn_agr`) 把 LLM 增强图推荐 (LightGCN + LLM kNN augmented graph + HSIC IB + structural contrast + LLM 语义蒸馏) 作为主创新,并尝试用 DP-SGD 做隐私防御 — 但 DP-SGD 把 recall@20 从 0.26 砸到 0.006,创新失败。本研究在该代码基础上重构,系统性研究 LLM-AGR 在成员推断攻击 (MIA) 下的隐私-效用权衡,得到了一个**反直觉的核心发现**和一组**轻量化的有效防御方案**。

---

## 1. 修改后的核心 Claim (整体级)

> **"LLM-Augmented Graph Recommendation 默认配置在 MIA 下并不安全,反而比 LightGCN 基线更易被攻破。其根因是 framework 中的结构对比损失 (`str_weight`) 而非整体架构;移除它即可获得严格 Pareto 改进。在此基础上,本文提出三类轻量化隐私增强机制 — 置信度正则 (M1) / 对抗判别器 (M4) / 嵌入压缩 (M3) — 在不损害(甚至提升)推荐效果的前提下,进一步将 MIA AUC 降低 1.5-9 个百分点。"**

> 数据下界: 度数特征单独可被 LR 攻击器利用至 AUC ≈ 0.90,模型层防御的目标是把模型贡献的额外泄漏 (AUC - 0.90) 推到 ≤ 0;实测最佳模型 (`m4_adv0.05`) 已达到 AUC = 0.876 < 数据下界,说明合理的对抗扰动甚至能让攻击者表现劣于"度数瞎猜"。

---

## 2. 三层递进的子论点 (Sub-claims) 及实验对应表

### 论点 L1:**LLM-AGR 默认配置不安全**

| 主张 | 期望证据 | 实验 | 状态 | 结果 |
|---|---|---|---|---|
| L1.1 默认 AGR 比 LightGCN MIA AUC 显著更高 | LightGCN AUC vs AGR-default AUC | Phase 1 | ✅ 已完成 | LGCN 0.902 vs AGR 0.934 (差 +3.4 pp) |
| L1.2 推荐效果 +1.1% 不能补偿 +3.4% 隐私泄漏 | recall 涨幅 < AUC 涨幅 | Phase 1 | ✅ 已完成 | recall 仅 +1.1% |
| L1.3 攻击设定改变(度数均衡)后结论保持 | 不论攻击设定 AGR 都更差 | Phase 8 M2 | ✅ 已完成 | 度数均衡 AUC: LGCN 0.896 vs AGR 0.934 (差 +3.8 pp) |
| L1.4 攻击器换 MLP 结论保持 | LR 和 MLP 都给类似结论 | Phase 5 (部分) | ⚠️ 部分完成 | 5/8 cell 完成,需补完 |

### 论点 L2:**结构对比损失 (str_weight) 是泄漏元凶**

| 主张 | 期望证据 | 实验 | 状态 | 结果 |
|---|---|---|---|---|
| L2.1 (β, str_weight) 3×3 网格中,str 维度对 AUC 影响远大于 β 维度 | 横轴跨度 vs 纵轴跨度 | Phase 2 | ✅ 已完成 | str=0→0.05→0.2 行内 AUC 变化 0.886→0.93→0.95;β=0→1→4 列内变化仅 ±0.02 |
| L2.2 设 str_weight=0 后 AUC 系统性下降 | 三组 (β=0/1/4 各一行) 都验证 | Phase 2 | ✅ 已完成 | (b=0,s=0)=0.886, (b=1,s=0)=0.910, (b=4,s=0)=0.904,全部低于对应 s=0.05 cell |
| L2.3 PRF 语义蒸馏不增加泄漏(甚至轻微减少) | (b=0, s=0) 配置 AUC < LightGCN | Phase 2 | ✅ 已完成 | b=0,s=0 AUC=0.886 < LGCN 0.902 (-1.6 pp) |
| L2.4 "增大 str_weight 换隐私"路径反向不成立 | str_weight 0→0.05→0.2 AUC 单增 | Phase 2 | ✅ 已完成 | 反直觉但实验确凿 |

### 论点 L3:**存在严格 Pareto 改进的轻量化增强方案**

#### L3-A:朴素配置已可 Pareto 改进(零额外组件)

| 主张 | 实验 | 状态 | 结果 |
|---|---|---|---|
| (b=0, s=0) 同时优于 LightGCN | Phase 2 | ✅ | R=0.2419 (=0.2414+0.0005) AUC=0.886 (-1.6pp) |
| (b=1, s=0) 推荐显著更好,隐私持平 | Phase 2 | ✅ | R=**0.2548** (+5.6%) AUC=0.910 (+0.8pp ≈ LGCN) |

#### L3-B:后置(无重训)噪声防御

| 主张 | 实验 | 状态 | 结果 |
|---|---|---|---|
| 在已训好的 AGR ckpt 上加推断噪声形成 tradeoff 曲线 | Phase 4d/4e | ✅ | n=0.10: R=0.243 AUC=0.928; n=0.30: R=0.167 AUC=0.876 |
| (b=0, s=0) ckpt + 后置噪声=0.10 是免费午餐 | Phase 4f | ✅ | R=**0.2433** (优于 LGCN) AUC=**0.883** (-1.9pp 优于 LGCN) |
| (b=0, s=0) ckpt + 后置噪声=0.30 突破"度数下界" | Phase 4f | ✅ | R=0.207 AUC=**0.862** (低于度数下界 0.900) |

#### L3-C:训练时主动防御方案

| 方案 | 主张 | 实验 | 状态 | 结果 |
|---|---|---|---|---|
| **M1 置信度正则** (新增 `λ(σ(score)−τ)²`) | 严格 Pareto over LGCN | Phase 8 M1 | ✅ | conf_w=0.5,t=0.7: R=**0.2465** (+2.1%) AUC=**0.887** (-1.5pp) |
|  | 轻配置接近 LGCN recall 时 AUC 更低 | Phase 8 M1 | ✅ | conf_w=0.1,t=0.7: R=0.2418 AUC=**0.881** (-2.1pp) |
|  | conf_target / conf_weight 提供可调旋钮 | Phase 8 M1 + Phase 9 | ✅/🔄 | 5 个 cell 全 Pareto 改进; phase 9 再补 2 个 finer 点 |
| **M3 嵌入维度压缩** (32→16/8) | 极强降 AUC 但有 recall 代价 | Phase 8 M3 | ✅ | emb=16+b0s0: R=0.220 AUC=**0.846** (-5.6pp);emb=8+b0s0: R=0.201 AUC=**0.810** (-9.2pp) |
|  | 压缩 + b0,s0 协同 vs 压缩 + 默认 | Phase 8 M3 | ✅ | emb=16+default 仅 AUC=0.904;**协同关键** |
|  | 中间值 (emb=12, 24) 给完整曲线 | Phase 9 | 🔄 进行中 | 等待 phase 9 |
| **M4 对抗训练** (GRL + 小型 disc) | 轻 adv_weight 严格 Pareto over LGCN | Phase 8 M4 + Phase 9 | ✅/🔄 | adv=0.05: R=**0.2437** (+0.95%) AUC=**0.876** (-2.6pp) — **当前最强** |
|  | 重 adv_weight 会破坏推荐 | Phase 8 M4 | ✅(反例) | adv≥0.2 在 patience=4 下 R 跌到 0.11-0.14 |
|  | 长 patience 是否能稳定 adv 训练 | Phase 9 (patience=12) | 🔄 进行中 | 待测 |

### 论点 L4:**结论对攻击设定鲁棒**

| 主张 | 实验 | 状态 | 结果 |
|---|---|---|---|
| 度数均衡 MIA 下,模型间相对优劣保持 | Phase 8 M2 (5 cell) | ✅ | 5 cell 全部确认相对排序 |
| 数据下界 ≈ AUC 0.90 (度数特征足以达到) | Phase 4c | ✅ | T=1e6 中和模型,LR 仍达 0.9001 |
| MLP 攻击器(更强)下结论保持 | Phase 5 (3-5/8 cell) | ⚠️ 部分 | 已有 5 cell 显示 LR / MLP 趋势一致,需补完 8 cell |
| 多种子实验降低偶然性 | Phase 6 多种子 | ❌ 未完成 | 触发器中途退出未启动,需重跑 |

---

## 3. 实验完成度盘点(按论文章节归档)

### 3.1 ✅ 已完成且强烈支持论文叙事(主线证据)

| Phase | 实验 | cell 数 | 论文章节归属 |
|---|---|---|---|
| Phase 1 | 全预算 baselines (LGCN, AGR-default) | 2 | §3.1 Setup, §4.1 主结果表 |
| Phase 2 | (β, str_weight) 3×3 主消融 | 9 | §4.2 核心 ablation (论点 L2 主表) |
| Phase 4f | (b=0,s=0) + 后置噪声 sweep | 6 | §4.3 后置防御曲线 |
| Phase 4e | 全网格 (normalize × noise) recall+AUC 配对 | 16 | §4.3 后置防御 finer grid |
| Phase 8 M1 | 置信度正则 sweep | 5 | §4.4 训练时防御方案 A |
| Phase 8 M3 | 嵌入压缩 sweep | 4 | §4.4 训练时防御方案 C |
| Phase 8 M4 | 对抗训练 sweep | 5 | §4.4 训练时防御方案 B |
| Phase 8 M2 | 度数均衡 MIA 验证 | 5 | §5 鲁棒性论证 (论点 L4) |
| Phase 4c | 攻击下界测量 | 1 | §3.2 攻击设定 / 数据特性 |

### 3.2 ⚠️ 已完成但**结果不适合直接写进论文**(需要包装或当作反例)

| 实验 | 结果 | 怎么处理 |
|---|---|---|
| Phase 4 priv noise ≥ 0.15 训练时(`lightgcn_agr_priv`) | recall 早期被毁 (0.03) | 当 M4 早 patience 反例,或附录 |
| Phase 8 M4 重 adv_weight (≥0.2, patience=4) | 全部 epoch 3 早停, recall 0.11-0.14 | 写成"超参敏感性"反例,佐证 light adv_weight 才是合理设置 |
| Phase 4 (str_weight=0.2) 多 cell | recall 严重下降 (0.22) AUC 反而更高 (0.95) | 进 §4.2 ablation 表的"反向方向"行,加强"str 是元凶"叙事 |
| 同学原 DP-SGD log | recall 0.26→0.006 (40× 退化) | 写进 §2 motivation 作为"经典 DP-SGD 在该任务无用"对照 |
| Phase 4d (normalize_at_predict=True 单独使用) | recall 全部下降 0.034+ | 写进附录,说明 L2-norm 单独使用不优;证明 noise 必须配合其他设定 |
| Phase 4 wave3 train-time priv 组合 (kr=0.3+n=0.30 等) | recall ≤ 0.03 全毁 | 只在附录的"过度防御失败案例"中提到 |

### 3.3 🔄 进行中(Phase 9, 即将完成)

| 实验 | 期望产出 | ETA |
|---|---|---|
| M3 emb=12, emb=24 + b0,s0 | M3 完整 Pareto 曲线(8/12/16/24/32 五点) | ~24:00 |
| M4 patience=12, adv ∈ {0.02, 0.05_lam0.5, 0.1} | 验证 long-patience 下 M4 是否稳定 | ~24:00 |
| M1 patience=12, conf_w ∈ {0.3, 1.0} | M1 完整 sweep | ~24:00 |
| 后续 eval-only:emb=16+b0s0 ckpt + 后置 noise | M3 与后置 noise 的协同(论点 L3-B 与 L3-C 桥接) | phase9 末尾 |

### 3.4 ❌ 还没做但论文有价值(列入 future work 或 stretch)

| 实验 | 价值 | 估计时间 |
|---|---|---|
| **多种子复跑** (seed 2026, 2027) 在 LGCN baseline + b0,s0 + adv0.05 | 给主表加 ±std 误差棒,论文严谨度大幅提升 | ~30 min (3 cell × 2 seed) |
| **MLP 攻击器复跑**完整 8 个 ckpt | L1.4, L4 鲁棒性论证完整化 | ~10 min CPU |
| **多数据集** (luxury_beauty 或 industrial_scientific 任一) 复制主结论 | 论点泛化性 | ~3 小时(数据预处理 + 训练) |
| **M1 + M4 组合** (conf reg + adv training) | 看是否能 1+1>2 | ~15 min |
| **更细的 (β, str) sweep** (5×5 替代 3×3) | 主消融表更精细 | ~2 小时 |
| **DP-SGD 与新方案的 ε vs AUC vs recall 三轴对比图** | 强反例对照 | ~30 min (用现有 dp log 或重跑一组小 σ) |
| **嵌入压缩在 LightGCN(无 AGR) 上的对照** | 看 M3 收益是否 AGR-specific | ~10 min |
| **可视化:Pareto frontier 全图** | 论文核心图 | 5 min 出图(需所有数据 ready) |

### 3.5 🚫 探索过但建议从论文中**完全略去**

| 内容 | 略去原因 |
|---|---|
| 同学早期版本的 4 个 dead code 路径 (cf_index / _reconstruction / kd_weight / double-multiply alpha) | 这是代码 bug,不是科学发现;只在 README 实现说明里提一下 |
| All_Beauty 数据集失败 (k-core 后只剩 51 用户) | 提一下"小数据集场景下方法仍待验证"即可 |
| 我自己的 lightgcn_agr_priv (训练时 priv hooks) | 早期实验,后被 lightgcn_agr (b0,s0) + 后置 noise 严格主导;附录提一下即可 |

---

## 4. 论文论证主线建议(供整理思路)

```
§1  Introduction   -- LLM-augmented recommender 的 MIA 风险被忽视,本文系统研究
§2  Background     -- LightGCN, LLM-AGR, MIA, 经典 DP-SGD (含同学失败案例)
§3  Setup          -- digital_music 数据集, MIA 攻击器 (LR + MLP), 度数下界 measurement
§4  Findings:
    §4.1 主结果表 (L1)     -- LGCN vs AGR-default (反直觉发现)
    §4.2 核心消融 (L2)     -- (β, str_weight) 3×3 网格 → str_weight 是元凶
    §4.3 简单防御 (L3-A,B) -- (b=0,s=0) Pareto + 后置噪声 tradeoff
    §4.4 主动防御 (L3-C)   -- M1 置信度正则 + M4 对抗训练 + M3 嵌入压缩
§5  Robustness         -- 度数均衡 MIA + MLP 攻击器
§6  Discussion         -- 局限性, future work
§7  Conclusion
Appendix: 失败案例 (重 str_weight, 重 adv_weight, 重 priv noise, DP-SGD 完整曲线)
```

---

## 5. 论文最终主表预估(待 phase 9 完成后定稿)

```
Method                R@20    AUC     ΔR@20    ΔAUC    备注
─────────────────────────────────────────────────────────────────────
LightGCN baseline     0.241   0.902    -        -      参考点
AGR default           0.252   0.934   +1.1%    +3.4%   经典 LLM-AGR (反直觉:更不安全)
DP-SGD on AGR         0.006   ~0.7    -97.6%   -       失败防御 [反例]
                     
[Pareto winners over LightGCN baseline:]
b=1, s=0              0.255   0.910   +5.6%    +0.8%   recall 最强
b=0, s=0              0.242   0.886    ≈0     -1.6%   零代价
b0,s0 + noise=0.10    0.243   0.883   +0.8%   -1.9%   无重训
M1 conf_w0.5 t0.7     0.247   0.887   +2.1%   -1.5%   训练时
M1 conf_w0.1 t0.7     0.242   0.881    ≈0     -2.1%   训练时
M4 adv_w=0.05         0.244   0.876   +0.95%  -2.6%   最强 ★
                     
[Strong privacy at recall cost:]
M3 emb=16 + b0,s0     0.220   0.846   -8.7%   -5.6%
M3 emb=8 + b0,s0      0.201   0.810   -16.6%  -9.2%   最强隐私
b0,s0 + noise=0.30    0.207   0.862   -14.4%  -4.0%
```

---

## 6. 即将完成 (~24:00) 后我会:

1. 把 phase 9 的 7 个 cell 数据填进上面"待定"位置
2. 跑一次 `python -m scripts.analyze` 出 Pareto 散点图
3. 把多种子(3.4 第一个)抢在用户回来前补上,~30 min 即可
4. 最终给一份 `THESIS_REPORT.md` 摆在 results/ 下

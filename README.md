# LLM-ARG 项目说明

本项目是一个基于图协同过滤的推荐系统代码库，包含普通推荐模型、AGR 变体模型，以及成员推理攻击（MIA）与差分隐私相关实验代码。

当前仓库里的主要用途有四类：

1. 训练普通推荐模型
2. 训练带 AGR 的推荐模型
3. 训练差分隐私实验模型
4. 对训练好的模型做成员推理攻击评估

这份 README 只描述**当前仓库中的实际代码流程**，不讨论论文里的理想化设定。

## 1. 项目结构

```text
config/          模型 YAML 配置与运行时配置解析
models/          推荐模型实现
load_data/       数据加载与 dataloader 构造
trainer/         普通训练器、DP 训练器、日志与评估
attack/          成员推理攻击与数据切分泄漏检查
pre/             预处理与 embedding 生成脚本
data/            数据集目录
checkpoint/      模型权重保存目录
log/             训练日志目录
main.py          普通训练入口
main_dp.py       旧版 DP 训练入口
main_dp_microbatch.py  新版 microbatch DP 实验入口
```

## 2. 已实现模型

`models/general_cf/` 下当前可用模型包括：

- `lightgcn`
- `lightgcn_agr`
- `sgl`
- `sgl_agr`
- `simgcl`
- `simgcl_agr`
- `bigcf`
- `bigcf_agr`

说明：

- `bigcf` 和 `bigcf_agr` 依赖 `torch_sparse`
- `*_agr` 模型依赖 `usr_emb_np.pkl` 和 `itm_emb_np.pkl`
- `lightgcn` / `sgl` / `simgcl` 系列使用稀疏邻接矩阵传播

## 3. 数据目录与运行前提

程序运行时会从 `data/<dataset_name>/` 读取数据。

一个可训练的数据集目录至少要包含：

- `trn_mat.pkl`
- `val_mat.pkl`
- `tst_mat.pkl`
- `usr_emb_np.pkl`
- `itm_emb_np.pkl`

通常还会包含：

- `reviews_clean.jsonl`
- `user_ids.pkl`
- `item_ids.pkl`
- `data_stats.json`

当前仓库里已有的数据集目录：

- `data/digital_music/`
- `data/industrial/`
- `data/luxury/`

注意：

- `digital_music` 已有较完整的中间产物
- `industrial` 和 `luxury` 如果只有原始 JSON，还需要先做预处理和 embedding 生成

## 4. 环境依赖

最少依赖包括：

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `pyyaml`

AGR 预处理与 embedding 生成还需要：

- `sentence-transformers`
- `tqdm`

`bigcf` / `bigcf_agr` 还需要：

- `torch_sparse`

一个基础安装示例：

```bash
python -m pip install torch numpy scipy scikit-learn pyyaml sentence-transformers tqdm
```

`torch_sparse` 需要按本机 PyTorch 和 CUDA 版本单独安装匹配的 wheel。

## 5. 配置文件说明

模型配置位于：

- [config/models_config/lightgcn.yml](D:/design/LLM-ARG/config/models_config/lightgcn.yml)
- [config/models_config/lightgcn_agr.yml](D:/design/LLM-ARG/config/models_config/lightgcn_agr.yml)
- [config/models_config/sgl.yml](D:/design/LLM-ARG/config/models_config/sgl.yml)
- [config/models_config/sgl_agr.yml](D:/design/LLM-ARG/config/models_config/sgl_agr.yml)
- [config/models_config/simgcl.yml](D:/design/LLM-ARG/config/models_config/simgcl.yml)
- [config/models_config/simgcl_agr.yml](D:/design/LLM-ARG/config/models_config/simgcl_agr.yml)
- [config/models_config/bigcf.yml](D:/design/LLM-ARG/config/models_config/bigcf.yml)
- [config/models_config/bigcf_agr.yml](D:/design/LLM-ARG/config/models_config/bigcf_agr.yml)

运行时配置入口：

- [config/configurator.py](D:/design/LLM-ARG/config/configurator.py)

常用命令行参数：

```bash
--model
--dataset
--device
--seed
--cuda
--diverse
--exp_tag
```

当前差分隐私相关覆盖参数：

```bash
--dp_noise_multiplier
--noise_multiplier
--dp_delta
--dp_clip_norm
--dp_target_epsilon
```

说明：

- `--exp_tag` 用于给本次训练打实验标签
- 标签会进入 checkpoint 文件名和运行元数据，便于参数试验留痕

## 6. 整体流程

如果你要从原始 Amazon 风格评论数据一路跑到 MIA 评估，流程应当是：

1. 准备原始 JSON 评论文件
2. 运行 `pre/preprocess.py`
3. 运行 `pre/generate_user_embeddings.py`
4. 运行 `pre/generate_item_embeddings.py`
5. 运行普通训练或 DP 训练
6. 使用 `attack/MIA.py` 做成员推理攻击
7. 比较推荐指标和 MIA 指标

下面按这个顺序展开。

## 7. 数据预处理流程

### 7.1 第一步：构建交互矩阵与清洗评论

脚本：

- [pre/preprocess.py](D:/design/LLM-ARG/pre/preprocess.py)

作用：

- 读取原始评论 JSON
- 输出清洗后的 `reviews_clean.jsonl`
- 构建 `trn_mat.pkl / val_mat.pkl / tst_mat.pkl`
- 生成 `user_ids.pkl / item_ids.pkl`
- 保存 `data_stats.json`

### 7.2 第二步：生成用户语义 embedding

脚本：

- [pre/generate_user_embeddings.py](D:/design/LLM-ARG/pre/generate_user_embeddings.py)

作用：

- 读取 `reviews_clean.jsonl`
- 汇总每个用户的评论文本
- 使用 `SentenceTransformer` 生成用户 embedding
- 输出 `usr_emb_np.pkl`

### 7.3 第三步：生成物品语义 embedding

脚本：

- [pre/generate_item_embeddings.py](D:/design/LLM-ARG/pre/generate_item_embeddings.py)

作用：

- 读取 `reviews_clean.jsonl`
- 汇总每个物品的评论文本
- 使用 `SentenceTransformer` 生成物品 embedding
- 输出 `itm_emb_np.pkl`

## 8. 预处理命令示例

### 8.1 `industrial`

原始文件：

- `data/industrial/Industrial_and_Scientific_5.json`

命令：

```bash
python pre/preprocess.py --dataset industrial --src data/industrial/Industrial_and_Scientific_5.json
python pre/generate_user_embeddings.py --dataset industrial --reviews_path data/industrial/reviews_clean.jsonl --output_dir data/industrial --mode both --min_reviews 1 --max_users 0
python pre/generate_item_embeddings.py --dataset industrial --reviews_path data/industrial/reviews_clean.jsonl --output_dir data/industrial --min_reviews_per_item 1
```

### 8.2 `luxury`

原始文件：

- `data/luxury/Luxury_Beauty_5.json`

命令：

```bash
python pre/preprocess.py --dataset luxury --src data/luxury/Luxury_Beauty_5.json
python pre/generate_user_embeddings.py --dataset luxury --reviews_path data/luxury/reviews_clean.jsonl --output_dir data/luxury --mode both --min_reviews 1 --max_users 0
python pre/generate_item_embeddings.py --dataset luxury --reviews_path data/luxury/reviews_clean.jsonl --output_dir data/luxury --min_reviews_per_item 1
```

### 8.3 `digital_music`

如果你要重建 `digital_music` 的中间文件，也可以这样跑：

```bash
python pre/preprocess.py --dataset digital_music --src data/digital_music/Digital_Music_5.json
python pre/generate_user_embeddings.py --dataset digital_music --reviews_path data/digital_music/reviews_clean.jsonl --output_dir data/digital_music --mode both --min_reviews 1 --max_users 0
python pre/generate_item_embeddings.py --dataset digital_music --reviews_path data/digital_music/reviews_clean.jsonl --output_dir data/digital_music --min_reviews_per_item 1
```

## 9. 普通训练流程

普通训练入口：

- [main.py](D:/design/LLM-ARG/main.py)

流程：

1. 读取模型配置
2. 读取交互矩阵与 embedding
3. 构造模型
4. 用普通 `Trainer` 训练
5. 在验证集上 early stop
6. 保存 checkpoint 并在测试集评估

### 9.1 示例命令

训练 `lightgcn_agr`：

```bash
python main.py --model lightgcn_agr --dataset digital_music --device cuda
```

带实验标签：

```bash
python main.py --model lightgcn_agr --dataset digital_music --device cuda --exp_tag reg1e5_keep07_beta6
```

训练 `bigcf_agr`：

```bash
python main.py --model bigcf_agr --dataset digital_music --device cuda
```

训练 `lightgcn`：

```bash
python main.py --model lightgcn --dataset digital_music --device cuda
```

训练 `bigcf`：

```bash
python main.py --model bigcf --dataset digital_music --device cuda
```

如果只用 CPU：

```bash
python main.py --model lightgcn_agr --dataset digital_music --device cpu
```

如果指定 GPU 编号：

```bash
python main.py --model lightgcn_agr --dataset digital_music --device cuda --cuda 0
```

## 10. 差分隐私训练入口说明

当前仓库里有两个 DP 训练入口。

### 10.1 旧版 DP 入口

入口：

- [main_dp.py](D:/design/LLM-ARG/main_dp.py)
- [trainer/dp_trainer.py](D:/design/LLM-ARG/trainer/dp_trainer.py)

特点：

- 是旧版手写 DP 训练器
- 当前实现更偏 selective / embedding-level DP 风格
- 适合保留做对照

示例：

```bash
python main_dp.py --model lightgcn_agr --dataset digital_music --device cuda
```

### 10.2 新版 microbatch DP 入口

入口：

- [main_dp_microbatch.py](D:/design/LLM-ARG/main_dp_microbatch.py)
- [trainer/dp_trainer_microbatch.py](D:/design/LLM-ARG/trainer/dp_trainer_microbatch.py)

特点：

- 以 microbatch clipping + Gaussian noise 为基础
- 比逐样本版本更容易跑出结果
- 当前更适合作为 DP 实验主入口

示例：

```bash
python main_dp_microbatch.py --model lightgcn_agr --dataset digital_music --device cuda
```

带实验标签：

```bash
python main_dp_microbatch.py --model lightgcn_agr --dataset digital_music --device cuda --exp_tag clip08
```

在 `industrial` 上运行：

```bash
python main_dp_microbatch.py --model lightgcn_agr --dataset industrial --device cuda
```

在 `luxury` 上运行：

```bash
python main_dp_microbatch.py --model lightgcn_agr --dataset luxury --device cuda
```

## 11. 推荐的 DP 配置

对于 `lightgcn_agr`，建议先用比较保守的配置起步：

```yaml
privacy:
  microbatch_size: 8
  clip_norm: 1.0
  noise_multiplier: 1.0
  delta: 1e-5
  target_scope: all
  save_suffix: dpmb
  disable_model_randomness: true
```

说明：

- `microbatch_size` 越小，越接近标准 sample-level DP-SGD，但越慢
- `noise_multiplier` 越大，隐私更强，但推荐效果通常更差
- `clip_norm` 控制梯度裁剪阈值
- `target_scope: all` 表示整模型加噪
- `disable_model_randomness: true` 用于减小模型自身随机性对 DP 实验的干扰

## 12. 日志与模型保存

训练日志保存在：

```text
log/<model_name>/
```

普通训练权重保存路径：

```text
checkpoint/<model>/<model>-<dataset>-<seed>-<run_id>.pth
```

DP 权重保存路径：

```text
checkpoint/<model>/<model>-<dataset>-<seed>-<save_suffix>-<run_id>.pth
```

例如：

```text
checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2026-dpmb-20260507_153012_clip08.pth
```

同时会生成同名元数据文件：

```text
checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2026-dpmb-20260507_153012_clip08.json
```

其中会记录：

- 模型名称
- 数据集名称
- seed
- run_id
- 训练配置
- 测试配置
- DP 配置（如果是 DP 训练）

这样可以避免不同参数实验重复覆盖同一个 checkpoint。

## 13. 成员推理攻击（MIA）

脚本：

- [attack/MIA.py](D:/design/LLM-ARG/attack/MIA.py)

作用：

- 加载一个训练好的推荐模型
- 从训练集和非训练集交互中构造 member / non-member 样本
- 基于 embedding 特征训练攻击器
- 输出隐私泄露指标

输出指标包括：

- `AUC`
- `ACC`
- `Precision`
- `Recall`
- `F1`

### 13.1 对普通模型做 MIA

```bash
python attack/MIA.py --model lightgcn_agr --dataset digital_music --device cuda --checkpoint_path checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2026.pth
```

### 13.2 对 DP 模型做 MIA

```bash
python attack/MIA.py --model lightgcn_agr --dataset digital_music --device cuda --checkpoint_path checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2026-dpmb.pth
```

### 13.3 使用攻击标签并保存独立日志

```bash
python attack/MIA.py --model lightgcn_agr --dataset digital_music --device cuda --checkpoint_path checkpoint/lightgcn_agr/xxx.pth --attack_tag clip08_mia
```

MIA 运行后会额外保存：

- 文本日志：`log/mia/<model_name>/*.log`
- 结构化结果：`log/mia/<model_name>/*.json`

结果 JSON 中会包含：

- checkpoint 路径
- 攻击参数
- member / non-member 数量
- AUC / ACC / Precision / Recall / F1
- 隐私风险等级

### 13.4 可选参数

```bash
--member_limit
--nonmember_limit
--nonmember_source
--test_size
--attack_seed
--attack_tag
```

示例：

```bash
python attack/MIA.py --model lightgcn_agr --dataset industrial --device cuda --checkpoint_path checkpoint/lightgcn_agr/lightgcn_agr-industrial-2026-dpmb.pth --member_limit 20000 --nonmember_limit 20000 --nonmember_source mixed
```

## 14. 数据切分泄漏检查

在跑 MIA 之前，建议先确认 train / val / test 是否存在重叠。

脚本：

- [attack/check_split_overlap.py](D:/design/LLM-ARG/attack/check_split_overlap.py)

检查某个数据集：

```bash
python attack/check_split_overlap.py --dataset digital_music
```

检查全部数据集：

```bash
python attack/check_split_overlap.py
```

## 15. 典型实验流程

### 15.1 普通模型 vs DP 模型

以 `lightgcn_agr` 为例：

1. 预处理数据
2. 训练普通模型
3. 训练 DP 模型
4. 分别做 MIA
5. 比较推荐效果与 MIA AUC

命令示例：

```bash
python main.py --model lightgcn_agr --dataset industrial --device cuda
python main_dp_microbatch.py --model lightgcn_agr --dataset industrial --device cuda
python attack/MIA.py --model lightgcn_agr --dataset industrial --device cuda --checkpoint_path checkpoint/lightgcn_agr/lightgcn_agr-industrial-2026.pth
python attack/MIA.py --model lightgcn_agr --dataset industrial --device cuda --checkpoint_path checkpoint/lightgcn_agr/lightgcn_agr-industrial-2026-dpmb.pth
```

### 15.2 普通 AGR 模型隐私实验

以 `bigcf_agr` 为例：

```bash
python main.py --model bigcf_agr --dataset digital_music --device cuda
python attack/MIA.py --model bigcf_agr --dataset digital_music --device cuda --checkpoint_path checkpoint/bigcf_agr/bigcf_agr-digital_music-2025.pth
```

## 16. 当前代码边界

这部分很重要。

1. `main_dp.py` 对应的旧版 DP 代码不是严格整模型 DP-SGD
2. `main_dp_microbatch.py` 对应的是更实用的 microbatch DP 近似实现
3. 图推荐任务里的“隐私单位”到底是 interaction-level 还是 user-level，当前代码没有完全形式化
4. `*_agr` 模型依赖额外的语义 embedding，因此预处理必须先完成
5. `bigcf` / `bigcf_agr` 依赖 `torch_sparse`

所以当前仓库最稳妥的说法是：

- 它支持推荐模型训练
- 支持成员推理攻击评估
- 支持手写的 DP 风格训练实验

但不要把当前实现直接等同于成熟工业级 DP 框架。

## 17. 最简启动命令汇总

### 17.1 预处理

```bash
python pre/preprocess.py --dataset industrial --src data/industrial/Industrial_and_Scientific_5.json
python pre/generate_user_embeddings.py --dataset industrial --reviews_path data/industrial/reviews_clean.jsonl --output_dir data/industrial --mode both --min_reviews 1 --max_users 0
python pre/generate_item_embeddings.py --dataset industrial --reviews_path data/industrial/reviews_clean.jsonl --output_dir data/industrial --min_reviews_per_item 1
```

### 17.2 普通训练

```bash
python main.py --model lightgcn_agr --dataset industrial --device cuda
```

### 17.2.1 带实验标签的普通训练

```bash
python main.py --model lightgcn_agr --dataset industrial --device cuda --exp_tag reg1e5_keep07_beta6
```

### 17.3 新版 DP 训练

```bash
python main_dp_microbatch.py --model lightgcn_agr --dataset industrial --device cuda
```

### 17.4 成员推理攻击

```bash
python attack/MIA.py --model lightgcn_agr --dataset industrial --device cuda --checkpoint_path checkpoint/lightgcn_agr/lightgcn_agr-industrial-2026-dpmb.pth
```

### 17.4.1 带攻击标签的成员推理攻击

```bash
python attack/MIA.py --model lightgcn_agr --dataset industrial --device cuda --checkpoint_path checkpoint/lightgcn_agr/<对应checkpoint>.pth --attack_tag reg1e5_keep07_beta6
```

## 19. 自动化参数试验

如果你想让 `lightgcn_agr` 在你休息时自动持续跑参数试验，可以使用：

- [auto_tune_lightgcn_agr.py](D:/design/LLM-ARG/auto_tune_lightgcn_agr.py)

这个脚本会按阶段顺序运行实验：

1. 训练 `lightgcn_agr`
2. 自动运行 `MIA.py`
3. 读取训练结果和 MIA 结果
4. 按当前阶段选择最优参数
5. 用该最优参数进入下一阶段

当前默认阶段顺序是：

1. `reg_weight`
2. `keep_rate`
3. `beta`
4. `train_patience`
5. `prf_weight`
6. `str_weight`
7. `alpha`

### 19.1 示例命令

```bash
python auto_tune_lightgcn_agr.py --dataset industrial --device cuda --cuda 0 --seed 2026 --train_batch_size 32 --train_epoch 20 --member_limit 20000 --nonmember_limit 20000 --nonmember_source mixed --privacy_weight 0.25 --tag_prefix autotune_industrial
```

### 19.2 结果输出

脚本运行后会自动生成：

- 每次训练的 checkpoint 和元数据 JSON
- 每次 MIA 的日志和结果 JSON
- 一个自动调参总汇总文件：

```text
log/auto_tune/lightgcn_agr/<dataset>_<timestamp>_<tag_prefix>.json
```

这个汇总文件会记录：

- 每个阶段测试过的候选值
- 每个候选值对应的推荐指标和 MIA AUC
- 当前阶段选出的最优值
- 全部阶段结束后的最佳参数组合

### 19.3 断点续跑

如果之前已经跑过一部分实验，可以加：

```bash
--resume_existing
```

脚本会优先复用已有的 checkpoint 元数据和 MIA 结果，避免重复训练和重复攻击。

## 18. 运行时参数覆盖说明

当前 `main.py` / `main_dp.py` / `main_dp_microbatch.py` 运行时，**只能覆盖少量配置项**，不是所有 yml 参数都能直接从命令行传入。

目前已支持直接命令行覆盖的主要参数有：

```bash
--model
--dataset
--device
--seed
--cuda
--diverse
--exp_tag
--train_batch_size
--train_epoch
--train_patience
--test_batch_size
--reg_weight
--keep_rate
--beta
--prf_weight
--str_weight
--alpha
--kd_temperature
--mask_ratio
--cl_weight
--cen_weight
--dp_noise_multiplier
--noise_multiplier
--dp_delta
--dp_clip_norm
--dp_target_epsilon
--dp_microbatch_size
--dp_num_microbatches
--dp_target_scope
--dp_disable_model_randomness
```

也就是说：

- 数据集、设备、seed、实验标签，可以直接通过命令行改
- 部分 DP 参数可以直接通过命令行改
- 常用训练超参和 AGR/DP 相关核心超参，现在也支持通过命令行覆盖
- 仍然有一部分较少使用的参数需要手动改 yml

例如现在可以直接这样跑：

```bash
python main.py --model lightgcn_agr --dataset industrial --device cuda --reg_weight 1e-5 --keep_rate 0.7 --beta 6.0 --prf_weight 0.03 --str_weight 1.5 --alpha 0.15 --train_patience 3 --exp_tag reg1e5_keep07_beta6
```

DP 入口也可以直接覆盖：

```bash
python main_dp_microbatch.py --model lightgcn_agr --dataset industrial --device cuda --train_batch_size 64 --dp_microbatch_size 8 --dp_clip_norm 1.0 --dp_noise_multiplier 0.8 --dp_delta 1e-5 --dp_target_scope all --dp_disable_model_randomness true --exp_tag clip08
```

以下这类参数如果你后面要改，通常仍然需要手动修改对应 yml：

```yaml
model.layer_num
model.embedding_size
model.intent_num
model.sigma
model.recon_weight
model.re_temperature
privacy.target_names
train.test_step
```

因此你现在做参数试验时，推荐流程是：

1. 优先使用命令行覆盖常用参数
2. 再用 `--exp_tag` 运行训练
3. 再用 `--attack_tag` 运行 MIA
4. 只有在命令行未覆盖的参数上，才手动修改 yml

例如：

```bash
python main.py --model lightgcn_agr --dataset industrial --device cuda --reg_weight 1e-5 --keep_rate 0.7 --beta 6.0 --exp_tag reg1e5_keep07_beta6
python attack/MIA.py --model lightgcn_agr --dataset industrial --device cuda --checkpoint_path checkpoint/lightgcn_agr/<对应checkpoint>.pth --attack_tag reg1e5_keep07_beta6
```

### 17.5 切分泄漏检查

```bash
python attack/check_split_overlap.py --dataset industrial
```

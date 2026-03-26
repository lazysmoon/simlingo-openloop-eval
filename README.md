# SimLingo 开环评估

基于 [SimLingo (CVPR 2025 Spotlight)](https://github.com/RenzKa/simlingo) 的开环评估框架，支持在单张消费级显卡（RTX 4060/4070）上完成评估，无需启动 CARLA 仿真器。

## 项目背景

SimLingo 是 CARLA 2024 自动驾驶挑战赛冠军模型，基于 InternVL2-1B + Qwen2-0.5B 构建，仅使用摄像头输入，直接输出路径路点和速度路点。

本仓库在 SimLingo 原始训练框架基础上：
- 添加了**开环评估脚本**（不需要启动 CARLA，直接在离线数据集上评估）
- 添加了**可视化脚本**（ADE/FDE 分布图、单帧完整分析图）
- 修复了若干本地单机运行的兼容性问题
- 提供了 patch 文件，方便应用到原始仓库

## 评估结果

在 SimLingo 验证集（chunk_015，160帧）上的开环评估结果：

| 指标 | 数值 |
|------|------|
| ADE (路径路点) | 0.0618 m |
| FDE (路径路点) | 0.1699 m |
| 评估帧数 | 160 |

## 硬件要求

| 用途 | 最低显存 |
|------|---------|
| 开环评估（推理） | 4GB（RTX 4060 可用） |
| 闭环评估（CARLA）| 12GB（需要 RTX 4070s 以上） |

## 快速开始

### 1. 克隆原始 SimLingo 仓库

```bash
git clone https://github.com/RenzKa/simlingo.git
cd simlingo
chmod +x setup_carla.sh
./setup_carla.sh
```

### 2. 安装环境

```bash
conda env create -f environment.yaml
conda activate simlingo
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.0.post2
```

### 3. 应用 patch（修复本地单机运行问题）

```bash
# 克隆本仓库
git clone https://github.com/YOUR_USERNAME/simlingo-openloop-eval.git

# 进入 simlingo 目录，应用所有 patch
cd simlingo
git apply ../simlingo-openloop-eval/patches/driving_model.patch
git apply ../simlingo-openloop-eval/patches/agent_simlingo.patch
git apply ../simlingo-openloop-eval/patches/leaderboard_evaluator.patch

# 复制评估脚本
cp ../simlingo-openloop-eval/predict.py simlingo_training/
cp ../simlingo-openloop-eval/visualize_open_loop.py .
cp ../simlingo-openloop-eval/visualize_single_frame.py .
```

### 4. 下载模型权重

```bash
# 使用国内镜像加速
HF_ENDPOINT=https://hf-mirror.com python download_model.py
```

权重下载后位于 `checkpoints/simlingo/`，约 3.2GB。

### 5. 下载验证集数据（选一个 chunk 即可）

```bash
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='RenzKa/simlingo',
    repo_type='dataset',
    filename='data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_015.tar.gz',
    local_dir='./database'
)
"
mkdir -p database/simlingo
tar -xzf database/data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_015.tar.gz \
    -C database/simlingo/
```

同时下载 buckets 文件：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='RenzKa/simlingo',
    repo_type='dataset',
    filename='buckets_paths.pkl',
    local_dir='./database'
)
"
mkdir -p database/bucketsv2_simlingo
cp database/buckets_paths.pkl database/bucketsv2_simlingo/
```

### 6. 配置路径

```bash
cd simlingo_training

# 建立软链接（让训练框架找到正确的路径）
ln -sf ../database database
ln -sf ../data data
ln -sf ../checkpoints/simlingo/.hydra .hydra
```

### 7. 运行开环评估

```bash
cd simlingo_training
conda activate simlingo

HYDRA_FULL_ERROR=1 python predict.py \
    'experiment=simlingo_seed1' \
    'checkpoint="/path/to/simlingo/checkpoints/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"' \
    'data_module.base_dataset.data_path=database/simlingo' \
    'data_module.base_dataset.bucket_path=database/bucketsv2_simlingo' \
    'data_module.batch_size=1' \
    'data_module.num_workers=0' \
    'gpus=1'
```

评估结果保存在：
```
simlingo_training/outputs/OpenGVLab/InternVL2-1B/predictions/dreamer_results_rank_0.json
```

## 可视化

### 批量可视化（ADE/FDE 分布 + 最优/最差帧对比）

```bash
cd simlingo

python visualize_open_loop.py \
    --data simlingo_training/outputs/OpenGVLab/InternVL2-1B/predictions/per_frame_waypoints_rank_0.json \
    --output eval_results/visualization
```

输出文件：

| 文件 | 内容 |
|------|------|
| `01_summary.png` | ADE/FDE 汇总统计 |
| `02_metrics_distribution.png` | ADE/FDE 分布直方图 |
| `03_worst_frames.png` | 预测最差的 12 帧 |
| `04_best_frames.png` | 预测最优的 12 帧 |
| `05_random_frames.png` | 随机 12 帧 |

### 单帧完整分析（图像 + 路点 + 场景描述）

```bash
python visualize_single_frame.py \
    --waypoints_json simlingo_training/outputs/OpenGVLab/InternVL2-1B/predictions/per_frame_waypoints_rank_0.json \
    --frame_id 50 \
    --output eval_results/single_frame
```

加 `--no_llm` 跳过语言描述生成（节省约 30 秒）。

## 文件说明

```
simlingo-openloop-eval/
├── predict.py                  # 开环评估主脚本
├── visualize_open_loop.py      # 批量可视化
├── visualize_single_frame.py   # 单帧完整分析
├── run_eval_local.sh           # 闭环评估脚本（需要 CARLA）
├── download_model.py           # 模型权重下载
└── patches/
    ├── driving_model.patch     # driving.py 修复（ADE/FDE 计算、qa_templates 兼容）
    ├── agent_simlingo.patch    # agent 路径修复
    ├── leaderboard_evaluator.patch  # 移除 debugpy，添加 quality-level
    └── debug_config.patch      # 实验配置修复
```

## 常见问题

**Q: CUDA out of memory**

CARLA 启动后会占用约 5.7GB 显存，如果显卡总显存不足 12GB，闭环评估会 OOM。开环评估不需要启动 CARLA，4GB 显存即可。

**Q: `get_original_cwd()` 报错**

需要在 `simlingo_training/` 目录下运行 `predict.py`，且需要建立软链接（见步骤 6）。

**Q: 下载速度慢**

所有 HuggingFace 下载均支持 `HF_ENDPOINT=https://hf-mirror.com` 镜像加速。

## 致谢

本项目基于 [SimLingo](https://github.com/RenzKa/simlingo)（CVPR 2025 Spotlight）构建，感谢原作者的开源工作。

## 引用

```bibtex
@inproceedings{renz2025simlingo,
  title={SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author={Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle={CVPR},
  year={2025}
}
```

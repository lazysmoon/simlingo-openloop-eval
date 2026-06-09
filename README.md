# SimLingo 开环评估

基于 [SimLingo (CVPR 2025 Spotlight)](https://github.com/RenzKa/simlingo) 的开环评估框架，支持在单张消费级显卡上完成评估，无需启动 CARLA 仿真器。

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

| 用途 | 最低显存 | 备注 |
|------|----------|------|
| 开环评估（推理） | 4GB（RTX 4060 可用） | 无需 CARLA |
| 闭环评估（CARLA） | 12GB（需要 RTX 4070s 以上） | 需要启动 CARLA |

### GPU 架构兼容性

| GPU | 架构 | 计算能力 | 环境要求 |
|-----|------|----------|----------|
| RTX 4060/4070/4090 | Ada Lovelace | sm_89 | 方式一/方式二均可 ✅ |
| RTX 5070/5080/5090 | Blackwell | sm_120 | **必须使用方式一**（Python 3.11 + PyTorch ≥ 2.6） ⚠️ |
| H100/H200 | Hopper | sm_90 | 方式一/方式二均可 ✅ |
| B100/B200 | Blackwell | sm_120 | **必须使用方式一** ⚠️ |

---

## 环境安装

提供两种方式，任选其一：

### 方式一（推荐）：直接拷贝预配置环境

已配置好 Python 3.11 + PyTorch 2.11 + CUDA 12.8 的完整 conda 环境，**支持所有 GPU 架构（包括 RTX 50 系列）**。

**下载地址**：

| 平台 | 链接 |
|------|------|
| 夸克网盘 | `链接: [https://pan.quark.cn/s/661275bacd53] |

下载文件：
- `simlingo_packed.tar.gz`（约 6.6GB）— conda 环境压缩包
- `install_env.sh` — 一键安装脚本

**安装步骤**：

```bash
# 确保已安装 miniconda 或 anaconda
# 将下载的两个文件放在同一目录下，然后执行：
bash install_env.sh

# 如果 conda 路径未自动检测到，手动指定：
bash install_env.sh /path/to/miniconda3
```

安装完成后：
```bash
conda activate simlingo
export CUDA_HOME=$CONDA_PREFIX
export HF_ENDPOINT=https://hf-mirror.com

# 验证
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 方式二：从零搭建环境

适用于 Ada Lovelace 及更早架构的 GPU（RTX 40 系列及以下）。

```bash
# 1. 克隆原始 SimLingo 仓库
git clone https://github.com/RenzKa/simlingo.git
cd simlingo
chmod +x setup_carla.sh
./setup_carla.sh

# 2. 创建 conda 环境
conda env create -f environment.yaml
conda activate simlingo

# 3. 安装 PyTorch
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.0.post2
```

<details>
<summary><b>如果你使用 RTX 50 系列（Blackwell 架构），点击展开手动搭建步骤</b></summary>

RTX 5070/5080/5090 需要 PyTorch ≥ 2.6.0 + CUDA 12.8 + Python ≥ 3.10，原始 environment.yaml 不兼容。

```bash
# 1. 新建 Python 3.11 环境
conda create -n simlingo python=3.11 -y
conda activate simlingo

# 2. 安装支持 sm_120 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 安装 CUDA Toolkit 和 cuDNN
conda install -c nvidia cuda-toolkit -y
pip install nvidia-cudnn-cu12
export CUDA_HOME=$CONDA_PREFIX

# 4. 安装项目依赖（使用仓库提供的 requirements 文件）
pip install -r requirements.txt

# 5. 如果有包安装失败，逐个跳过再补装：
#    pip install opencv-python shapely numpy carla==0.9.16
```

</details>

---

## 设置 CUDA 环境变量

**无论使用哪种安装方式，都需要设置以下环境变量**，否则 DeepSpeed 会报 `CUDA_HOME does not exist`：

```bash
# 设置 CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX
# 或者根据 nvcc 路径设置：
# export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# 设置 HuggingFace 镜像（国内网络必需）
export HF_ENDPOINT=https://hf-mirror.com

# 建议写入 .bashrc 永久生效
echo 'export CUDA_HOME=$CONDA_PREFIX' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

---

## 快速开始

### 1. 克隆仓库并应用 patch

```bash
# 克隆原始 SimLingo 仓库
git clone https://github.com/RenzKa/simlingo.git
# 克隆本仓库
git clone https://github.com/lazysmoon/simlingo-openloop-eval.git

# 进入 simlingo 目录，应用所有 patch
cd simlingo
git apply ../simlingo-openloop-eval/patches/driving_model.patch
git apply ../simlingo-openloop-eval/patches/agent_simlingo.patch
git apply ../simlingo-openloop-eval/patches/leaderboard_evaluator.patch

# 复制评估和可视化脚本
cp ../simlingo-openloop-eval/predict.py simlingo_training/
cp ../simlingo-openloop-eval/visualize_open_loop.py .
cp ../simlingo-openloop-eval/visualize_single_frame.py .
cp ../simlingo-openloop-eval/requirements.txt .
```

### 2. 获取模型权重和 InternVL2-1B

> **服务器用户**：权重已预置在公共目录，脚本会自动检测并跳过下载，直接建立软链接。
> **本地用户**：若本地不存在权重，脚本会自动从 HuggingFace 下载。

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd simlingo
python3 -c "
import os, subprocess, sys

# ── 可按需修改的路径配置 ──────────────────────────────────────────────────
SERVER_SIMLINGO_CKPT = '/opt/models/checkpoints/simlingo'   # 服务器上 SimLingo 权重目录
SERVER_INTERNVL2     = '/opt/models/InternVL2-1B'           # 服务器上 InternVL2-1B 目录
LOCAL_SIMLINGO_CKPT  = './checkpoints/simlingo'             # 本地目标路径（软链接 or 下载目录）
LOCAL_INTERNVL2      = './models/InternVL2-1B'              # 本地目标路径（软链接 or 下载目录）
# ─────────────────────────────────────────────────────────────────────────

def ensure(server_path, local_path, download_fn):
    if os.path.exists(local_path):
        print(f'[skip] 已存在: {local_path}')
        return
    if os.path.exists(server_path):
        parent = os.path.dirname(os.path.abspath(local_path))
        os.makedirs(parent, exist_ok=True)
        os.symlink(os.path.abspath(server_path), local_path)
        print(f'[link] {local_path} -> {server_path}')
    else:
        print(f'[download] 本地及服务器均未找到，开始下载到 {local_path} ...')
        download_fn(local_path)

# SimLingo checkpoint
def dl_simlingo(local_path):
    subprocess.check_call([sys.executable, 'download_model.py'])

# InternVL2-1B
def dl_internvl2(local_path):
    subprocess.check_call([
        'huggingface-cli', 'download',
        'OpenGVLab/InternVL2-1B',
        '--local-dir', local_path
    ])

ensure(SERVER_SIMLINGO_CKPT, LOCAL_SIMLINGO_CKPT, dl_simlingo)
ensure(SERVER_INTERNVL2,     LOCAL_INTERNVL2,     dl_internvl2)
print('完成。')
"
```

### 3. 获取验证集数据

> **服务器用户**：数据集已预置在公共目录，脚本会自动检测并跳过下载，直接建立软链接。
> **本地用户**：若本地不存在数据，脚本会自动从 HuggingFace 下载并解压。

```bash
export HF_ENDPOINT=https://hf-mirror.com

python3 -c "
import os, subprocess, sys, shutil

# ── 可按需修改的路径配置 ──────────────────────────────────────────────────
SERVER_SIMLINGO_DATA = '/data/datasets/database/simlingo'              # 服务器上已解压的 simlingo 数据目录
SERVER_BUCKETS       = '/data/datasets/database/bucketsv2_simlingo'    # 服务器上 buckets 目录
LOCAL_SIMLINGO_DATA  = './database/simlingo'                 # 本地目标路径
LOCAL_BUCKETS        = './database/bucketsv2_simlingo'       # 本地目标路径

CHUNK_FILENAME = 'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_015.tar.gz'
# ─────────────────────────────────────────────────────────────────────────

def ensure_dir(server_path, local_path, download_fn):
    if os.path.exists(local_path):
        print(f'[skip] 已存在: {local_path}')
        return
    if os.path.exists(server_path):
        parent = os.path.dirname(os.path.abspath(local_path))
        os.makedirs(parent, exist_ok=True)
        os.symlink(os.path.abspath(server_path), local_path)
        print(f'[link] {local_path} -> {server_path}')
    else:
        print(f'[download] 本地及服务器均未找到，开始下载到 {local_path} ...')
        download_fn(local_path)

# chunk 数据下载 + 解压
def dl_chunk(local_path):
    from huggingface_hub import hf_hub_download
    os.makedirs('./database', exist_ok=True)
    gz = hf_hub_download(
        repo_id='RenzKa/simlingo', repo_type='dataset',
        filename=CHUNK_FILENAME, local_dir='./database'
    )
    os.makedirs(local_path, exist_ok=True)
    print(f'[extract] 解压 {gz} -> {local_path}')
    subprocess.check_call(['tar', '-xzf', gz, '-C', local_path])

# buckets 下载
def dl_buckets(local_path):
    from huggingface_hub import hf_hub_download
    os.makedirs('./database', exist_ok=True)
    pkl = hf_hub_download(
        repo_id='RenzKa/simlingo', repo_type='dataset',
        filename='buckets_paths.pkl', local_dir='./database'
    )
    os.makedirs(local_path, exist_ok=True)
    dst = os.path.join(local_path, 'buckets_paths.pkl')
    shutil.copy(pkl, dst)
    print(f'[copy] {pkl} -> {dst}')

ensure_dir(SERVER_SIMLINGO_DATA, LOCAL_SIMLINGO_DATA, dl_chunk)
ensure_dir(SERVER_BUCKETS,       LOCAL_BUCKETS,       dl_buckets)
print('完成。')
"
```

### 4. 配置路径

```bash
cd simlingo_training

# 建立软链接
ln -sf ../database database
ln -sf ../data data
ln -sf ../checkpoints/simlingo/.hydra .hydra
```

#### 4.1 修改 simlingo_seed1.yaml 中的模型路径

`config/experiment/simlingo_seed1.yaml` 中默认使用 HuggingFace 远程仓库名：

```yaml
model:
  language_model:
    variant: 'OpenGVLab/InternVL2-1B'
  vision_model:
    variant: 'OpenGVLab/InternVL2-1B'
```

**需要将其替换为本地实际路径**（步骤 2 中已下载或软链接到 `./models/InternVL2-1B`）：

```bash
# 方式一：自动替换（推荐）
# 在 simlingo_training/ 目录下执行：
INTERNVL2_LOCAL_PATH="$(cd .. && pwd)/models/InternVL2-1B"

sed -i "s|'OpenGVLab/InternVL2-1B'|'${INTERNVL2_LOCAL_PATH}'|g" \
    config/experiment/simlingo_seed1.yaml

# 验证替换结果
grep "variant" config/experiment/simlingo_seed1.yaml
```

```bash
# 方式二：手动编辑
# 将 config/experiment/simlingo_seed1.yaml 中两处 variant 改为本地路径：
#
#   language_model:
#     variant: '/your/abs/path/to/models/InternVL2-1B'   ← 改这里
#   vision_model:
#     variant: '/your/abs/path/to/models/InternVL2-1B'   ← 改这里
```

> ⚠️ 必须使用**绝对路径**，相对路径会因 Hydra 改变工作目录而失效。
>
> ⚠️ 替换后，评估结果的输出目录名会从 `OpenGVLab/InternVL2-1B/` 变为本地路径末段目录名（如 `InternVL2-1B/`）。可用以下命令查找结果文件：
> ```bash
> find . -name "per_frame_waypoints_rank_0.json" 2>/dev/null
> ```

### 5. 运行开环评估

```bash
cd simlingo_training

# ⚠️ 将 checkpoint 路径替换为实际绝对路径！
# 查找方法: find .. -name "pytorch_model.pt" 2>/dev/null
# 典型路径: ../checkpoints/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

HYDRA_FULL_ERROR=1 python predict.py \
    'experiment=simlingo_seed1' \
    'checkpoint="path/to/pytorch_model.pt"' \            #绝对路径
    'data_module.base_dataset.data_path=database/simlingo' \
    'data_module.base_dataset.bucket_path=database/bucketsv2_simlingo' \
    'data_module.batch_size=1' \
    'data_module.num_workers=0' \
    'gpus=1'
```

评估结果保存在：
```
simlingo_training/outputs/path/to/predictions/dreamer_results_rank_0.json
```

---

## 使用本地 InternVL2-1B 模型（离线环境必读）

如果你的机器无法访问 HuggingFace（即使设置了镜像），需要提前下载 InternVL2-1B 到本地，并做以下三处修改：

**修改 1：配置文件中的 variant**

```bash
# 将所有实验配置中的远程仓库名替换为本地路径
sed -i "s|'OpenGVLab/InternVL2-1B'|'/your/path/to/models/InternVL2-1B'|g" config/experiment/*.yaml
```

**修改 2：internvl2_utils.py 中的 snapshot_download**

在 `simlingo_training/utils/internvl2_utils.py` 约第 107 行，将：
```python
    cache_dir = f"{cache_root_dir}/{(encoder_variant.split('/')[1])}"
    # get absolute path from workspace dir not wokring dir
    cache_dir = to_absolute_path(cache_dir)
```
替换为：
```python
    if os.path.isdir(encoder_variant):
        cache_dir = encoder_variant
    else:
        cache_dir = f"{cache_root_dir}/{(encoder_variant.split('/')[1])}"
        cache_dir = to_absolute_path(cache_dir)
```

**修改 3（可选）：visualize_single_frame.py 中的模型路径**

修改约第 196 行：
```python
model_name = "/your/path/to/models/InternVL2-1B"
```
或直接使用 `--no_llm` 参数跳过场景描述生成。

> **注意**：修改 variant 后，评估输出目录名也会变化。可用以下命令查找结果文件：
> ```bash
> find . -name "per_frame_waypoints_rank_0.json" 2>/dev/null
> ```

---

## 可视化

### 安装中文字体（避免图表乱码）

```bash
# 安装文泉驿正黑字体
sudo apt-get install fonts-wqy-zenhei -y

# 清除 matplotlib 字体缓存
python -c "import matplotlib; import shutil; shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)"

# 确认字体路径（如果不同需修改脚本中的 font_path）
find / -name "wqy*" 2>/dev/null
```

确保 `visualize_open_loop.py` 和 `visualize_single_frame.py` 中的字体路径正确：
```python
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
```

### 批量可视化（ADE/FDE 分布 + 最优/最差帧对比）

```bash
cd simlingo

python visualize_open_loop.py \
    --data simlingo_training/outputs/path/to/predictions/per_frame_waypoints_rank_0.json \
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
    --waypoints_json simlingo_training/outputs/path/to/predictions/per_frame_waypoints_rank_0.json \
    --frame_id 50 \
    --output eval_results/single_frame
```

加 `--no_llm` 跳过语言描述生成（无网络环境推荐，节省约 30 秒）。

---

## 文件说明

```
simlingo-openloop-eval/
├── predict.py                     # 开环评估主脚本
├── visualize_open_loop.py         # 批量可视化
├── visualize_single_frame.py      # 单帧完整分析
├── run_eval_local.sh              # 闭环评估脚本（需要 CARLA）
├── download_model.py              # 模型权重下载
├── requirements.txt   # Python 3.11 环境依赖列表
├── install_env.sh                 # conda 环境一键安装脚本
└── patches/
    ├── driving_model.patch        # driving.py 修复
    ├── agent_simlingo.patch       # agent 路径修复
    ├── leaderboard_evaluator.patch # 移除 debugpy，添加 quality-level
    └── debug_config.patch         # 实验配置修复
```

---

## 常见问题

**Q: `MissingCUDAException: CUDA_HOME does not exist`**

DeepSpeed 需要 `CUDA_HOME` 环境变量。运行前执行 `export CUDA_HOME=$CONDA_PREFIX`。如果只想跳过 CUDA 编译：`DS_BUILD_OPS=0 python predict.py ...`

**Q: `CUDA error: no kernel image is available for execution on the device`**

GPU 架构不被当前 PyTorch 支持。RTX 50 系列用户必须使用方式一安装环境，或参考方式二中的 Blackwell 适配步骤。

**Q: `Connection to huggingface.co timed out`**

国内无法直接访问 HuggingFace。所有命令前加 `export HF_ENDPOINT=https://hf-mirror.com`。InternVL2-1B 务必提前下载到本地（见步骤 2）。

**Q: `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`**

使用了本地模型路径但代码不兼容。请参考 [使用本地 InternVL2-1B 模型](#使用本地-internvl2-1b-模型离线环境必读) 中的修改 2。

**Q: `FileNotFoundError: .../path/to/simlingo/checkpoints/...`**

checkpoint 路径是占位符，请替换为实际路径：`find .. -name "pytorch_model.pt" 2>/dev/null`

**Q: `No module named 'carla'`（Python 3.11 环境）**

Python 3.11 下安装：`pip install carla==0.9.16`

**Q: 可视化图表中文乱码**

系统缺少中文字体，参考 [安装中文字体](#安装中文字体避免图表乱码) 部分。

**Q: CUDA out of memory**

闭环评估 CARLA 会占用约 5.7GB 显存，总显存不足 12GB 会 OOM。开环评估无需 CARLA，4GB 显存即可。

**Q: `get_original_cwd()` 报错**

需要在 `simlingo_training/` 目录下运行 `predict.py`，且需建立软链接（见步骤 4）。

**Q: 下载速度慢**

所有 HuggingFace 下载均支持 `HF_ENDPOINT=https://hf-mirror.com` 镜像加速。

---

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

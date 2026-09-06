# SimLingo Open-Loop Evaluation

An open-loop evaluation framework based on [SimLingo (CVPR 2025 Spotlight)](https://github.com/RenzKa/simlingo). It supports evaluation on a single consumer-grade GPU without launching the CARLA simulator.

## Project Background

SimLingo is the winning model of the CARLA 2024 Autonomous Driving Challenge. It is built on InternVL2-1B + Qwen2-0.5B, uses camera input only, and directly outputs route waypoints and speed waypoints.

Based on the original SimLingo training framework, this repository:

- Adds an **open-loop evaluation script** (no CARLA required; evaluation is performed directly on an offline dataset)
- Adds **visualization scripts** (ADE/FDE distribution plots and complete single-frame analysis plots)
- Fixes several compatibility issues for local single-machine execution
- Provides patch files that can be conveniently applied to the original repository

## Evaluation Results

Open-loop evaluation results on the SimLingo validation set (chunk_015, 160 frames):

| Metric | Value |
|--------|-------|
| ADE (route waypoints) | 0.0618 m |
| FDE (route waypoints) | 0.1699 m |
| Number of evaluated frames | 160 |

## Hardware Requirements

| Use Case | Minimum VRAM | Notes |
|----------|--------------|-------|
| Open-loop evaluation (inference) | 4GB (RTX 4060 supported) | CARLA not required |
| Closed-loop evaluation (CARLA) | 16GB (RTX 4080 or above required) | CARLA must be launched |

### GPU Architecture Compatibility

| GPU | Architecture | Compute Capability | Environment Requirement |
|-----|--------------|--------------------|-------------------------|
| RTX 4060/4070/4090 | Ada Lovelace | sm_89 | Both Method 1 and Method 2 are supported ✅ |
| RTX 5070/5080/5090 | Blackwell | sm_120 | **Method 1 is required** (Python 3.11 + PyTorch ≥ 2.6) ⚠️ |
| H100/H200 | Hopper | sm_90 | Both Method 1 and Method 2 are supported ✅ |
| B100/B200 | Blackwell | sm_120 | **Method 1 is required** ⚠️ |

---

## Prerequisite: Install Miniconda

> If `conda` is already installed on the server (`conda --version` returns a valid result), skip this step.

Servers often **do not allow writing to `/home`**, so Miniconda should be installed in your own working directory `$WORK`:

```bash
# Download the Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install under $WORK (⚠️ Do not install it under /home!)
bash Miniconda3-latest-Linux-x86_64.sh -b -p $WORK/miniconda3

# Initialize Conda and reload the shell
$WORK/miniconda3/bin/conda init bash
source ~/.bashrc

# Verify the installation
conda --version
```

> All subsequent `conda` commands should be executed in this environment.

---

## Environment Installation

Two installation methods are provided. Choose either one:

### Method 1 (Recommended): Copy the Preconfigured Environment

A complete Conda environment with Python 3.11 + PyTorch 2.11 + CUDA 12.8 has already been configured. It **supports all GPU architectures, including the RTX 50 series**.

**Obtain the archive using one of the following options:**

**1. Server users (recommended):** The archive is already available in a shared directory. Simply copy it; no download is required:

```bash
cp /opt/models/simlingo_packed.tar.gz $WORK/
```

**2. Local/external-network users:** Download it from the cloud drive:

| Platform | Link |
|----------|------|
| BUAA Cloud Drive | [https://bhpan.buaa.edu.cn/link/AAEB1B9E8FDD754BD99ECBDD0A00173B9D] |

Download the following files:
- `simlingo_packed.tar.gz` (approximately 6.6GB) — Conda environment archive
- `install_env.sh` — one-click installation script

**Installation steps:**

```bash
# Make sure Miniconda is installed (see "Prerequisite: Install Miniconda" above)
# Place simlingo_packed.tar.gz and install_env.sh in the same directory, then run:
bash install_env.sh

# If the Conda path is not detected automatically, specify it manually:
bash install_env.sh $WORK/miniconda3
```

After installation:

```bash
conda activate simlingo
export CUDA_HOME=$CONDA_PREFIX
export HF_ENDPOINT=https://hf-mirror.com

# Verify
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Method 2: Build the Environment from Scratch

This method is suitable for Ada Lovelace and earlier GPU architectures (RTX 40 series and below).

```bash
# 1. Clone the original SimLingo repository
git clone https://github.com/RenzKa/simlingo.git
cd simlingo
chmod +x setup_carla.sh
./setup_carla.sh

# 2. Create the Conda environment
conda env create -f environment.yaml
conda activate simlingo

# 3. Install PyTorch
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.0.post2
```

<details>
<summary><b>If you are using an RTX 50-series GPU (Blackwell architecture), click to expand the manual setup steps</b></summary>

RTX 5070/5080/5090 requires PyTorch ≥ 2.6.0 + CUDA 12.8 + Python ≥ 3.10. The original `environment.yaml` is not compatible.

```bash
# 1. Create a new Python 3.11 environment
conda create -n simlingo python=3.11 -y
conda activate simlingo

# 2. Install PyTorch with sm_120 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install CUDA Toolkit and cuDNN
conda install -c nvidia cuda-toolkit -y
pip install nvidia-cudnn-cu12
export CUDA_HOME=$CONDA_PREFIX

# 4. Install project dependencies using the requirements file provided by this repository
pip install -r requirements.txt

# 5. If some packages fail to install, skip them temporarily and install them separately:
#    pip install opencv-python shapely numpy carla==0.9.16
```

</details>

---

## Set CUDA Environment Variables

**Regardless of which installation method you use, the following environment variables must be set.** Otherwise, DeepSpeed will report `CUDA_HOME does not exist`:

```bash
# Set CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX
# Or set it according to the nvcc path:
# export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# Set the HuggingFace mirror endpoint (required for networks in Mainland China)
export HF_ENDPOINT=https://hf-mirror.com

# Recommended: add the settings to .bashrc so they take effect permanently
echo 'export CUDA_HOME=$CONDA_PREFIX' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

---

## Quick Start

### 1. Clone the Repositories and Apply the Patches

```bash
# Clone the original SimLingo repository
git clone https://github.com/RenzKa/simlingo.git

# Clone this repository
git clone https://github.com/lazysmoon/simlingo-openloop-eval.git

# Enter the simlingo directory and apply all patches
cd simlingo
git apply ../simlingo-openloop-eval/patches/driving_model.patch
git apply ../simlingo-openloop-eval/patches/agent_simlingo.patch
git apply ../simlingo-openloop-eval/patches/leaderboard_evaluator.patch

# Copy the evaluation and visualization scripts
cp ../simlingo-openloop-eval/predict.py simlingo_training/
cp ../simlingo-openloop-eval/visualize_open_loop.py .
cp ../simlingo-openloop-eval/visualize_single_frame.py .
cp ../simlingo-openloop-eval/requirements.txt .
```

### 2. Obtain Model Weights and InternVL2-1B

> **Server users:** The model weights are already available in the shared directory. The script automatically detects them, skips downloading, and creates symbolic links directly.  
> **Local users:** If the weights do not exist locally, the script automatically downloads them from HuggingFace.

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd simlingo
python3 -c "
import os, subprocess, sys

# ── Path configuration that can be modified as needed ─────────────────────
SERVER_SIMLINGO_CKPT = '/opt/models/checkpoints/simlingo'   # SimLingo checkpoint directory on the server
SERVER_INTERNVL2     = '/opt/models/InternVL2-1B'           # InternVL2-1B directory on the server
LOCAL_SIMLINGO_CKPT  = './checkpoints/simlingo'             # Local target path (symbolic link or download directory)
LOCAL_INTERNVL2      = './models/InternVL2-1B'              # Local target path (symbolic link or download directory)
# ─────────────────────────────────────────────────────────────────────────

def ensure(server_path, local_path, download_fn):
    if os.path.exists(local_path):
        print(f'[skip] Already exists: {local_path}')
        return
    if os.path.exists(server_path):
        parent = os.path.dirname(os.path.abspath(local_path))
        os.makedirs(parent, exist_ok=True)
        os.symlink(os.path.abspath(server_path), local_path)
        print(f'[link] {local_path} -> {server_path}')
    else:
        print(f'[download] Not found locally or on the server. Downloading to {local_path} ...')
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
print('Done.')
"
```

### 3. Obtain the Validation Dataset

> **Server users:** The dataset is already available in the shared directory. The script automatically detects it, skips downloading, and creates symbolic links directly.  
> **Local users:** If the data does not exist locally, the script automatically downloads and extracts it from HuggingFace.

```bash
export HF_ENDPOINT=https://hf-mirror.com

python3 -c "
import os, subprocess, sys, shutil

# ── Path configuration that can be modified as needed ─────────────────────
SERVER_SIMLINGO_DATA = '/data/datasets/simlingo'             # Extracted SimLingo dataset directory on the server
SERVER_BUCKETS       = '/data/datasets/bucketsv2_simlingo'   # Buckets directory on the server
LOCAL_SIMLINGO_DATA  = './database/simlingo'                 # Local target path
LOCAL_BUCKETS        = './database/bucketsv2_simlingo'       # Local target path

CHUNK_FILENAME = 'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_015.tar.gz'
# ─────────────────────────────────────────────────────────────────────────

def ensure_dir(server_path, local_path, download_fn):
    if os.path.exists(local_path):
        print(f'[skip] Already exists: {local_path}')
        return
    if os.path.exists(server_path):
        parent = os.path.dirname(os.path.abspath(local_path))
        os.makedirs(parent, exist_ok=True)
        os.symlink(os.path.abspath(server_path), local_path)
        print(f'[link] {local_path} -> {server_path}')
    else:
        print(f'[download] Not found locally or on the server. Downloading to {local_path} ...')
        download_fn(local_path)

# Download and extract the chunk data
def dl_chunk(local_path):
    from huggingface_hub import hf_hub_download
    os.makedirs('./database', exist_ok=True)
    gz = hf_hub_download(
        repo_id='RenzKa/simlingo', repo_type='dataset',
        filename=CHUNK_FILENAME, local_dir='./database'
    )
    os.makedirs(local_path, exist_ok=True)
    print(f'[extract] Extracting {gz} -> {local_path}')
    subprocess.check_call(['tar', '-xzf', gz, '-C', local_path])

# Download the buckets file
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
print('Done.')
"
```

### 4. Configure Paths

```bash
cd simlingo_training

# Create symbolic links
ln -sf ../database database
ln -sf ../data data
ln -sf ../checkpoints/simlingo/.hydra .hydra
```

#### 4.1 Modify the Model Path in simlingo_seed1.yaml

By default, `config/experiment/simlingo_seed1.yaml` uses the remote HuggingFace repository name:

```yaml
model:
  language_model:
    variant: 'OpenGVLab/InternVL2-1B'
  vision_model:
    variant: 'OpenGVLab/InternVL2-1B'
```

**Replace it with the actual local path** (downloaded or symbolically linked to `./models/InternVL2-1B` in Step 2):

```bash
# Method 1: Automatic replacement (recommended)
# Run this command inside the simlingo_training/ directory:
INTERNVL2_LOCAL_PATH="$(cd .. && pwd)/models/InternVL2-1B"

sed -i "s|'OpenGVLab/InternVL2-1B'|'${INTERNVL2_LOCAL_PATH}'|g" \
    config/experiment/simlingo_seed1.yaml

# Verify the replacement
grep "variant" config/experiment/simlingo_seed1.yaml
```

```bash
# Method 2: Edit manually
# Change both variant entries in config/experiment/simlingo_seed1.yaml to the local path:
#
#   language_model:
#     variant: '/your/abs/path/to/models/InternVL2-1B'   <- Change this
#   vision_model:
#     variant: '/your/abs/path/to/models/InternVL2-1B'   <- Change this
```

> ⚠️ You **must use an absolute path**. A relative path may become invalid because Hydra changes the working directory.
>
> ⚠️ After replacing the path, the output directory name for evaluation results will change from `OpenGVLab/InternVL2-1B/` to the final directory name of the local path (for example, `InternVL2-1B/`). Use the following command to locate the result file:
>
> ```bash
> find . -name "per_frame_waypoints_rank_0.json" 2>/dev/null
> ```

### 5. Run Open-Loop Evaluation

```bash
cd simlingo_training

# ⚠️ Replace the checkpoint path with the actual absolute path!
# Search command: find .. -name "pytorch_model.pt" 2>/dev/null
# Typical path: ../checkpoints/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

HYDRA_FULL_ERROR=1 python predict.py \
    'experiment=simlingo_seed1' \
    'checkpoint="path/to/pytorch_model.pt"' \            # Absolute path
    'data_module.base_dataset.data_path=database/simlingo' \
    'data_module.base_dataset.bucket_path=database/bucketsv2_simlingo' \
    'data_module.batch_size=1' \
    'data_module.num_workers=0' \
    'gpus=1'
```

Evaluation results are saved to:

```text
simlingo_training/outputs/path/to/predictions/dreamer_results_rank_0.json
```

---

## Using a Local InternVL2-1B Model (Required Reading for Offline Environments)

If your machine cannot access HuggingFace, even after configuring the mirror endpoint, download InternVL2-1B locally in advance and make the following three modifications:

**Modification 1: `variant` in the configuration files**

```bash
# Replace the remote repository name with the local path in all experiment configurations
sed -i "s|'OpenGVLab/InternVL2-1B'|'/your/path/to/models/InternVL2-1B'|g" config/experiment/*.yaml
```

**Modification 2: `snapshot_download` in `internvl2_utils.py`**

At approximately line 107 in `simlingo_training/utils/internvl2_utils.py`, replace:

```python
    cache_dir = f"{cache_root_dir}/{(encoder_variant.split('/')[1])}"
    # get absolute path from workspace dir not wokring dir
    cache_dir = to_absolute_path(cache_dir)
```

with:

```python
    if os.path.isdir(encoder_variant):
        cache_dir = encoder_variant
    else:
        cache_dir = f"{cache_root_dir}/{(encoder_variant.split('/')[1])}"
        cache_dir = to_absolute_path(cache_dir)
```

**Modification 3 (optional): Model path in `visualize_single_frame.py`**

Modify approximately line 196:

```python
model_name = "/your/path/to/models/InternVL2-1B"
```

Alternatively, use the `--no_llm` option to skip scene-description generation.

> **Note:** After changing `variant`, the evaluation output directory name will also change. Use the following command to locate the result file:
>
> ```bash
> find . -name "per_frame_waypoints_rank_0.json" 2>/dev/null
> ```

---

## Visualization

### Install a CJK Font to Avoid Garbled Plot Text

```bash
# Install the WenQuanYi Zen Hei font
sudo apt-get install fonts-wqy-zenhei -y

# Clear the Matplotlib font cache
python -c "import matplotlib; import shutil; shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)"

# Verify the font path (if it differs, update font_path in the scripts)
find / -name "wqy*" 2>/dev/null
```

Make sure the font path in `visualize_open_loop.py` and `visualize_single_frame.py` is correct:

```python
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
```

### Batch Visualization (ADE/FDE Distributions + Best/Worst Frame Comparison)

```bash
cd simlingo

python visualize_open_loop.py \
    --data simlingo_training/outputs/path/to/predictions/per_frame_waypoints_rank_0.json \
    --output eval_results/visualization
```

Output files:

| File | Description |
|------|-------------|
| `01_summary.png` | ADE/FDE summary statistics |
| `02_metrics_distribution.png` | ADE/FDE distribution histograms |
| `03_worst_frames.png` | 12 frames with the worst predictions |
| `04_best_frames.png` | 12 frames with the best predictions |
| `05_random_frames.png` | 12 randomly selected frames |

### Complete Single-Frame Analysis (Image + Waypoints + Scene Description)

```bash
python visualize_single_frame.py \
    --waypoints_json simlingo_training/outputs/path/to/predictions/per_frame_waypoints_rank_0.json \
    --frame_id 50 \
    --output eval_results/single_frame
```

Add `--no_llm` to skip language-description generation. This is recommended in offline environments and saves approximately 30 seconds.

---

## File Description

```text
simlingo-openloop-eval/
├── predict.py                     # Main open-loop evaluation script
├── visualize_open_loop.py         # Batch visualization
├── visualize_single_frame.py      # Complete single-frame analysis
├── run_eval_local.sh              # Closed-loop evaluation script (requires CARLA)
├── download_model.py              # Model checkpoint downloader
├── requirements.txt               # Dependency list for the Python 3.11 environment
├── install_env.sh                 # One-click Conda environment installation script
└── patches/
    ├── driving_model.patch        # Fixes for driving.py
    ├── agent_simlingo.patch       # Agent path fixes
    ├── leaderboard_evaluator.patch # Removes debugpy and adds quality-level
    └── debug_config.patch         # Experiment configuration fixes
```

---

## FAQ

**Q: `MissingCUDAException: CUDA_HOME does not exist`**

DeepSpeed requires the `CUDA_HOME` environment variable. Run `export CUDA_HOME=$CONDA_PREFIX` before execution. If you only want to skip CUDA compilation, use: `DS_BUILD_OPS=0 python predict.py ...`

**Q: `CUDA error: no kernel image is available for execution on the device`**

The current PyTorch installation does not support your GPU architecture. RTX 50-series users must use Method 1 to install the environment, or follow the Blackwell adaptation steps in Method 2.

**Q: `Connection to huggingface.co timed out`**

HuggingFace may not be directly accessible from networks in Mainland China. Add `export HF_ENDPOINT=https://hf-mirror.com` before all related commands. Make sure InternVL2-1B is downloaded locally in advance (see Step 2).

**Q: `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`**

You are using a local model path, but the code is not compatible with local paths. See Modification 2 in [Using a Local InternVL2-1B Model](#using-a-local-internvl2-1b-model-required-reading-for-offline-environments).

**Q: `FileNotFoundError: .../path/to/simlingo/checkpoints/...`**

The checkpoint path is a placeholder. Replace it with the actual path:

```bash
find .. -name "pytorch_model.pt" 2>/dev/null
```

**Q: `No module named 'carla'` (Python 3.11 environment)**

Install CARLA for Python 3.11:

```bash
pip install carla==0.9.16
```

**Q: Garbled text in visualization plots**

The system is missing the required font. See [Install a CJK Font to Avoid Garbled Plot Text](#install-a-cjk-font-to-avoid-garbled-plot-text).

**Q: CUDA out of memory**

Closed-loop CARLA evaluation uses approximately 5.7GB of VRAM, and the total VRAM requirement may exceed 16GB. Open-loop evaluation does not require CARLA and can run with 4GB of VRAM.

**Q: `get_original_cwd()` error**

Run `predict.py` from the `simlingo_training/` directory and make sure the required symbolic links have been created (see Step 4).

**Q: Slow download speed**

All HuggingFace downloads support mirror acceleration through `HF_ENDPOINT=https://hf-mirror.com`.

---

## Acknowledgments

This project is built on [SimLingo](https://github.com/RenzKa/simlingo) (CVPR 2025 Spotlight). We thank the original authors for making their work open source.

## Citation

```bibtex
@inproceedings{renz2025simlingo,
  title={SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author={Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle={CVPR},
  year={2025}
}
```

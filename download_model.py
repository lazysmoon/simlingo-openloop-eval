# 从 HuggingFace 下载模型（约 2-3GB）
# 在 Python 中：
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # 默认10秒，调大

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="RenzKa/simlingo",
    local_dir="./checkpoints/simlingo",
    resume_download=True,   # 断点续传，断了重跑就行
    max_workers=2,          # 减少并发，降低超时概率
)
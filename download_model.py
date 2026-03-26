# 从 HuggingFace 下载模型（约 2-3GB）
# 在 Python 中：
from huggingface_hub import snapshot_download
snapshot_download(repo_id="RenzKa/simlingo", local_dir="./checkpoints/simlingo")
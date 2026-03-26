"""
SimLingo 开环评估脚本（使用 Hydra 正确初始化）
===============================================
用法（在 simlingo_training 目录下运行）：

    cd ~/Document/python_code/VLA/simlingo/simlingo_training
    conda activate simlingo

    python predict.py \
        experiment=simlingo_seed1 \
        checkpoint=/home/robot/Document/python_code/VLA/simlingo/checkpoints/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
        data_module.base_dataset.data_path=/home/robot/Document/python_code/VLA/simlingo/database/simlingo \
        data_module.batch_size=1 \
        data_module.num_workers=0 \
        gpus=1

输出：
    ADE/FDE 结果打印到终端，预测文件保存到 checkpoint 目录下的 predictions/
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from transformers import AutoProcessor
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from simlingo_training.config import TrainConfig


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    # 关闭 wandb
    os.environ["WANDB_MODE"] = "disabled"

    print("=" * 60)
    print(" SimLingo 开环评估")
    print("=" * 60)
    print(f" checkpoint : {cfg.checkpoint}")
    print(f" data_path  : {cfg.data_module.base_dataset.data_path}")
    print(f" batch_size : {cfg.data_module.batch_size}")
    print("=" * 60)

    # ── 关闭数据增强 ──────────────────────────────────────────────
    cfg.data_module.base_dataset.img_shift_augmentation = False
    cfg.data_module.base_dataset.qa_augmentation = False

    # ── processor ────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant, trust_remote_code=True
    )

    # ── DataModule ───────────────────────────────────────────────
    print("[1/3] 初始化 DataModule ...")
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )

    # ── Model ────────────────────────────────────────────────────
    print("[2/3] 初始化模型 ...")
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=None,
        _recursive_=False,
    )
    #print(model)
    # ── 加载权重 ──────────────────────────────────────────────────
    if cfg.checkpoint is not None:
        print(f"[2/3] 加载权重: {cfg.checkpoint}")
        if os.path.isdir(cfg.checkpoint):
            # DeepSpeed ZeRO checkpoint 目录
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            # 单文件 pytorch_model.pt
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "module" in state_dict:
                state_dict = state_dict["module"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [!] 缺失参数 {len(missing)} 个（前5个）: {missing[:5]}")
        if unexpected:
            print(f"  [!] 多余参数 {len(unexpected)} 个（前5个）: {unexpected[:5]}")
        print("  [✓] 权重加载完成")
    else:
        print("[!] 未指定 checkpoint，使用随机初始化权重（仅供测试）")

    # ── Trainer.predict() ────────────────────────────────────────
    print("[3/3] 开始开环推理（predict mode）...")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
        precision=cfg.get("precision", 32),
    )

    # predict 会自动调用：
    #   predict_step()         → 模型推理，收集预测路点
    #   on_predict_epoch_end() → 计算 ADE/FDE，保存结果文件
    data_module.setup('fit')
    trainer.predict(
        model=model,
        dataloaders=data_module.val_dataloader(),
        ckpt_path=None,  # 已手动加载权重
    )

    print("\n[✓] 开环评估完成！")
    print("    ADE/FDE 指标已打印在上方日志中")
    print("    语言预测结果保存在 output 目录下的 predictions/ 文件夹")


if __name__ == "__main__":
    main()

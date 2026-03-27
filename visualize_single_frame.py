"""
SimLingo 单帧完整可视化
========================
对指定帧生成：
  - 左上：前向摄像头图像
  - 右上：路点预测 vs GT 对比图
  - 下方：InternVL2 对场景的语言描述

用法：
    cd ~/Document/python_code/VLA/simlingo
    conda activate simlingo

    python visualize_single_frame.py \
        --waypoints_json simlingo_training/outputs/OpenGVLab/InternVL2-1B/predictions/per_frame_waypoints_rank_0.json \
        --frame_id 0 \
        --output eval_results/single_frame
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import os
import matplotlib
from matplotlib import font_manager

# 定义系统中的中文字体路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

# 备用字体路径（可根据实际环境添加）
fallback_paths = [
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'
]

# 寻找可用的中文字体
valid_font_path = None
if os.path.exists(font_path):
    valid_font_path = font_path
else:
    for fp in fallback_paths:
        if os.path.exists(fp):
            valid_font_path = fp
            break

if valid_font_path:
    # 动态加载字体并获取准确的内部名称
    font_manager.fontManager.addfont(valid_font_path)
    prop = font_manager.FontProperties(fname=valid_font_path)
    matplotlib.rcParams['font.family'] = prop.get_name()
    print(f"[i] 成功加载中文字体: {prop.get_name()} ({valid_font_path})")
else:
    print("[!] 警告: 未找到指定的中文字体文件，图片可能出现乱码。")

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import textwrap


# ── 1. 数据加载 ────────────────────────────────────────────────────────────────

def load_waypoints(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_route_dir(frame_path: str) -> Path:
    """从 path 字符串还原路线目录"""
    # path 格式: .../Town13_Rep0_998_route0_01_11_...
    # 需要找到对应的 rgb 目录
    p = Path(frame_path)
    # 向上找到包含 rgb 子目录的那一级
    for parent in [p, p.parent, p.parent.parent]:
        if (parent / 'rgb').exists():
            return parent
    # fallback: 直接用路径
    return p


def find_frame_image(route_dir: Path, frame_idx_in_route: int = None):
    """找到路线目录下的一张 RGB 图像"""
    rgb_dir = route_dir / 'rgb'
    if not rgb_dir.exists():
        return None
    images = sorted(rgb_dir.glob('*.jpg')) + sorted(rgb_dir.glob('*.png'))
    if not images:
        return None
    if frame_idx_in_route is not None and frame_idx_in_route < len(images):
        return images[frame_idx_in_route]
    return images[len(images) // 2]  # 取中间帧


def load_measurement(route_dir: Path, img_path: Path):
    """加载对应帧的 measurement"""
    frame_name = img_path.stem  # e.g. "0042"
    meas_dir = route_dir / 'measurements'
    meas_file = meas_dir / f'{frame_name}.json.gz'
    if not meas_file.exists():
        meas_file = meas_dir / f'{frame_name}.json'
    if not meas_file.exists():
        return None
    try:
        if str(meas_file).endswith('.gz'):
            with gzip.open(meas_file, 'rt') as f:
                return json.load(f)
        else:
            with open(meas_file) as f:
                return json.load(f)
    except Exception:
        return None


# ── 2. 场景描述生成（InternVL2） ───────────────────────────────────────────────

SCENE_PROMPT = (
    "You are an autonomous driving assistant. "
    "Look at this front-facing camera image from a self-driving car. "
    "Briefly describe: (1) road type and curvature, (2) traffic conditions, "
    "(3) weather/lighting, (4) what the car should do next. "
    "Keep it under 60 words."
)


# def generate_scene_description(img_path: Path, device='cuda') -> str:
#     """用 InternVL2-1B 生成场景描述"""
#     print("[i] 加载 InternVL2-1B 生成场景描述...")
#     try:
#         model_name = "OpenGVLab/InternVL2-1B"
#         cache_dir = "simlingo_training/pretrained/InternVL2-1B"

#         processor = AutoProcessor.from_pretrained(
#             model_name, trust_remote_code=True,
#             cache_dir=cache_dir
#         )
#         model = AutoModel.from_pretrained(
#             model_name, trust_remote_code=True,
#             torch_dtype=torch.bfloat16,
#             cache_dir=cache_dir
#         ).to(device).eval()

#         img = Image.open(img_path).convert('RGB')

#         # InternVL2 的对话格式
#         if hasattr(processor, 'tokenizer'):
#             tokenizer = processor.tokenizer
#         else:
#             tokenizer = processor

#         # 构造输入
#         pixel_values = processor.image_processor(
#             images=img, return_tensors='pt'
#         ).pixel_values.to(device, dtype=torch.bfloat16)

#         question = f"<image>\n{SCENE_PROMPT}"
#         inputs = tokenizer(question, return_tensors='pt').to(device)

#         with torch.no_grad():
#             outputs = model.generate(
#                 pixel_values=pixel_values,
#                 input_ids=inputs.input_ids,
#                 attention_mask=inputs.attention_mask,
#                 max_new_tokens=120,
#                 do_sample=False,
#             )

#         response = tokenizer.decode(
#             outputs[0][inputs.input_ids.shape[1]:],
#             skip_special_tokens=True
#         ).strip()

#         del model
#         torch.cuda.empty_cache()
#         return response

#     except Exception as e:
#         print(f"[!] 语言描述生成失败: {e}")
#         return "（场景描述生成失败，请检查模型路径）"

def generate_scene_description(img_path: Path, device='cuda') -> str:
    """用 InternVL2-1B 生成场景描述"""
    print("[i] 加载 InternVL2-1B 生成场景描述...")
    try:
        from transformers import AutoTokenizer
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        model_name = "OpenGVLab/InternVL2-1B"
        cache_dir = "simlingo_training/pretrained/InternVL2-1B"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16, cache_dir=cache_dir
        ).to(device).eval()

        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

        img = Image.open(img_path).convert('RGB')
        pixel_values = transform(img).unsqueeze(0).to(device, dtype=torch.bfloat16)

        generation_config = dict(max_new_tokens=150, do_sample=False)
        question = f"<image>\n{SCENE_PROMPT}"

        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, question, generation_config)

        del model
        torch.cuda.empty_cache()
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"（生成失败: {e}）"
# ── 3. 可视化 ──────────────────────────────────────────────────────────────────

COMMAND_MAP = {
    1: "左转", 2: "右转", 3: "直行",
    4: "跟随路线", 5: "变道左", 6: "变道右"
}


def plot_waypoints(ax, pred, gt, title="路点预测 vs GT"):
    pred = np.array(pred)
    gt   = np.array(gt)

    ax.plot(gt[:, 1],   gt[:, 0],   'o-',  color='#2ecc71', lw=2,
            ms=5, label='GT 路径', zorder=3)
    ax.plot(pred[:, 1], pred[:, 0], 's--', color='#e74c3c', lw=2,
            ms=5, label='预测路径', zorder=4)
    ax.scatter([0], [0], s=150, color='#3498db', zorder=5,
               marker='*', label='自车位置')

    for i in range(0, min(len(pred), len(gt)), 4):
        ax.plot([gt[i,1], pred[i,1]], [gt[i,0], pred[i,0]],
                color='gray', lw=0.8, alpha=0.5)

    n = min(len(pred), len(gt))
    ade = float(np.mean(np.linalg.norm(pred[:n] - gt[:n], axis=-1)))
    fde = float(np.linalg.norm(pred[-1] - gt[-1]))

    ax.set_xlabel('横向 (m)', fontsize=10)
    ax.set_ylabel('纵向 (m)', fontsize=10)
    ax.set_title(f'{title}\nADE={ade:.4f}m   FDE={fde:.4f}m', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    #ax.invert_xaxis()
    x_range = max(2.0, np.max(np.abs(np.concatenate([pred[:,1], gt[:,1]]))) * 2 + 0.5)
    ax.set_xlim(-x_range, x_range)
    return ade, fde


def make_single_frame_figure(
    frame_data: dict,
    img_path: Path,
    measurement: dict,
    scene_description: str,
    save_path: str,
):
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.patch.set_facecolor('#f8f9fa')

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[2.5, 1],
    )

    # ── 左上：摄像头图像 ────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    if img_path and img_path.exists():
        img = Image.open(img_path).convert('RGB')
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, '图像未找到', ha='center', va='center',
                    transform=ax_img.transAxes, fontsize=14, color='gray')
    ax_img.axis('off')

    # 叠加基本信息
    if measurement:
        speed   = measurement.get('speed', 0)
        command = measurement.get('command', 4)
        cmd_str = COMMAND_MAP.get(command, f'命令{command}')
        info_str = f"速度: {speed:.1f} m/s   导航: {cmd_str}"
        ax_img.set_title(f"帧 #{frame_data['frame_id']}  —  {info_str}",
                         fontsize=11, pad=6)
    else:
        ax_img.set_title(f"帧 #{frame_data['frame_id']}", fontsize=11)

    # ── 右上：路点预测对比 ──────────────────────────────────
    ax_wp = fig.add_subplot(gs[0, 1])
    ade, fde = plot_waypoints(
        ax_wp,
        frame_data['route_pred'],
        frame_data['route_gt'],
        title="路径路点预测 vs GT"
    )

    # ── 下方左：速度路点对比 ────────────────────────────────
    ax_sp = fig.add_subplot(gs[1, 0])
    sp_pred = np.array(frame_data['speed_wp_pred'])
    sp_gt   = np.array(frame_data['speed_wp_gt'])
    t = np.arange(len(sp_pred)) * 0.2  # 5Hz → 每步 0.2s

    # 速度路点是 2D (累积距离) 或 1D，统一取第一维
    sp_pred_v = sp_pred[:, 0] if sp_pred.ndim == 2 else sp_pred
    sp_gt_v   = sp_gt[:, 0]   if sp_gt.ndim == 2   else sp_gt

    ax_sp.plot(t, sp_gt_v,   'o-', color='#2ecc71', lw=2, ms=4,
               label='GT 速度路点')
    ax_sp.plot(t, sp_pred_v, 's--', color='#e74c3c', lw=2, ms=4,
               label='预测速度路点')
    ax_sp.fill_between(t, sp_gt_v, sp_pred_v, alpha=0.15, color='gray')
    ax_sp.set_xlabel('时间 (s)', fontsize=10)
    ax_sp.set_ylabel('累积距离 (m)', fontsize=10)
    ax_sp.set_title('速度路点预测 vs GT', fontsize=10)
    ax_sp.legend(fontsize=9)
    ax_sp.grid(True, alpha=0.3)

    # ── 下方右：场景描述文字框 ──────────────────────────────
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis('off')

    wrapped = textwrap.fill(scene_description, width=70)
    ax_txt.text(
        0.05, 0.95, "🤖 模型场景描述 (InternVL2-1B):",
        transform=ax_txt.transAxes,
        fontsize=10, fontweight='bold', va='top', color='#2c3e50'
    )
    ax_txt.text(
        0.05, 0.80, wrapped,
        transform=ax_txt.transAxes,
        fontsize=10, va='top', color='#34495e',
        wrap=True, linespacing=1.6
    )
    ax_txt.text(
        0.05, 0.05,
        f"ADE: {ade:.4f} m   |   FDE: {fde:.4f} m   |   样本路径: ...{str(img_path)[-50:]}",
        transform=ax_txt.transAxes,
        fontsize=8, va='bottom', color='gray'
    )
    ax_txt.set_facecolor('#ecf0f1')
    for spine in ax_txt.spines.values():
        spine.set_edgecolor('#bdc3c7')
        spine.set_linewidth(1)

    fig.suptitle(
        'SimLingo 开环评估 — 单帧完整分析',
        fontsize=14, fontweight='bold', y=0.98
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] 单帧可视化图保存到: {save_path}")


# ── 4. 主函数 ──────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--waypoints_json', required=True,
#                         help='per_frame_waypoints_rank_0.json 路径')
#     parser.add_argument('--frame_id', type=int, default=0,
#                         help='要可视化的帧编号（0-based）')
#     parser.add_argument('--output', default='eval_results/single_frame',
#                         help='输出目录')
#     parser.add_argument('--no_llm', action='store_true',
#                         help='跳过语言描述生成（节省时间）')
#     parser.add_argument('--device', default='cuda')
#     args = parser.parse_args()

#     # 加载路点数据
#     frames = load_waypoints(args.waypoints_json)
#     if args.frame_id >= len(frames):
#         print(f"[!] frame_id {args.frame_id} 超出范围（共 {len(frames)} 帧）")
#         return

#     frame = frames[args.frame_id]
#     print(f"[i] 处理帧 #{frame['frame_id']}，路径: {frame['path'][-60:]}")

#     # 找到图像
#     route_dir = get_route_dir(frame['path'])
#     img_path  = find_frame_image(route_dir)
#     print(f"[i] 图像路径: {img_path}")

#     # 加载 measurement
#     measurement = None
#     if img_path:
#         measurement = load_measurement(route_dir, img_path)

#     # 生成场景描述
#     if args.no_llm:
#         scene_desc = "（已跳过语言描述生成，使用 --no_llm 标志）"
#     elif img_path and img_path.exists():
#         scene_desc = generate_scene_description(img_path, device=args.device)
#     else:
#         scene_desc = "（图像未找到，无法生成描述）"

#     print(f"[i] 场景描述: {scene_desc}")

#     # 生成可视化
#     save_path = os.path.join(
#         args.output, f"frame_{args.frame_id:04d}_analysis.png"
#     )
#     make_single_frame_figure(
#         frame_data=frame,
#         img_path=img_path,
#         measurement=measurement,
#         scene_description=scene_desc,
#         save_path=save_path,
#     )

#     # 同时打印数值
#     pred = np.array(frame['route_pred'])
#     gt   = np.array(frame['route_gt'])
#     n    = min(len(pred), len(gt))
#     ade  = float(np.mean(np.linalg.norm(pred[:n] - gt[:n], axis=-1)))
#     fde  = float(np.linalg.norm(pred[-1] - gt[-1]))
#     print(f"\n{'='*50}")
#     print(f"  帧编号:  #{frame['frame_id']}")
#     print(f"  ADE:     {ade:.4f} m")
#     print(f"  FDE:     {fde:.4f} m")
#     if measurement:
#         print(f"  速度:    {measurement.get('speed', 'N/A'):.2f} m/s")
#         cmd = measurement.get('command', 4)
#         print(f"  导航命令: {COMMAND_MAP.get(cmd, cmd)}")
#     print(f"  场景描述: {scene_desc[:100]}...")
#     print(f"{'='*50}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--waypoints_json', default=None,
                        help='per_frame_waypoints_rank_0.json 路径')
    parser.add_argument('--route_dir', default=None,
                        help='直接指定路线目录，不依赖 waypoints_json')
    parser.add_argument('--frame_id', type=int, default=0,
                        help='帧编号')
    parser.add_argument('--output', default='eval_results/single_frame')
    parser.add_argument('--no_llm', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # ── 模式一：直接指定路线目录（不需要 waypoints_json）──
    if args.route_dir:
        route_dir = Path(args.route_dir)
        img_path = find_frame_image(route_dir, args.frame_id)
        measurement = load_measurement(route_dir, img_path) if img_path else None

        # 从 measurements 里取 GT 路点
        gt_route = []
        if measurement and 'route' in measurement:
            gt_route = measurement['route'][:20]
        while len(gt_route) < 20:
            gt_route.append(gt_route[-1] if gt_route else [0, 0])

        # 构造假的 frame_data（没有模型预测，pred 和 gt 一样，仅展示场景）
        frame = {
            'frame_id': args.frame_id,
            'path': str(img_path) if img_path else str(route_dir),
            'route_pred': gt_route,   # 无预测时用 GT 占位
            'route_gt':   gt_route,
            'speed_wp_pred': [[i*0.5, 0] for i in range(10)],
            'speed_wp_gt':   [[i*0.5, 0] for i in range(10)],
        }

        if args.no_llm:
            scene_desc = "（已跳过语言描述生成）"
        elif img_path and img_path.exists():
            scene_desc = generate_scene_description(img_path, device=args.device)
        else:
            scene_desc = "（图像未找到）"

        save_path = os.path.join(
            args.output,
            f"{route_dir.name}_frame{args.frame_id:04d}.png"
        )
        make_single_frame_figure(
            frame_data=frame,
            img_path=img_path,
            measurement=measurement,
            scene_description=scene_desc,
            save_path=save_path,
        )
        print(f"[✓] 保存到: {save_path}")
        return

    # ── 模式二：从 waypoints_json 读取（原来的逻辑）──
    if args.waypoints_json is None:
        print("[!] 请提供 --waypoints_json 或 --route_dir")
        return

    frames = load_waypoints(args.waypoints_json)
    if args.frame_id >= len(frames):
        print(f"[!] frame_id {args.frame_id} 超出范围（共 {len(frames)} 帧）")
        return

    frame = frames[args.frame_id]
    print(f"[i] 处理帧 #{frame['frame_id']}，路径: {frame['path'][-60:]}")

    route_dir = get_route_dir(frame['path'])
    img_path  = find_frame_image(route_dir)
    measurement = load_measurement(route_dir, img_path) if img_path else None

    if args.no_llm:
        scene_desc = "（已跳过语言描述生成，使用 --no_llm 标志）"
    elif img_path and img_path.exists():
        scene_desc = generate_scene_description(img_path, device=args.device)
    else:
        scene_desc = "（图像未找到，无法生成描述）"

    print(f"[i] 场景描述: {scene_desc}")

    save_path = os.path.join(
        args.output, f"frame_{args.frame_id:04d}_analysis.png"
    )
    make_single_frame_figure(
        frame_data=frame,
        img_path=img_path,
        measurement=measurement,
        scene_description=scene_desc,
        save_path=save_path,
    )

    pred = np.array(frame['route_pred'])
    gt   = np.array(frame['route_gt'])
    n    = min(len(pred), len(gt))
    ade  = float(np.mean(np.linalg.norm(pred[:n] - gt[:n], axis=-1)))
    fde  = float(np.linalg.norm(pred[-1] - gt[-1]))
    print(f"\n{'='*50}")
    print(f"  帧编号:  #{frame['frame_id']}")
    print(f"  ADE:     {ade:.4f} m")
    print(f"  FDE:     {fde:.4f} m")
    if measurement:
        print(f"  速度:    {measurement.get('speed', 'N/A'):.2f} m/s")
        cmd = measurement.get('command', 4)
        print(f"  导航命令: {COMMAND_MAP.get(cmd, cmd)}")
    print(f"  场景描述: {scene_desc[:100]}...")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()

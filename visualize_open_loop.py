"""
SimLingo 开环评估可视化脚本
============================
读取 per_frame_waypoints_rank_0.json，生成：
1. 单帧路点对比图（预测 vs GT）
2. 多帧 ADE/FDE 分布图
3. 所有帧的汇总 HTML 报告

用法：
    cd ~/Document/python_code/VLA/simlingo
    python visualize_open_loop.py \
        --data simlingo_training/outputs/OpenGVLab/InternVL2-1B/predictions/per_frame_waypoints_rank_0.json \
        --output eval_results/visualization
"""

import json
import argparse
import numpy as np
# import os

# import matplotlib
# from matplotlib import font_manager
# font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
# matplotlib.rcParams['font.family'] = 'Noto Sans CJK SC'
# matplotlib.rcParams['axes.unicode_minus'] = False
# matplotlib.rcParams['font.family'] = ['Noto Sans CJK SC', 'AR PL UMing TW MBE', 'DejaVu Sans']
# matplotlib.rcParams['axes.unicode_minus'] = False
# matplotlib.use('Agg')  # 无显示器模式
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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path


def load_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    print(f"[i] 加载 {len(data)} 帧数据")
    return data


def compute_metrics(data):
    ades, fdes = [], []
    for frame in data:
        pred = np.array(frame['route_pred'])   # [20, 2]
        gt   = np.array(frame['route_gt'])     # [20, 2]
        min_len = min(len(pred), len(gt))
        pred, gt = pred[:min_len], gt[:min_len]
        ade = np.mean(np.linalg.norm(pred - gt, axis=-1))
        fde = np.linalg.norm(pred[-1] - gt[-1])
        ades.append(ade)
        fdes.append(fde)
    return np.array(ades), np.array(fdes)


def plot_single_frame(frame, ax, title=""):
    pred  = np.array(frame['route_pred'])
    gt    = np.array(frame['route_gt'])
    sp    = np.array(frame['speed_wp_pred'])
    sg    = np.array(frame['speed_wp_gt'])

    # 路径路点
    ax.plot(gt[:, 1],   gt[:, 0],   'o-', color='#2ecc71', linewidth=2,
            markersize=4, label='GT路径', zorder=3)
    ax.plot(pred[:, 1], pred[:, 0], 's--', color='#e74c3c', linewidth=2,
            markersize=4, label='预测路径', zorder=4)

    # 起点标记
    ax.scatter([0], [0], s=120, color='#3498db', zorder=5, marker='*', label='自车位置')

    # 误差连线（每隔5个点画一条）
    for i in range(0, min(len(pred), len(gt)), 5):
        ax.plot([gt[i, 1], pred[i, 1]], [gt[i, 0], pred[i, 0]],
                color='gray', linewidth=0.8, alpha=0.5, zorder=2)

    ade = np.mean(np.linalg.norm(pred[:min(len(pred),len(gt))] -
                                  gt[:min(len(pred),len(gt))], axis=-1))
    fde = np.linalg.norm(pred[-1] - gt[-1])

    ax.set_xlabel('横向 (m)', fontsize=9)
    ax.set_ylabel('纵向 (m)', fontsize=9)
    ax.set_title(f'{title}\nADE={ade:.3f}m  FDE={fde:.3f}m', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()  # 左右符合驾驶视角


def plot_metrics_distribution(ades, fdes, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('SimLingo 开环评估指标分布', fontsize=13, fontweight='bold')

    # ADE 分布
    ax = axes[0]
    ax.hist(ades, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(ades), color='red', linestyle='--', linewidth=2,
               label=f'均值={np.mean(ades):.4f}m')
    ax.axvline(np.median(ades), color='orange', linestyle='--', linewidth=2,
               label=f'中位数={np.median(ades):.4f}m')
    ax.set_xlabel('ADE (m)')
    ax.set_ylabel('帧数')
    ax.set_title('路径 ADE 分布')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # FDE 分布
    ax = axes[1]
    ax.hist(fdes, bins=30, color='#e74c3c', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(fdes), color='red', linestyle='--', linewidth=2,
               label=f'均值={np.mean(fdes):.4f}m')
    ax.axvline(np.median(fdes), color='orange', linestyle='--', linewidth=2,
               label=f'中位数={np.median(fdes):.4f}m')
    ax.set_xlabel('FDE (m)')
    ax.set_ylabel('帧数')
    ax.set_title('路径 FDE 分布')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ADE 随帧变化
    ax = axes[2]
    ax.plot(ades, color='#3498db', alpha=0.6, linewidth=0.8, label='ADE')
    ax.plot(fdes, color='#e74c3c', alpha=0.6, linewidth=0.8, label='FDE')
    # 滑动平均
    window = min(20, len(ades)//5)
    if window > 1:
        ade_smooth = np.convolve(ades, np.ones(window)/window, mode='valid')
        fde_smooth = np.convolve(fdes, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(ades)), ade_smooth, color='#2980b9',
                linewidth=2, label=f'ADE (滑动均值)')
        ax.plot(range(window-1, len(fdes)), fde_smooth, color='#c0392b',
                linewidth=2, label=f'FDE (滑动均值)')
    ax.set_xlabel('帧序号')
    ax.set_ylabel('误差 (m)')
    ax.set_title('误差随时间变化')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] 指标分布图保存到: {save_path}")


def plot_frame_grid(data, ades, save_path, n_cols=4, n_rows=3, mode='worst'):
    """画最好/最差/随机的若干帧"""
    n_frames = n_cols * n_rows

    if mode == 'worst':
        indices = np.argsort(ades)[-n_frames:][::-1]
        title_prefix = "最差帧"
    elif mode == 'best':
        indices = np.argsort(ades)[:n_frames]
        title_prefix = "最优帧"
    else:
        indices = np.random.choice(len(data), n_frames, replace=False)
        title_prefix = "随机帧"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    fig.suptitle(f'SimLingo 开环评估 — {title_prefix} (共{n_frames}帧)',
                 fontsize=13, fontweight='bold')

    for idx, (ax, frame_idx) in enumerate(zip(axes.flat, indices)):
        plot_single_frame(data[frame_idx], ax,
                          title=f"帧 #{frame_idx}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] {title_prefix}对比图保存到: {save_path}")


def plot_summary(ades, fdes, save_path):
    """汇总统计面板"""
    fig = plt.figure(figsize=(10, 4))
    fig.patch.set_facecolor('#f8f9fa')

    stats = {
        'ADE 均值':    f'{np.mean(ades):.4f} m',
        'ADE 中位数':  f'{np.median(ades):.4f} m',
        'ADE 标准差':  f'{np.std(ades):.4f} m',
        'ADE P90':     f'{np.percentile(ades, 90):.4f} m',
        'FDE 均值':    f'{np.mean(fdes):.4f} m',
        'FDE 中位数':  f'{np.median(fdes):.4f} m',
        'FDE 标准差':  f'{np.std(fdes):.4f} m',
        'FDE P90':     f'{np.percentile(fdes, 90):.4f} m',
        '评估帧数':    str(len(ades)),
    }

    ax = fig.add_subplot(111)
    ax.axis('off')

    col1 = [(k, v) for k, v in stats.items() if 'ADE' in k]
    col2 = [(k, v) for k, v in stats.items() if 'FDE' in k]
    col3 = [(k, v) for k, v in stats.items() if 'ADE' not in k and 'FDE' not in k]

    y = 0.9
    for items, x, color in [(col1, 0.1, '#3498db'), (col2, 0.45, '#e74c3c'),
                              (col3, 0.78, '#2ecc71')]:
        for k, v in items:
            ax.text(x, y, k, transform=ax.transAxes, fontsize=11,
                    color='#555', ha='left')
            ax.text(x+0.18, y, v, transform=ax.transAxes, fontsize=11,
                    color=color, fontweight='bold', ha='left')
            y -= 0.18
        y = 0.9

    ax.set_title('SimLingo 开环评估汇总', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] 汇总统计图保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='simlingo_training/outputs/OpenGVLab/InternVL2-1B/predictions/per_frame_waypoints_rank_0.json',
                        help='per_frame_waypoints_rank_0.json 路径')
    parser.add_argument('--output', default='eval_results/visualization',
                        help='输出目录')
    parser.add_argument('--n_sample_frames', type=int, default=12,
                        help='展示的样本帧数')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载数据
    data = load_data(args.data)
    ades, fdes = compute_metrics(data)

    print(f"\n{'='*50}")
    print(f"  ADE 均值:   {np.mean(ades):.4f} m")
    print(f"  ADE 中位数: {np.median(ades):.4f} m")
    print(f"  FDE 均值:   {np.mean(fdes):.4f} m")
    print(f"  FDE 中位数: {np.median(fdes):.4f} m")
    print(f"  评估帧数:   {len(ades)}")
    print(f"{'='*50}\n")

    # 1. 汇总统计
    plot_summary(ades, fdes,
                 os.path.join(args.output, '01_summary.png'))

    # 2. 指标分布
    plot_metrics_distribution(ades, fdes,
                              os.path.join(args.output, '02_metrics_distribution.png'))

    # 3. 最差帧
    n_cols, n_rows = 4, 3
    plot_frame_grid(data, ades,
                    os.path.join(args.output, '03_worst_frames.png'),
                    n_cols=n_cols, n_rows=n_rows, mode='worst')

    # 4. 最优帧
    plot_frame_grid(data, ades,
                    os.path.join(args.output, '04_best_frames.png'),
                    n_cols=n_cols, n_rows=n_rows, mode='best')

    # 5. 随机帧
    plot_frame_grid(data, ades,
                    os.path.join(args.output, '05_random_frames.png'),
                    n_cols=n_cols, n_rows=n_rows, mode='random')

    print(f"\n[✓] 所有可视化图保存到: {args.output}/")
    print(f"    01_summary.png          — 汇总统计")
    print(f"    02_metrics_distribution.png — ADE/FDE 分布")
    print(f"    03_worst_frames.png     — 预测最差的12帧")
    print(f"    04_best_frames.png      — 预测最好的12帧")
    print(f"    05_random_frames.png    — 随机12帧")


if __name__ == '__main__':
    main()

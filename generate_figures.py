#!/usr/bin/env python3
"""生成论文所需的全部6张图片到 images/ 目录"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Polygon
import numpy as np
import os

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
os.makedirs(out_dir, exist_ok=True)

# 全局中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════
# 图4-1: 算法流程图
# ═══════════════════════════════════════════════════════════════
def draw_fig_4_1():
    fig, ax = plt.subplots(1, 1, figsize=(8, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    ax.set_title('图4-1 改进分支界限算法流程图', fontsize=12, fontweight='bold', pad=15)

    box_w, box_h = 3.0, 1.0
    diamond_w, diamond_h = 2.8, 1.5
    small_w, small_h = 2.5, 0.8
    center_x = 5.0

    def add_box(x, y, w, h, text, fill_color='#E8F0FE', edge_color='#1a3a5c', fontsize=8):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1", facecolor=fill_color, edgecolor=edge_color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, wrap=True)

    def add_diamond(x, y, w, h, text, fontsize=8):
        diamond_verts = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
        diamond = Polygon(diamond_verts, facecolor='#FFF8E1', edgecolor='#1a3a5c', linewidth=1.5)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)

    def add_arrow(start, end, style='->'):
        ax.annotate('', xy=end, xytext=start,
            arrowprops=dict(arrowstyle=style, color='#1a3a5c', lw=1.5, connectionstyle='arc3,rad=0'))

    # Title box
    add_box(center_x, 19.2, box_w, box_h, '开始\n(物品已按v/w降序排序)', fontsize=9)

    # Step 1: Greedy
    add_box(center_x, 17.7, box_w, box_h, '贪心法构造初始可行解\n作为全局下界 best_val', fontsize=9)
    add_arrow((center_x, 18.7), (center_x, 18.2))

    # Step 2: Call backtrack
    add_box(center_x, 16.2, box_w, box_h, '调用递归函数\nBacktrack(idx=0, cur_w=0, cur_v=0)', fontsize=9)
    add_arrow((center_x, 17.2), (center_x, 16.7))

    # Node counter
    add_box(center_x, 14.8, box_w, 0.7, 'node_count++', fontsize=9, fill_color='#f0f0f0')
    add_arrow((center_x, 15.7), (center_x, 15.15))

    # Diamond 1: idx == n?
    add_diamond(center_x, 13.3, diamond_w, diamond_h, 'idx == n\n(叶节点)?', fontsize=9)
    add_arrow((center_x, 14.45), (center_x, 14.05))

    # Left branch from diamond: Yes -> update best_val
    add_box(center_x + 3, 13.3, small_w, small_h, '若cur_v > best_val\n更新best_val', fontsize=8, fill_color='#E8F5E9')
    ax.annotate('', xy=(center_x + 1.75, 13.3), xytext=(center_x + 1.4, 13.3),
        arrowprops=dict(arrowstyle='->', color='#1a3a5c', lw=1.5))
    ax.text(center_x + 2.0, 13.8, '是', fontsize=8, ha='center', va='center')
    # return arrow
    ax.annotate('', xy=(center_x + 3.0, 12.6), xytext=(center_x + 3.0, 12.9),
        arrowprops=dict(arrowstyle='->', color='#1a3a5c', lw=1.5))
    ax.text(center_x + 3.0, 12.55, '返回', fontsize=7, ha='center')

    # Down from diamond: No -> compute bound
    add_arrow((center_x, 12.55), (center_x, 12.0))
    ax.text(center_x + 0.3, 12.25, '否', fontsize=8, ha='center', va='center')

    # Compute bound
    add_box(center_x, 11.2, box_w, 0.9, '计算上界:\nub = cur_v + Bound(idx, C - cur_w)', fontsize=9, fill_color='#FFF3E0')
    add_arrow((center_x, 11.65), (center_x, 11.55))

    # Diamond 2: ub <= best_val?
    add_diamond(center_x, 9.7, diamond_w, diamond_h, 'ub ≤ best_val\n(可剪枝)?', fontsize=9)
    add_arrow((center_x, 10.75), (center_x, 10.45))

    # Left branch from diamond 2: Yes -> prune
    add_box(center_x + 3.2, 9.7, 2.2, 0.8, '剪枝返回\n(不再搜索该子树)', fontsize=8, fill_color='#FFCDD2')
    ax.annotate('', xy=(center_x + 2.1, 9.7), xytext=(center_x + 1.4, 9.7),
        arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.8))
    ax.text(center_x + 2.45, 10.2, '是', fontsize=8, ha='center', color='#c62828', fontweight='bold')

    # Down from diamond 2: No -> branch
    add_arrow((center_x, 8.95), (center_x, 8.3))
    ax.text(center_x + 0.3, 8.6, '否', fontsize=8, ha='center')

    # Branch point
    add_box(center_x, 7.6, box_w, 0.8, '探索两条分支', fontsize=9, fill_color='#E8F5E9')

    # Two branches
    # Left: select item
    add_box(center_x - 2.2, 6.1, 2.5, 1.2,
            '分支1: 选择当前物品\nif cur_w + w ≤ C\nBacktrack(idx+1, cur_w+w, cur_v+v)',
            fontsize=7.5, fill_color='#e3f2fd')
    add_arrow((center_x - 0.5, 7.2), (center_x - 1.5, 6.7))

    # Right: skip item
    add_box(center_x + 2.2, 6.1, 2.5, 1.2,
            '分支2: 不选当前物品\nBacktrack(idx+1,\ncur_w, cur_v)',
            fontsize=7.5, fill_color='#f3e5f5')
    add_arrow((center_x + 0.5, 7.2), (center_x + 1.5, 6.7))

    # End
    add_box(center_x, 3.8, 2.5, 0.8, '搜索结束，返回 best_val', fontsize=9, fill_color='#C8E6C9')

    # Arrows from branches back to top (recursion)
    ax.annotate('', xy=(center_x - 2.0, 14.5), xytext=(center_x - 2.0, 5.5),
        arrowprops=dict(arrowstyle='->', color='#666666', lw=1, linestyle='dashed',
        connectionstyle='arc3,rad=-0.6'))
    ax.text(center_x - 4.5, 9.5, '递归\n返回', fontsize=7, color='#666666', ha='center')
    ax.annotate('', xy=(center_x + 2.0, 14.5), xytext=(center_x + 2.0, 5.5),
        arrowprops=dict(arrowstyle='->', color='#666666', lw=1, linestyle='dashed',
        connectionstyle='arc3,rad=0.6'))

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_4_1_flowchart.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('图4-1 已生成')

draw_fig_4_1()


# ═══════════════════════════════════════════════════════════════
# 图4-2: 搜索树剪枝图示
# ═══════════════════════════════════════════════════════════════
def draw_fig_4_2():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('图4-2 小规模实例搜索树剪枝示意图 (C=10)', fontsize=13, fontweight='bold', pad=15)

    # Tree structure: nodes as boxes with (w, v, ub)
    # Position: (x, y), label
    node_w, node_h = 2.2, 1.2

    def draw_node(x, y, text, fill='#E8F0FE', edge='#1a3a5c', fontsize=7.5):
        rect = FancyBboxPatch((x - node_w/2, y - node_h/2), node_w, node_h,
            boxstyle="round,pad=0.1", facecolor=fill, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)

    def draw_arrow(x1, y1, x2, y2, color='#1a3a5c', lw=1.2, style='->'):
        ax.annotate('', xy=(x2, y2 + node_h/2 + 0.1), xytext=(x1, y1 - node_h/2 - 0.1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, connectionstyle='arc3,rad=0'))

    # Root: (0, 0, ub=18.67)
    draw_node(10, 9, 'root\n(0, 0, ub=18.67)', fill='#BBDEFB')
    # Decision: x1=1 or x1=0
    ax.text(6.5, 8.3, '选x1=1', fontsize=8, ha='center')
    draw_arrow(10, 9, 6, 7.2)

    ax.text(12.2, 8.3, '不选x1=0', fontsize=8, ha='center')
    # Pruned branch - draw red X
    draw_arrow(10, 9, 14, 7.2, color='#c62828', lw=2)
    # Prune node
    draw_node(14, 7.2, '剪枝!\n(5, ub=15)', fill='#FFCDD2', edge='#c62828', fontsize=8)
    # Red cross
    ax.plot([13.2, 14.8], [6.9, 7.5], color='#c62828', lw=2.5)
    ax.plot([13.2, 14.8], [7.5, 6.9], color='#c62828', lw=2.5)

    # x1=1 branch: (5, 10, ub=18.67)
    draw_node(6, 7.2, '(5, 10, ub=18.67)', fill='#E3F2FD')
    # Decision x2=1 or x2=0
    ax.text(4.3, 6.5, '选x2=1', fontsize=8, ha='center')
    draw_arrow(6, 7.2, 3.5, 5.5)
    draw_node(3.5, 5.5, '(9, 17, ub=18.67)', fill='#E3F2FD')
    # x2=1, x3=0 -> leaf (can't add x3)
    draw_arrow(3.5, 5.5, 3.5, 4.0)
    draw_node(3.5, 4.0, '叶(9,17)\nv=17不更新', fill='#f5f5f5', fontsize=7.5)

    ax.text(6.8, 6.5, '不选x2=0', fontsize=8, ha='center')
    draw_arrow(6, 7.2, 8.5, 5.5)
    draw_node(8.5, 5.5, '(5, 10, ub=18.67)', fill='#E3F2FD')
    # x2=0, x3=1
    ax.text(7.5, 4.8, '选x3=1', fontsize=8, ha='center')
    draw_arrow(8.5, 5.5, 7, 4.2)
    draw_node(7, 4.2, '(8, 15, ub=18)', fill='#E8F5E9')
    # x3=1, x4=1 -> best!
    ax.text(6.0, 3.5, '选x4=1', fontsize=8, ha='center')
    draw_arrow(7, 4.2, 5.5, 3.0)
    draw_node(5.5, 3.0, '★(10,18)\nbest=18!', fill='#C8E6C9', edge='#2E7D32', fontsize=8)

    # x2=0, x3=0, x4=1
    ax.text(8.5, 4.8, '不选x3=0', fontsize=8, ha='center')
    draw_arrow(8.5, 5.5, 10, 4.2)
    draw_node(10, 4.2, '(5, 10, ub=18)', fill='#E8F5E9')
    ax.text(10.4, 3.5, '选x4=1', fontsize=8, ha='center')
    draw_arrow(10, 4.2, 11, 3.0)
    draw_node(11, 3.0, '(7, 13, ub=18)', fill='#E8F5E9')
    # continue to x3=1 (after x4)
    draw_arrow(11, 3.0, 11.5, 2.0)
    draw_node(11.5, 2.0, '叶\n...', fill='#f5f5f5', fontsize=7)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='#1a3a5c', label='待搜索节点'),
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='最优解节点'),
        mpatches.Patch(facecolor='#FFCDD2', edgecolor='#c62828', label='剪枝节点'),
        mpatches.Patch(facecolor='#f5f5f5', edgecolor='#888888', label='叶节点'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_4_2_search_tree.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('图4-2 已生成')

draw_fig_4_2()


# ═══════════════════════════════════════════════════════════════
# 图6-1: 不同n下运行时间对比折线图
# ═══════════════════════════════════════════════════════════════
def draw_fig_6_1():
    n_vals = [20, 22, 25, 28, 30]
    dp_time = [812, 1423, 2321, 4102, 5690]
    bb_time = [104, 189, 211, 358, 437]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(n_vals, dp_time, 'o-', color='#d32f2f', linewidth=2, markersize=8, label='DP (动态规划)')
    ax.plot(n_vals, bb_time, 's-', color='#1976d2', linewidth=2, markersize=8, label='BB (分支界限)')
    ax.set_xlabel('任务数 n', fontsize=12)
    ax.set_ylabel('运行时间 / µs', fontsize=12)
    ax.set_title('图6-1 不同任务数下运行时间对比 (α=0.6)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate speedup
    for i, (n, dp, bb) in enumerate(zip(n_vals, dp_time, bb_time)):
        speedup = dp / bb
        ax.annotate(f'{speedup:.1f}×', (n, bb), textcoords="offset points", xytext=(0, 15),
            ha='center', fontsize=8, color='#1976d2')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_6_1_runtime_vs_n.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('图6-1 已生成')

draw_fig_6_1()


# ═══════════════════════════════════════════════════════════════
# 图6-2: 搜索状态数对比柱状图
# ═══════════════════════════════════════════════════════════════
def draw_fig_6_2():
    n_vals = [20, 22, 25, 28, 30]
    dp_states_m = [2.35, 4.18, 6.87, 12.54, 18.94]
    bb_nodes_k = [11.872, 25.434, 38.156, 67.439, 92.820]

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5.5))

    x = np.arange(len(n_vals))
    width = 0.35

    bars1 = ax1.bar(x - width/2, dp_states_m, width, label='DP 状态数 (×10⁶)', color='#d32f2f', alpha=0.85)
    ax1.set_ylabel('DP 状态数 (×10⁶)', fontsize=12, color='#d32f2f')
    ax1.tick_params(axis='y', labelcolor='#d32f2f')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, bb_nodes_k, width, label='BB 节点数 (×10³)', color='#1976d2', alpha=0.85)
    ax2.set_ylabel('BB 节点数 (×10³)', fontsize=12, color='#1976d2')
    ax2.tick_params(axis='y', labelcolor='#1976d2')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'n={n}' for n in n_vals], fontsize=10)
    ax1.set_title('图6-2 搜索状态数对比', fontsize=13, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # Add ratio annotation
    for i in range(len(n_vals)):
        ratio = dp_states_m[i] * 1000 / bb_nodes_k[i]
        ax1.annotate(f'DP/BB≈{ratio:.0f}×', (x[i], dp_states_m[i]), textcoords="offset points",
            xytext=(0, 8), ha='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_6_2_state_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('图6-2 已生成')

draw_fig_6_2()


# ═══════════════════════════════════════════════════════════════
# 图6-3: 不同α下运行时间折线图
# ═══════════════════════════════════════════════════════════════
def draw_fig_6_3():
    alpha_vals = [0.2, 0.4, 0.6, 0.8]
    dp_time = [1021, 1638, 2321, 2896]
    bb_time = [158, 193, 211, 224]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(alpha_vals, dp_time, 'o-', color='#d32f2f', linewidth=2, markersize=8, label='DP (动态规划)')
    ax.plot(alpha_vals, bb_time, 's-', color='#1976d2', linewidth=2, markersize=8, label='BB (分支界限)')
    ax.set_xlabel('容量比例 α', fontsize=12)
    ax.set_ylabel('运行时间 / µs', fontsize=12)
    ax.set_title('图6-3 不同容量比例下运行时间对比 (n=25)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate speedup
    for i, (a, dp, bb) in enumerate(zip(alpha_vals, dp_time, bb_time)):
        speedup = dp / bb
        ax.annotate(f'{speedup:.1f}×', (a, bb), textcoords="offset points", xytext=(0, 15),
            ha='center', fontsize=8, color='#1976d2')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_6_3_runtime_vs_alpha.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('图6-3 已生成')

draw_fig_6_3()


# ═══════════════════════════════════════════════════════════════
# 图6-4: 空间占用对比图
# ═══════════════════════════════════════════════════════════════
def draw_fig_6_4():
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    methods = ['动态规划 (DP)\nn=25, C=895', '分支界限 (BB)\nn=25']
    memory_bytes = [895 * 4, 200]  # DP: int array of size C; BB: ~200 bytes

    bars = ax.bar(methods, memory_bytes, color=['#d32f2f', '#1976d2'], alpha=0.85, width=0.5)
    ax.set_ylabel('内存占用 / Bytes', fontsize=12)
    ax.set_title('图6-4 空间占用对比 (n=25, C=895)', fontsize=13, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, memory_bytes):
        if val >= 1000:
            label = f'{val/1024:.2f} KB'
        else:
            label = f'{val} B'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, label,
            ha='center', fontsize=11, fontweight='bold')

    # Add ratio annotation
    ratio = memory_bytes[0] / memory_bytes[1]
    ax.annotate(f'DP内存占用是BB的 {ratio:.0f} 倍', xy=(0.5, 0.9),
        xycoords='axes fraction', ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_6_4_space_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('图6-4 已生成')

draw_fig_6_4()

print(f'\n全部6张图片已生成至: {out_dir}/')
for f in sorted(os.listdir(out_dir)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f'  {f} ({size_kb:.1f} KB)')

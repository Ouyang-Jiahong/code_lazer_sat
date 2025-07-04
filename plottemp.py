import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

def split_arc_window(total_time, min_obs_time, overlap_ratio=0.5):
    """
    根据最小观测时间和重叠比例从总时间的中点开始切割弧段。
    """
    step = int(min_obs_time * (1 - overlap_ratio))  # 步长，即相邻子弧段之间的间隔
    windows = []

    # 第一个子弧段的中点与总时间的中点对齐
    mid_point = total_time // 2
    start = mid_point - min_obs_time // 2
    end = mid_point + min_obs_time // 2
    windows.append((start, end))

    # 从中点向左拓展
    pos = mid_point
    while pos - min_obs_time >= 0:
        start = pos - min_obs_time
        end = pos
        windows.insert(0, (start, end))  # 插入到列表开头，确保顺序
        pos -= step

    # 重设pos为中点，向右拓展
    pos = mid_point
    while pos + min_obs_time <= total_time:
        start = pos
        end = pos + min_obs_time
        windows.append((start, end))
        pos += step

    return sorted(windows)  # 按时间排序

def visualize_split(total_time, min_obs_time, overlap_ratio=0.5):
    """
    可视化弧段切割过程，增强直观性
    """
    # 切割弧段
    windows = split_arc_window(total_time, min_obs_time, overlap_ratio)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 绘制总弧段（蓝色背景）
    plt.fill_between([0, total_time], 0, 2, color='lightblue', alpha=0.3, label='总弧段区间')
    plt.plot([0, total_time], [1, 1], color='blue', lw=6, label='总弧段')
    
    # 绘制切割后的弧段（红色）
    for i, (start, end) in enumerate(windows):
        # 绘制每个子弧段的实际位置（红色）
        plt.plot([start, end], [i + 2, i + 2], color='red', lw=4, label=f'子弧段 {i+1}' if i == 0 else "")
        
        # 在每个子弧段上添加标注（起始时间和结束时间）
        plt.text(start, i + 2.1, f'{start}', fontsize=9, ha='center', color='black')
        plt.text(end, i + 2.1, f'{end}', fontsize=9, ha='center', color='black')

        # 显示重叠区域
        if i > 0:
            overlap_start = windows[i-1][1]  # 前一个子弧段的结束时间
            overlap_end = start  # 当前子弧段的开始时间
            if overlap_start < overlap_end:
                plt.fill_between([overlap_start, overlap_end], i + 1.5, i + 2.5, color='green', alpha=0.5, label=f'重叠区间 {i}' if i == 1 else "")
    
    # 设置图形参数
    plt.xlabel('时间单位')
    plt.yticks([1] + list(range(2, len(windows) + 2)), ['总弧段'] + [f'子弧段 {i+1}' for i in range(len(windows))])
    plt.title('弧段切割示意图（带重叠区间）')
    plt.legend(loc='upper left')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 添加更多辅助线
    for i in range(len(windows)):
        plt.axvline(x=windows[i][0], color='gray', linestyle='--', lw=1)  # 起始时间的垂直辅助线
        plt.axvline(x=windows[i][1], color='gray', linestyle='--', lw=1)  # 结束时间的垂直辅助线

    # 为每个弧段添加边框
    for i, (start, end) in enumerate(windows):
        plt.plot([start, start], [i + 1.5, i + 2.5], color='black', lw=1)  # 左边框
        plt.plot([end, end], [i + 1.5, i + 2.5], color='black', lw=1)  # 右边框
        plt.plot([start, end], [i + 1.5, i + 1.5], color='black', lw=1)  # 上边框
        plt.plot([start, end], [i + 2.5, i + 2.5], color='black', lw=1)  # 下边框

    plt.show()

# 参数设定
total_time = 100        # 总弧段长度（时间单位）
min_obs_time = 30       # 最小观测时间
overlap_ratio = 0.5     # 重叠比例

# 可视化切割过程
visualize_split(total_time, min_obs_time, overlap_ratio)

import numpy as np
import pandas as pd
from astropy.time import Time
import scipy.io as sio

# ----------------------------
# 数据加载阶段
# ----------------------------
sensor_data = pd.read_excel("simData/sensorData.xlsx")  # 测站参数表
# require_data = pd.read_excel("simData/requireData.xlsx")  # 任务需求表
# require_data = pd.read_excel("simData/requireData5000.xlsx")  # 任务需求表
require_data = pd.read_excel("simData/requireData10000.xlsx")  # 任务需求表
# usable_arcs = sio.loadmat("simData/usableArcs.mat")["usableArcs"]  # 可见弧段数据
# usable_arcs = np.array(sio.loadmat("simData/AllArcChain_A.mat")["AllArcChain_A"])  # 可见弧段数据
usable_arcs = np.array(sio.loadmat("simData/AllArcChain.mat")["AllArcChain"])  # 可见弧段数据
simDate = sio.loadmat("simData/simDate.mat")["simDate"]  # 仿真时间节点（UTC）

# ----------------------------
# 基础参数提取
# ----------------------------
num_radars = len(sensor_data)  # 总测站数
num_targets = len(require_data)  # 总目标数

radar_capacities = sensor_data["最大探测目标数"].values  # 各测站最大可观测目标数

required_stations = require_data["需要的测站数量"].values  # 各目标所需最小观测测站数
required_observation_time = require_data["需要的观测时间(min)"].values  # 各目标所需最小观测时长（单位：分钟）
required_arc_count = require_data["需要的弧段数量"].values  # 各目标所需最小有效观测次数
priority_weights = require_data["优先级(数值越大，优先级越高)"].values  # 各目标优先级权重

start_time = Time("2021-10-14T04:00:00", format='isot', scale='utc')  # 仿真开始时间

def split_arc_windows_exact(radar_target_vis_dict, simDate, required_observation_time, sat_id_to_index, overlap_ratio=0.5):
    """
    用整数索引滑窗重叠切割弧段，每个子弧段持续时间等于最小观测时间。
    overlap_ratio: 相邻子弧段的重叠比例（0~1），如0.5表示滑动窗口步长为min_obs_time的一半。
    """
    new_dict = {}
    sim_times = simDate[0]
    for (radar_id, sat_id), windows in radar_target_vis_dict.items():
        s_idx = sat_id_to_index[sat_id]
        min_obs_time = int(round(required_observation_time[s_idx]))
        step = max(1, int(round(min_obs_time * (1 - overlap_ratio))))
        new_windows = []
        for (start_idx, end_idx, duration) in windows:
            if end_idx <= start_idx:
                continue
            t0 = int(round(sim_times[start_idx]))
            t1 = int(round(sim_times[end_idx]))
            if t1 - t0 < min_obs_time:
                new_windows.append((start_idx, end_idx, int(round(duration))))
                continue
            pos = t0
            while pos + min_obs_time <= t1:
                seg_start = pos
                seg_end = seg_start + min_obs_time
                # 找到最近的索引
                seg_start_idx = int(np.searchsorted(sim_times, seg_start, side='left'))
                seg_end_idx = int(np.searchsorted(sim_times, seg_end, side='right')) - 1
                if seg_end_idx > end_idx:
                    seg_end_idx = end_idx
                if seg_start_idx < start_idx:
                    seg_start_idx = start_idx
                seg_duration = int(round(sim_times[seg_end_idx] - sim_times[seg_start_idx]))
                # 只保留等于最小观测时间的子弧段
                if seg_duration == min_obs_time:
                    new_windows.append((seg_start_idx, seg_end_idx, seg_duration))
                pos += step
        if not new_windows:
            # 保底：保留原弧段
            for (start_idx, end_idx, duration) in windows:
                new_windows.append((start_idx, end_idx, int(round(duration))))
        new_dict[(radar_id, sat_id)] = new_windows
    return new_dict

# ----------------------------
# 构建雷达-目标可见性字典
# 键为 (radar_id, sat_id)，值为该组合下所有可见时间窗口及对应时长（分钟）
# ----------------------------
radar_target_vis_dict = {}

for i in range(len(usable_arcs[0])):
    sat_id = usable_arcs[0][i][0][0][0]  # 目标编号
    radar_id = usable_arcs[0][i][1][0][0]  # 测站编号
    arc_chain = usable_arcs[0][i][2]  # 所有可见时间段（起止索引）
    arc_durations = usable_arcs[0][i][3] / 60 # 对应时间段的观测时长（单位：分钟）

    visible_windows = []

    for j in range(arc_chain.shape[0]):
        s_idx = arc_chain[j, 0] - 1  # 起始时间索引（MATLAB从1开始，转换为Python索引）
        e_idx = arc_chain[j, 1] - 1  # 结束时间索引

        # 存储时间窗口及其持续时间
        visible_windows.append((s_idx, e_idx, arc_durations[j, 0]))

    # 写入全局可见性字典
    # radar_target_vis_dict[(101,63399)][i][0]代表取第101号测站与63399号卫星之间的第i个可用弧段的起始时间点
    # radar_target_vis_dict[(101,63399)][i][1]代表取第101号测站与63399号卫星之间的第i个可用弧段的终止时间点
    # radar_target_vis_dict[(101,63399)][i][2]代表取第101号测站与63399号卫星之间的第i个可用弧段的持续时长
    # 也可以这样理解：
    # 获取第101号测站与63399号卫星之间的第一个可用弧段：
    # window = radar_target_vis_dict[(101, 63399)][0]
    # start_time_idx = window[0]   # 起始时间索引
    # end_time_idx = window[1]     # 结束时间索引
    # duration = window[2]         # 弧段持续时间（分钟）
    radar_target_vis_dict[(radar_id, sat_id)] = visible_windows

# 构建目标编号到索引的映射
sat_id_to_index = {}
for idx, sat_id in enumerate(require_data["目标编号"]):
    sat_id_to_index[sat_id] = idx

print("[data.py] 数据预处理完成，radar_target_vis_dict 已构建。")

# ----------- 新增：以中心优先切割弧段 -----------
radar_target_vis_dict = split_arc_windows_exact(
    radar_target_vis_dict, simDate, required_observation_time, sat_id_to_index, overlap_ratio=0.5
)
print(f"[data.py] 已按中心优先、每目标最小观测时间切割弧段。")
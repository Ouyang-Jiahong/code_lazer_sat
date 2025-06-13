import pandas as pd
from astropy.time import Time
import scipy.io as sio

# ----------------------------
# 数据加载阶段
# ----------------------------
sensor_data = pd.read_excel("simData/sensorData.xlsx")  # 测站参数表
require_data = pd.read_excel("simData/requireData.xlsx")  # 任务需求表
usable_arcs = sio.loadmat("simData/usableArcs.mat")["usableArcs"]  # 可见弧段数据
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

# ----------------------------
# 构建雷达-目标可见性字典
# 键为 (radar_id, sat_id)，值为该组合下所有可见时间窗口及对应时长（分钟）
# ----------------------------
radar_target_vis_dict = {}

for i in range(len(usable_arcs[0])):
    sat_id = usable_arcs[0][i][0][0][0]  # 目标编号
    radar_id = usable_arcs[0][i][1][0][0]  # 测站编号
    arc_chain = usable_arcs[0][i][2]  # 所有可见时间段（起止索引）
    arc_durations = usable_arcs[0][i][3]  # 对应时间段的观测时长（单位：分钟）

    visible_windows = []

    for j in range(arc_chain.shape[0]):
        s_idx = arc_chain[j, 0] - 1  # 起始时间索引（MATLAB从1开始，转换为Python索引）
        e_idx = arc_chain[j, 1] - 1  # 结束时间索引

        # 构造UTC时间格式的起止时间
        s_time = Time(f"{int(simDate[0, s_idx])}-{int(simDate[1, s_idx]):02d}-{int(simDate[2, s_idx]):02d}T"
                      f"{int(simDate[3, s_idx]):02d}:{int(simDate[4, s_idx]):02d}:{int(simDate[5, s_idx]):02d}",
                      format='isot', scale='utc')

        e_time = Time(f"{int(simDate[0, e_idx])}-{int(simDate[1, e_idx]):02d}-{int(simDate[2, e_idx]):02d}T"
                      f"{int(simDate[3, e_idx]):02d}:{int(simDate[4, e_idx]):02d}:{int(simDate[5, e_idx]):02d}",
                      format='isot', scale='utc')

        # 存储时间窗口及其持续时间
        visible_windows.append((s_time, e_time, arc_durations[j, 0]))

    # 写入全局可见性字典
    radar_target_vis_dict[(radar_id, sat_id)] = visible_windows

print("[data.py] 数据预处理完成，radar_target_vis_dict 已构建。")
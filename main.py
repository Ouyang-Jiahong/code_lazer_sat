import pandas as pd
import pulp
from astropy.time import Time
import scipy.io as sio
from functions import *

# 加载仿真数据
sensor_data = pd.read_excel("simData/sensorData.xlsx")  # 测站信息
require_data = pd.read_excel("simData/requireData.xlsx")  # 目标与任务需求信息
usable_arcs = sio.loadmat("simData/usableArcs.mat")["usableArcs"]  # 可用观测弧段信息
simDate = sio.loadmat("simData/simDate.mat")["simDate"]  # 仿真时间节点（UTC）

# 初始化基础参数
num_radars = len(sensor_data)  # 测站总数
num_targets = len(require_data)  # 目标总数

# 提取测站能力参数
radar_capacities = sensor_data["最大探测目标数"].values  # 各测站最大可同时观测目标数

# 提取任务需求参数
required_stations = require_data["需要的测站数量"].values  # 各目标所需的最小观测测站数
required_observation_time = require_data["需要的观测时间(min)"].values  # 有效观测所需的最小累计观测时长
required_arc_count = require_data["需要的弧段数量"].values  # 判断任务完成所需的最小有效观测次数
priority_weights = require_data["优先级(数值越大，优先级越高)"].values  # 各目标任务的优先级权重

start_time = Time("2021-10-14T04:00:00", format='isot', scale='utc')

# 构建观测可见性字典：键为(测站编号, 目标编号)，值为该组合下所有可见时间窗口及其时长
radar_target_vis_dict = {}
for i in range(len(usable_arcs[0])):
    sat_id = usable_arcs[0][i][0][0][0]  # 目标编号
    radar_id = usable_arcs[0][i][1][0][0]  # 测站编号
    arc_chain = usable_arcs[0][i][2]  # 当前测站-目标组合的所有可见时间段（起止索引）
    arc_durations = usable_arcs[0][i][3]  # 对应时间段的观测时长（单位：分钟）

    visible_windows = []  # 用于存储当前组合的所有可见窗口

    # 遍历该测站-目标组合的所有可见时间段
    for j in range(arc_chain.shape[0]):
        s_idx = arc_chain[j, 0] - 1  # 起始时间索引（减1是因为 MATLAB 索引从1开始）
        e_idx = arc_chain[j, 1] - 1  # 终止时间索引

        # 构造起始时间（UTC格式）
        s_time = Time(f"{int(simDate[0, s_idx])}-{int(simDate[1, s_idx]):02d}-{int(simDate[2, s_idx]):02d}T"
                      f"{int(simDate[3, s_idx]):02d}:{int(simDate[4, s_idx]):02d}:{int(simDate[5, s_idx]):02d}",
                      format='isot', scale='utc')

        # 构造终止时间（UTC格式）
        e_time = Time(f"{int(simDate[0, e_idx])}-{int(simDate[1, e_idx]):02d}-{int(simDate[2, e_idx]):02d}T"
                      f"{int(simDate[3, e_idx]):02d}:{int(simDate[4, e_idx]):02d}:{int(simDate[5, e_idx]):02d}",
                      format='isot', scale='utc')

        # 将时间窗口及对应时长加入列表
        visible_windows.append((s_time, e_time, arc_durations[j, 0]))

    # 存入可见性字典
    radar_target_vis_dict[(radar_id, sat_id)] = visible_windows

print("加载数据完成！")

## 构建模型
print("模型构建中...")
prob = pulp.LpProblem("Observation_Planning", pulp.LpMaximize)

# 决策变量
x = [[pulp.LpVariable(f"x_{r}_{t}", cat="Binary") for t in range(num_targets)] for r in range(num_radars)]
y = [[
    [pulp.LpVariable(f"y_{r}_{t}_{a}", cat="Binary") for a in range(len(radar_target_vis_dict.get((r, t), [])))]
    for t in range(num_targets)] for r in range(num_radars)]

# 松弛变量（软约束处理）
slack_arc = [pulp.LpVariable(f"slack_arc_{t}", lowBound=0) for t in range(num_targets)]
slack_time = [pulp.LpVariable(f"slack_time_{t}", lowBound=0) for t in range(num_targets)]

# 目标函数：最大化优先级 - 惩罚项
penalty_time = 100
penalty_arc = 50
prob += (
    pulp.lpSum(priority_weights[t] * x[r][t] for r in range(num_radars) for t in range(num_targets))
    - pulp.lpSum([penalty_time * slack_time[t] + penalty_arc * slack_arc[t] for t in range(num_targets)])
)

# 约束1：目标的测站数要求
for t in range(num_targets):
    prob += pulp.lpSum([x[r][t] for r in range(num_radars)]) >= required_stations[t]

# 约束2：观测时间 + 松弛
for t in range(num_targets):
    total_time = []
    for r in range(num_radars):
        visibles = radar_target_vis_dict.get((r, t), [])
        total_time.append(pulp.lpSum([y[r][t][a] * visibles[a][2] / 60 for a in range(len(visibles))]))
    prob += pulp.lpSum(total_time) + slack_time[t] >= required_observation_time[t]

# 约束3：弧段数量 + 松弛
for t in range(num_targets):
    total_arcs = []
    for r in range(num_radars):
        total_arcs.append(pulp.lpSum([y[r][t][a] for a in range(len(y[r][t]))]))
    prob += pulp.lpSum(total_arcs) + slack_arc[t] >= required_arc_count[t]

# 约束4：雷达容量
for r in range(num_radars):
    prob += pulp.lpSum([x[r][t] for t in range(num_targets)]) <= radar_capacities[r]

# 约束5：只有分配了雷达才能启用弧段
for r in range(num_radars):
    for t in range(num_targets):
        for a in range(len(y[r][t])):
            prob += y[r][t][a] <= x[r][t]

# 求解
status = prob.solve(pulp.PULP_CBC_CMD(msg=True))
print(f"求解状态: {pulp.LpStatus[status]}")
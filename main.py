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

# 遍历所有可用观测弧段数据
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

# 构建模型
print("模型构建中...")
prob = pulp.LpProblem("Observation_Planning", pulp.LpMaximize)

# 定义索引集合
radar_ids = sensor_data.index.tolist()
target_ids = require_data.index.tolist()

# 定义主决策变量：x[r,s,a]，表示雷达r是否在弧段a上对目标s进行观测
x = {}
arc_index = {}  # 存储(r,s)下的弧段编号列表
arc_duration = {}  # 存储每个(r,s,a)对应的观测时长

for (r, s), vis_list in radar_target_vis_dict.items():
    arc_index[(r, s)] = list(range(len(vis_list)))
    for a, (start, end, duration) in enumerate(vis_list):
        x[(r, s, a)] = pulp.LpVariable(f"x_{r}_{s}_{a}", cat="Binary")
        arc_duration[(r, s, a)] = duration

# 定义辅助变量：y[s]，表示目标s是否满足任务要求
y = {s: pulp.LpVariable(f"y_{s}", cat="Binary") for s in target_ids}

# 定义辅助变量：δ[r,s]，表示雷达r是否对目标s形成有效观测
delta = {}
for (r, s) in arc_index:
    delta[(r, s)] = pulp.LpVariable(f"delta_{r}_{s}", cat="Binary")

# 定义辅助变量：z[s,a]，目标s在弧段a上是否至少被某雷达有效观测
z = {}
arc_map = {}  # s -> 所有可观测弧段（全局编号）
for s in target_ids:
    z[s] = {}
    arc_map[s] = set()
    for (r_, s_), arcs in arc_index.items():
        if s_ == s:
            for a in arcs:
                z[s][a] = pulp.LpVariable(f"z_{s}_{a}", cat="Binary")
                arc_map[s].add((r_, a))

# 目标函数：最大化加权任务完成数
prob += pulp.lpSum([priority_weights[s] * y[s] for s in target_ids])

# 约束1：测站数量约束
for s in target_ids:
    prob += pulp.lpSum([delta[(r, s)] for r in radar_ids if (r, s) in delta]) >= required_stations[s] * y[s]

    for r in radar_ids:
        if (r, s) in arc_index:
            total_time = pulp.lpSum([arc_duration[(r, s, a)] * x[(r, s, a)] for a in arc_index[(r, s)]])
            prob += total_time >= required_observation_time[s] * delta[(r, s)]
            prob += delta[(r, s)] <= pulp.lpSum([x[(r, s, a)] for a in arc_index[(r, s)]])

# 约束2：有效弧段数量约束
for s in target_ids:
    prob += pulp.lpSum([z[s][a] for a in z[s]]) >= required_arc_count[s] * y[s]

    for (r, a) in arc_map[s]:
        if (r, s, a) in x:
            prob += z[s][a] >= x[(r, s, a)]

# 约束3：雷达容量约束（按时间离散化）
time_steps = simDate.shape[1]  # 假设simDate列数为时间步长数量
for r in radar_ids:
    for t in range(time_steps):
        # 某一时刻雷达r可用的观测项
        relevant_xs = []
        for (r_, s_), arcs in arc_index.items():
            if r_ != r:
                continue
            for a in arcs:
                s_time, e_time, _ = radar_target_vis_dict[(r, s_)][a]
                if s_time <= start_time + t * u.min <= e_time:
                    relevant_xs.append(x[(r, s_, a)])
        if relevant_xs:
            prob += pulp.lpSum(relevant_xs) <= radar_capacities[r]

print("模型构建完成，开始求解...")
print("变量数量：", len(prob.variables()))
print("约束数量：", len(prob.constraints))
prob.solve()

# 输出结果
print("求解状态：", pulp.LpStatus[prob.status])
print("目标函数值：", pulp.value(prob.objective))
for s in target_ids:
    if pulp.value(y[s]) > 0.5:
        print(f"目标 {s} 完成观测任务，权重 {priority_weights[s]}")

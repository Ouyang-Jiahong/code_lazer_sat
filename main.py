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

# print("模型构建中...")
# prob = pulp.LpProblem("Observation_Planning", pulp.LpMaximize)
#
# # 雷达与目标集合
# radar_ids = sensor_data.index.tolist()
# target_ids = require_data.index.tolist()
#
# # 决策变量 x[r, s, a]：雷达r是否在弧段a上对目标s观测
# x = {}
# arc_index = {}
# arc_duration = {}
#
# for (r, s), vis_list in radar_target_vis_dict.items():
#     arc_index[(r, s)] = []
#     for a, (start, end, duration) in enumerate(vis_list):
#         var = pulp.LpVariable(f"x_{r}_{s}_{a}", cat="Binary")
#         x[(r, s, a)] = var
#         arc_index[(r, s)].append(a)
#         arc_duration[(r, s, a)] = duration
#
# # 变量 y[s]：目标s是否完成任务
# y = {}
# for s in target_ids:
#     y[s] = pulp.LpVariable(f"y_{s}", cat="Binary")
#
# # 变量 delta[r, s]：雷达r是否有效观测了目标s
# delta = {}
# for (r, s) in arc_index:
#     delta[(r, s)] = pulp.LpVariable(f"delta_{r}_{s}", cat="Binary")
#
# # 变量 z[s, a]：目标s在弧段a上是否被观测
# z = {}
# arc_map = {}
#
# for s in target_ids:
#     z[s] = {}
#     arc_map[s] = set()
#     for (r, s_), arcs in arc_index.items():
#         if s_ != s:
#             continue
#         for a in arcs:
#             z[s][a] = pulp.LpVariable(f"z_{s}_{a}", cat="Binary")
#             arc_map[s].add((r, a))
#
# # 目标函数：加权最大任务完成数
# objective = []
# for s in target_ids:
#     objective.append(priority_weights[s] * y[s])
# prob += pulp.lpSum(objective)
#
# # 约束1：测站数量约束 + 有效观测时间
# for s in target_ids:
#     # 至少有 enough stations
#     station_sum = []
#     for r in radar_ids:
#         if (r, s) in delta:
#             station_sum.append(delta[(r, s)])
#     prob += pulp.lpSum(station_sum) >= required_stations[s] * y[s]
#
#     # 每个雷达的时间与delta绑定
#     for r in radar_ids:
#         if (r, s) not in arc_index:
#             continue
#
#         time_sum = []
#         use_sum = []
#         for a in arc_index[(r, s)]:
#             time_sum.append(arc_duration[(r, s, a)] * x[(r, s, a)])
#             use_sum.append(x[(r, s, a)])
#
#         prob += pulp.lpSum(time_sum) >= required_observation_time[s] * delta[(r, s)]
#         prob += delta[(r, s)] <= pulp.lpSum(use_sum)
#
# # 约束2：弧段被观测次数约束
# for s in target_ids:
#     z_sum = []
#     for a in z[s]:
#         z_sum.append(z[s][a])
#     prob += pulp.lpSum(z_sum) >= required_arc_count[s] * y[s]
#
#     for (r, a) in arc_map[s]:
#         key = (r, s, a)
#         if key in x:
#             prob += z[s][a] >= x[key]
#
# # 约束3：雷达同时观测能力限制（每时刻不超过容量）
# time_steps = simDate.shape[1]
# for r in radar_ids:
#     for t in range(time_steps):
#         relevant_xs = []
#
#         for (r2, s), arcs in arc_index.items():
#             if r2 != r:
#                 continue
#
#             for a in arcs:
#                 s_time, e_time, _ = radar_target_vis_dict[(r, s)][a]
#                 current_time = start_time + t * u.min
#                 if s_time <= current_time <= e_time:
#                     relevant_xs.append(x[(r, s, a)])
#
#         if relevant_xs:
#             prob += pulp.lpSum(relevant_xs) <= radar_capacities[r]
#
# # 求解模型
# print("模型构建完成，开始求解...")
# print("变量数量：", len(prob.variables()))
# print("约束数量：", len(prob.constraints))
# prob.solve()
#
# # 输出结果
# print("求解状态：", pulp.LpStatus[prob.status])
# print("目标函数值：", pulp.value(prob.objective))
#
# for s in target_ids:
#     if pulp.value(y[s]) > 0.5:
#         print(f"目标 {s} 完成观测任务，权重 {priority_weights[s]}")
from warnings import catch_warnings

import coptpy as cp
from coptpy import COPT

## 数据导入
# 通过 from data import ... 语句，执行了 data.py 中的预处理逻辑，完成数据加载。
# 如需修改数据路径或调整数据结构，请修改 data.py 文件。
from data import (
    radar_target_vis_dict,   # 测站-目标可见性字典
    sensor_data,             # 测站基础信息（DataFrame）
    require_data,            # 目标观测需求信息（DataFrame）
    required_stations,       # 各目标所需最小测站数量
    priority_weights,        # 各目标优先级权重
    required_observation_time,  # 各目标所需最小累计观测时间（分钟）
    required_arc_count,      # 各目标所需最小有效观测次数
    simDate,                 # 时间节点数据（UTC）
    start_time,              # 仿真起始时间
    radar_capacities         # 各测站最大同时观测能力
)

print("【数据导入】成功加载所有数据！")

# ----------------------------
# 初始化 COPT 环境与模型
# ----------------------------
env = cp.Envr()
model = env.createModel("Radar_Target_Observation_Planning")

# ----------------------------
# 定义集合与索引
# ----------------------------
radars = list(range(len(sensor_data)))                # 雷达编号从 0 开始
targets = list(range(len(require_data)))              # 目标编号从 0 开始

# 构建卫星编号到索引的映射
sat_id_to_index = {}
for idx, sat_id in enumerate(require_data["目标编号"]):
    sat_id_to_index[sat_id] = idx

# 构建雷达编号到索引的映射
radar_id_to_index = {}
for idx, radar_id in enumerate(sensor_data["雷达编号"]):
    radar_id_to_index[radar_id] = idx

# 构建每个雷达-目标的所有可见弧段索引
# 如果arc_indices的元素为[-1]，则说明这个元素对应的雷达-目标没有可见弧段
arc_indices = {}
for (r, s), windows in radar_target_vis_dict.items():
    if len(windows) == 0:
        arc_indices[(radar_id_to_index[r], sat_id_to_index[s])] = [-1]
    else:
        arc_indices[(radar_id_to_index[r], sat_id_to_index[s])] = list(range(len(windows)))

# 时间点总数（用于雷达并发限制）
num_time_points = simDate.shape[1]

# ----------------------------
# 定义决策变量
# ----------------------------
# x[r][s][a]：雷达 r 在第 a 个可见弧段观测目标 s？x[r][s][a]获取的是一个coptpy.Var变量，这个变量可以通过x[r][s][a].x获取其数值，0或1
x = [[[] for _ in range(len(targets))] for _ in range(len(radars))]
for r in radars:
    for s in targets:
        arcs = arc_indices.get((r, s), [])
        for a in arcs:
            var_name = f"x_{r}_{s}_{a}"
            if a == -1:
                x[r][s].append(None)  # 无可见弧段
            else:
                x[r][s].append(model.addVar(vtype=COPT.BINARY, name=var_name))

# y[s]：目标 s 是否被有效观测
y = [model.addVar(vtype=COPT.BINARY, name=f"y_{s}") for s in targets]

# ----------------------------
# 设置目标函数
# ----------------------------
obj_expr = sum(priority_weights[s] * y[s] for s in targets)
model.setObjective(obj_expr, sense=COPT.MAXIMIZE)

# ----------------------------
# 添加约束条件
# ----------------------------


# ----------------------------
# 设置求解参数
# ----------------------------


# ----------------------------
# 求解模型
# ----------------------------
model.solve()

# ----------------------------
# 输出结果
# ----------------------------
if model.status == COPT.OPTIMAL:
    print("找到可行解，目标值为:", model.objval)
    for r in radars:
        for s in targets:
            for a in arc_indices.get((r, s), []):
                if x[r][s][a].x > 0.5:
                   print(f"雷达 {r} 观测目标 {s} 弧段 {a}："
                      f"{radar_target_vis_dict[(r, s)][a][0]} ~ "
                      f"{radar_target_vis_dict[(r, s)][a][1]}")
else:
    print("未找到可行解。")

# ----------------------------
# 可视化
# ----------------------------
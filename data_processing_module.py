import pandas as pd
import coptpy as cp
from coptpy import COPT
import numpy as np

from visible_arc_visualization_app import index_to_utc
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
arc_indices = {}
for (r, s), windows in radar_target_vis_dict.items():
    r_idx = radar_id_to_index[r]
    s_idx = sat_id_to_index[s]
    if len(windows) == 0:
        arc_indices[(r_idx, s_idx)] = [-1]
    else:
        arc_indices[(r_idx, s_idx)] = list(range(len(windows)))

# 时间点总数（用于雷达并发限制）
num_time_points = simDate.shape[1]

# ----------------------------
# 定义决策变量
# ----------------------------
x = {}  # x[r][s][a]: 是否雷达 r 在第 a 个弧段上观测目标 s
y = {}  # y[s]: 是否目标 s 被视为有效观测目标

for r in radars:
    x[r] = {}
    for s in targets:
        x[r][s] = {}
        arcs = arc_indices.get((r, s), [])
        for a in arcs:
            x[r][s][a] = model.addVar(vtype=COPT.BINARY, name=f"x_{r}_{s}_{a}")

for s in targets:
    y[s] = model.addVar(vtype=COPT.BINARY, name=f"y_{s}")

# ----------------------------
# 添加约束条件
# ----------------------------

# 1. 雷达观测容量约束：每部雷达在任一时刻最多只能同时观测 C_r 个目标
arc_time_windows = {}  # arc_time_windows[(r,s,a)] = [t_start, t_end]
for (r, s), windows in radar_target_vis_dict.items():
    r_idx = radar_id_to_index[r]
    s_idx = sat_id_to_index[s]
    for a, (t_start, t_end, duration) in enumerate(windows):
        arc_time_windows[(r_idx, s_idx, a)] = (int(t_start), int(t_end))

# 每个时间点 t 上，雷达 r 所有覆盖 t 的弧段 a
time_to_arcs = {t: {} for t in range(num_time_points)}
for (r, s, a), (t_start, t_end) in arc_time_windows.items():
    for t in range(t_start, t_end + 1):
        if t >= num_time_points:
            continue
        if r not in time_to_arcs[t]:
            time_to_arcs[t][r] = []
        time_to_arcs[t][r].append((s, a))

# 添加雷达容量约束
for t in range(num_time_points):
    for r in radars:
        if r not in time_to_arcs[t]:
            continue
        expr = cp.LinExpr()
        for (s, a) in time_to_arcs[t][r]:
            expr += x[r][s][a]
        model.addConstr(expr <= radar_capacities[r], name=f"capacity_r{r}_t{t}")

# 2. 只允许选择那些满足单次观测时长的弧段
arc_is_valid = {}  # arc_is_valid[(r,s,a)] = True/False 表示该弧段是否满足单次观测时长要求
for (r, s), windows in radar_target_vis_dict.items():
    r_idx = radar_id_to_index[r]
    s_idx = sat_id_to_index[s]
    min_duration = required_observation_time[s_idx]
    for a, (t_start, t_end, duration) in enumerate(windows):
        arc_is_valid[(r_idx, s_idx, a)] = (duration >= min_duration)

# 对每个弧段，如果它不满足单次观测时长，就禁止使用
for r in radars:
    for s in targets:
        arcs = arc_indices.get((r, s), [])
        if arcs == [-1]:
            continue
        for a in arcs:
            if not arc_is_valid.get((r, s, a), False):
                model.addConstr(x[r][s][a] == 0, name=f"invalid_arc_{r}_{s}_{a}")

# 3. 统计每个目标 s 被多少个不同的雷达观测过（至少 M_s^min）
observed_radars = {}  # 辅助变量：统计雷达数
for s in targets:
    observed_radars[s] = model.addVar(vtype=COPT.INTEGER, name=f"observed_radars_{s}")

# 使用辅助二进制变量 z[r][s] 表示雷达 r 是否对目标 s 有过有效观测
for s in targets:
    z = {}
    for r in radars:
        z[r] = model.addVar(vtype=COPT.BINARY, name=f"z_{r}_{s}")
        arcs = arc_indices.get((r, s), [])
        if arcs != [-1]:
            # 如果雷达 r 对目标 s 有任何一个有效弧段被选中，则 z[r] = 1
            valid_arcs = [a for a in arcs if arc_is_valid.get((r, s, a), False)]
            if valid_arcs:
                model.addConstr(cp.quicksum(x[r][s][a] for a in valid_arcs) >= 1e-6 * z[r])
        else:
            model.addConstr(z[r] == 0)
    model.addConstr(observed_radars[s] == cp.quicksum(z[r] for r in radars), name=f"count_radars_{s}")

# 4. 统计每个目标 s 的有效观测次数（至少 N_s^min）
observed_arcs = {}
for s in targets:
    observed_arcs[s] = model.addVar(vtype=COPT.INTEGER, name=f"observed_arcs_{s}")

for s in targets:
    valid_arcs_list = []
    for r in radars:
        arcs = arc_indices.get((r, s), [])
        if arcs == [-1]:
            continue
        for a in arcs:
            if arc_is_valid.get((r, s, a), False):  # 只考虑满足单次观测时长的弧段
                valid_arcs_list.append((r, a))
    if valid_arcs_list:
        model.addConstr(observed_arcs[s] == cp.quicksum(x[r][s][a] for (r, a) in valid_arcs_list),
                        name=f"count_arcs_{s}")
    else:
        model.addConstr(observed_arcs[s] == 0)

# 5. 判断目标是否为有效观测目标
for s in targets:
    # 条件1：观测雷达数 ≥ M_s^min
    model.addConstr(observed_radars[s] >= required_stations[s] * y[s], name=f"require_radars_{s}")
    # 条件2：观测次数 ≥ N_s^min
    model.addConstr(observed_arcs[s] >= required_arc_count[s] * y[s], name=f"require_arcs_{s}")

# ----------------------------
# 设置目标函数
# ----------------------------
# ----------------------------
# 第一阶段：最大化加权有效观测目标数
# ----------------------------
obj_main = cp.quicksum(priority_weights[s] * y[s] for s in targets)
model.setObjective(obj_main, sense=COPT.MAXIMIZE)

# 设置求解参数
model.setParam(COPT.Param.TimeLimit, 3600)  # 设置最大求解时间（秒）

# 求解第一阶段
model.solve()

if model.status == COPT.OPTIMAL:
    best_coverage = model.objval
    print(f"[第一阶段] 主目标最优值为：{best_coverage}")

    # ----------------------------
    # 添加约束：保持主目标值不低于最优值
    # ----------------------------
    model.addConstr(obj_main >= best_coverage - 1e-6, name="Fix_Coverage")

    # ----------------------------
    # 第二阶段：最小化雷达观测弧段数量
    # ----------------------------
    total_obs = cp.quicksum(
        x[r][s][a] for r in radars
        for s in targets
        for a in arc_indices.get((r, s), [])
    )
    model.setObjective(total_obs, sense=COPT.MINIMIZE)

    # 继续求解第二阶段
    model.solve()
    minimum_arc_num = model.objval
    if model.status == COPT.OPTIMAL:
        print(f"[第二阶段] 最终雷达观测弧段总数为：{total_obs.getValue()}")
    else:
        print("未找到满足条件的更优观测调度方案。")
else:
    print("未找到可行解。")

# ----------------------------
# 输出结果
# ----------------------------
if model.status == COPT.OPTIMAL:
    print("求解成功！")
    print(f"最终目标值（加权有效观测目标数）: {best_coverage}")
    print(f"最终雷达观测弧段总数为：: {minimum_arc_num}")
    schedule = []
    for r in radars:
        for s in targets:
            arcs = arc_indices.get((r, s), [])
            if arcs == [-1]:
                continue
            for a in arcs:
                if x[r][s][a].x > 0.5:
                    radar_name = sensor_data.iloc[r]["雷达编号"]
                    target_name = require_data.iloc[s]["目标编号"]
                    window = radar_target_vis_dict[(radar_name, target_name)][a]
                    schedule.append({
                        "雷达": radar_name,
                        "目标": target_name,
                        "弧段": a,
                        "开始时间(UTC)": index_to_utc(window[0]),
                        "结束时间(UTC)": index_to_utc(window[1]),
                        "持续时间(min)": window[2],
                    })
    # 转换为 DataFrame 并格式化显示
    df = pd.DataFrame(schedule)
    print("\n 部分调度方案摘要（前10项）：")
    print(df.head(10).to_string(index=False))

    # ----------------------------
    # 计算覆盖率：有效观测目标数量 / 总目标数量
    # ----------------------------
    num_targets = len(targets)
    num_effective = sum(round(y[s].x) for s in targets)

    coverage_rate = num_effective / num_targets

    print(f"\n【覆盖率统计】")
    print(f"总目标数量: {num_targets}")
    print(f"有效观测目标数量: {num_effective}")
    print(f"覆盖率: {coverage_rate:.2%}")

    # ----------------------------
    # 按优先级统计覆盖率（可选）
    # ----------------------------
    high_priority_count = 0
    high_priority_effective = 0

    medium_priority_count = 0
    medium_priority_effective = 0

    low_priority_count = 0
    low_priority_effective = 0

    for s in targets:
        weight = priority_weights[s]
        if weight >= 9:
            high_priority_count += 1
            high_priority_effective += round(y[s].x)
        elif weight >= 7:
            medium_priority_count += 1
            medium_priority_effective += round(y[s].x)
        else:
            low_priority_count += 1
            low_priority_effective += round(y[s].x)

    print("\n【按优先级统计覆盖率】")
    if high_priority_count > 0:
        print(
            f"高优先级目标覆盖率（优先级大于8）: {high_priority_effective}/{high_priority_count} -> {high_priority_effective / high_priority_count:.2%}")
    if medium_priority_count > 0:
        print(
            f"中优先级目标覆盖率（优先级大于6）: {medium_priority_effective}/{medium_priority_count} -> {medium_priority_effective / medium_priority_count:.2%}")
    if low_priority_count > 0:
        print(
            f"低优先级目标覆盖率: {low_priority_effective}/{low_priority_count} -> {low_priority_effective / low_priority_count:.2%}")
else:
    print("未找到可行解。")

# ----------------------------
# 可视化（后续扩展）
# ----------------------------
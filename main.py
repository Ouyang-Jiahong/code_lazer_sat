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
# 注意：我们遍历每一个时间点 t，并找出所有覆盖该时间点的弧段
# 先为每个 (r,s,a) 构造其时间范围
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

# 2. 目标有效性判定约束
# 每个目标 s 必须满足：
# - 至少 M_s^min 个不同雷达参与观测
# - 至少 N_s^min 个有效观测弧段
# - 总观测时长 ≥ T_s^min

# 辅助变量：统计雷达数、观测次数、总观测时长
observed_radars = {}  # 统计每个目标 s 被多少个雷达观测过
observed_arcs = {}    # 统计每个目标 s 被观测了多少次
total_duration = {}   # 统计每个目标 s 的总观测时长

for s in targets:
    observed_radars[s] = model.addVar(vtype=COPT.INTEGER, name=f"observed_radars_{s}")
    observed_arcs[s] = model.addVar(vtype=COPT.INTEGER, name=f"observed_arcs_{s}")
    total_duration[s] = model.addVar(vtype=COPT.CONTINUOUS, name=f"total_duration_{s}")

# 计算每个目标的观测雷达数、观测次数和总观测时间
for s in targets:
    # 观测雷达数：使用辅助二进制变量 z[r][s] 表示雷达 r 是否对目标 s 有过观测
    z = {}
    for r in radars:
        z[r] = model.addVar(vtype=COPT.BINARY, name=f"z_{r}_{s}")
        arcs = arc_indices.get((r, s), [])
        if arcs != [-1]:
            model.addConstr(cp.quicksum(x[r][s][a] for a in arcs) >= 1e-6 * z[r])  # 若有至少一个弧段被选中，则 z[r]=1
        else:
            model.addConstr(z[r] == 0)
    model.addConstr(observed_radars[s] == cp.quicksum(z[r] for r in radars), name=f"count_radars_{s}")

    # 观测次数
    arcs_all = []
    for r in radars:
        arcs = arc_indices.get((r, s), [])
        if arcs != [-1]:
            arcs_all.extend([(r, a) for a in arcs])
    if arcs_all:
        model.addConstr(observed_arcs[s] == cp.quicksum(x[r][s][a] for (r, a) in arcs_all), name=f"count_arcs_{s}")
    else:
        model.addConstr(observed_arcs[s] == 0)

    # 总观测时长（分钟）
    total_duration_expr = cp.LinExpr()
    for r in radars:
        arcs = arc_indices.get((r, s), [])
        if arcs == [-1]:
            continue
        for a in arcs:
            _, _, duration = radar_target_vis_dict[(sensor_data.iloc[r]["雷达编号"], require_data.iloc[s]["目标编号"])][a]
            total_duration_expr += duration * x[r][s][a]
    model.addConstr(total_duration[s] == total_duration_expr, name=f"duration_{s}")

# 最后连接到 y[s]
for s in targets:
    # 条件1：观测雷达数 ≥ M_s^min
    model.addConstr(observed_radars[s] >= required_stations[s] * y[s], name=f"require_radars_{s}")
    # 条件2：观测次数 ≥ N_s^min
    model.addConstr(observed_arcs[s] >= required_arc_count[s] * y[s], name=f"require_arcs_{s}")
    # 条件3：观测时间 ≥ T_s^min
    model.addConstr(total_duration[s] >= required_observation_time[s] * y[s], name=f"require_duration_{s}")

# ----------------------------
# 设置目标函数
# ----------------------------
obj = cp.quicksum(priority_weights[s] * y[s] for s in targets)
model.setObjective(obj, COPT.MAXIMIZE)

# ----------------------------
# 设置求解参数（可选）
# ----------------------------
model.setParam(COPT.Param.TimeLimit, 3600)  # 设置最大求解时间（秒）

# ----------------------------
# 求解模型
# ----------------------------
model.solve()

# ----------------------------
# 输出结果
# ----------------------------
if model.status == COPT.OPTIMAL or model.status == COPT.FEASIBLE:
    print("求解成功！")
    print(f"最终目标值（加权有效观测目标数）: {model.objval}")
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
                        "开始时间(UTC)": simDate[0][window[0]],
                        "结束时间(UTC)": simDate[0][window[1]],
                        "持续时间(min)": window[2],
                    })
    print("\n部分调度方案摘要：")
    for entry in schedule[:10]:  # 只显示前10项
        print(entry)
else:
    print("未找到可行解。")

# ----------------------------
# 可视化（后续扩展）
# ----------------------------
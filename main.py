import pandas as pd
import pulp
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation
from matplotlib import rcParams
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 加载数据
sensor_data = pd.read_excel("simData/sensorData.xlsx")
require_data = pd.read_excel("simData/requireData.xlsx")

# 参数准备
num_radars = len(sensor_data)
num_targets = len(require_data)

start_time = Time("2021-10-14T04:00:00", format='isot', scale='utc')

def create_orbit_from_elements(row):
    a = row["半长轴（km)"] * u.km
    ecc = row["偏心率"] * u.one
    inc = row["轨道倾角(°)"] * u.deg
    raan = row["升交点赤经(°)"] * u.deg
    argp = row["近地点幅角(°）"] * u.deg
    nu = row["平近点角(°)"] * u.deg
    return Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)

def radar_to_ecef(radar_row):
    lon = radar_row["部署经度(°）"] * u.deg
    lat = radar_row["部署纬度(°）"] * u.deg
    alt = radar_row["部署高度(km)"] * u.km
    loc = EarthLocation(lon=lon, lat=lat, height=alt)
    return loc.get_itrs(obstime=start_time).cartesian.xyz.to(u.km)

orbits = [create_orbit_from_elements(require_data.iloc[i]) for i in range(num_targets)]
radar_positions = [radar_to_ecef(sensor_data.iloc[i]) for i in range(num_radars)]
radar_capacities = sensor_data["最大探测目标数"].values
required_stations = require_data["需要的测站数量"].values
required_observation_time = require_data["需要的观测时间(min)"].values
required_arc_count = require_data["需要的弧段数量"].values
priority_weights = require_data["优先级(数值越大，优先级越高)"].values

# 加载可见弧段
usable_arcs = sio.loadmat("simData/usableArcs.mat")["usableArcs"]
simDate = sio.loadmat("simData/simDate.mat")["simDate"]

radar_target_vis_dict = {}
for i in range(len(usable_arcs[0])):
    sat_id = usable_arcs[0][i][0][0][0]
    radar_id = usable_arcs[0][i][1][0][0]
    arc_chain = usable_arcs[0][i][2]
    arc_durations = usable_arcs[0][i][3]
    visible_windows = []
    for j in range(arc_chain.shape[0]):
        s_idx = arc_chain[j, 0] - 1
        e_idx = arc_chain[j, 1] - 1
        s_time = Time(f"{int(simDate[0,s_idx])}-{int(simDate[1,s_idx]):02d}-{int(simDate[2,s_idx]):02d}T"
                      f"{int(simDate[3,s_idx]):02d}:{int(simDate[4,s_idx]):02d}:{int(simDate[5,s_idx]):02d}",
                      format='isot', scale='utc')
        e_time = Time(f"{int(simDate[0,e_idx])}-{int(simDate[1,e_idx]):02d}-{int(simDate[2,e_idx]):02d}T"
                      f"{int(simDate[3,e_idx]):02d}:{int(simDate[4,e_idx]):02d}:{int(simDate[5,e_idx]):02d}",
                      format='isot', scale='utc')
        visible_windows.append((s_time, e_time, arc_durations[j, 0]))
    radar_target_vis_dict[(radar_id, sat_id)] = visible_windows

# 构建模型
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
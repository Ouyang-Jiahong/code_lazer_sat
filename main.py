import pprint

import pandas as pd
import pulp
from astropy.time import Time
from matplotlib import pyplot as plt
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.coordinates import EarthLocation
import scipy.io as sio

## 加载数据
print("数据加载开始...")
sensor_data = None
require_data = None
try:
    sensor_path = r"simData/sensorData.xlsx"
    require_path = r"simData/requireData.xlsx"
    sensor_data = pd.read_excel(sensor_path)
    require_data = pd.read_excel(require_path)
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except pd.errors.ParserError as e:
    print(f"Excel 文件解析失败: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")
else:
    print("数据加载成功")
finally:
    print("数据加载结束。")

## 参数设置
print("参数设置开始...")
# 检查 sensor_data 和 require_data 是否已加载成功
if sensor_data is None or require_data is None:
    raise ValueError("数据未正确加载，请先完成数据读取。")

try:
    num_radars = len(sensor_data)
    num_targets = len(require_data)

    # 设置探测时间段：UTC 时间
    start_time_str = "2021-10-14T04:00:00"
    end_time_str = "2021-10-15T04:00:00"

    # 使用 astropy 定义时间
    start_time = Time(start_time_str, format='isot', scale='utc')
    end_time = Time(end_time_str, format='isot', scale='utc')

    # 计算总时间片数量（按每分钟划分）
    dt_per_minute = 60 * u.s  # 每分钟的时间间隔
    total_seconds = (end_time - start_time) * 86400  # 转换为秒
    time_slots = int(total_seconds.value // dt_per_minute.value)

    ## 相关函数定义
    def create_orbit_from_elements(row):
        try:
            a = row["半长轴（km)"] * u.km
            ecc = row["偏心率"] * u.one
            inc = row["轨道倾角(°)"] * u.deg
            raan = row["升交点赤经(°)"] * u.deg
            argp = row["近地点幅角(°）"] * u.deg
            nu = row["平近点角(°)"] * u.deg

            return Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
        except KeyError as e:
            raise KeyError(f"轨道参数列缺失: {e}")
        except Exception as e:
            raise ValueError(f"轨道参数转换失败: {e}")

    def radar_to_ecef(radar_row):
        try:
            lon = radar_row["部署经度(°）"] * u.deg
            lat = radar_row["部署纬度(°）"] * u.deg
            alt = radar_row["部署高度(km)"] * u.km
            loc = EarthLocation(lon=lon, lat=lat, height=alt)
            return loc.get_itrs(obstime=start_time).cartesian.xyz.to(u.km)
        except KeyError as e:
            raise KeyError(f"雷达部署信息列缺失: {e}")
        except Exception as e:
            raise ValueError(f"雷达位置转换失败: {e}")

    # 创建轨道和雷达位置列表
    orbits = [create_orbit_from_elements(require_data.iloc[i]) for i in range(num_targets)]
    radar_positions = [radar_to_ecef(sensor_data.iloc[i]) for i in range(num_radars)]

    # 提取观测时间和雷达容量
    required_times = require_data["需要的观测时间(min)"].values
    radar_capacities = sensor_data["最大探测目标数"].values

    # 创建优化问题
    prob = pulp.LpProblem("Space_Target_Detection_Planning", pulp.LpMaximize)

except KeyError as e:
    print(f"数据列缺失或名称错误: {e}")
except TypeError as e:
    print(f"数据类型错误或未正确加载: {e}")
except Exception as e:
    print(f"参数设置过程中发生错误: {e}")
else:
    print("参数设置成功")
finally:
    print("参数设置结束")
    
## 观测弧段导入
print("加载可见弧段时间数据...")
try:
    # 加载可用弧段数据
    usable_arcs_path = r"simData/usableArcs.mat"
    usable_arcs_data = sio.loadmat(usable_arcs_path)['usableArcs']

    # print(usable_arcs_data)
    # 加载时间戳数据
    simDate_path = r"simData/simDate.mat"
    simDate = sio.loadmat(simDate_path)['simDate']

    # 构建每个雷达-卫星对的可见弧段列表
    radar_target_visibilities = []
    for i in range(len(usable_arcs_data[0])):
        sat_id = usable_arcs_data[0][i][0][0][0]  # 卫星索引
        radar_id = usable_arcs_data[0][i][1][0][0]  # 雷达索引
        arc_chain = usable_arcs_data[0][i][2] # 弧段时间范围（列索引）
        arc_durations = usable_arcs_data[0][i][3]  # 每个弧段时长（秒）

        # 转换为起止时间戳（UTC 时间）
        visible_windows = []
        for j in range(arc_chain.shape[0]):
            start_idx = arc_chain[j, 0] - 1  # MATLAB 是1-based索引，start_idx是作用在python array上的，所以需要减去1
            end_idx = arc_chain[j, 1] - 1
            start_time_utc = Time(
                f"{int(simDate[0, start_idx])}-{int(simDate[1, start_idx]):02d}-{int(simDate[2, start_idx]):02d}T"
                f"{int(simDate[3, start_idx]):02d}:{int(simDate[4, start_idx]):02d}:{int(simDate[5, start_idx]):02d}",
                format='isot', scale='utc'
            )
            end_time_utc = Time(
                f"{int(simDate[0, end_idx])}-{int(simDate[1, end_idx]):02d}-{int(simDate[2, end_idx]):02d}T"
                f"{int(simDate[3, end_idx]):02d}:{int(simDate[4, end_idx]):02d}:{int(simDate[5, end_idx]):02d}",
                format='isot', scale='utc'
            )

            visible_windows.append((start_time_utc, end_time_utc, arc_durations[j, 0]))

        radar_target_visibilities.append({"radar_id": radar_id, "sat_id": sat_id, "visible_windows": visible_windows})

    radar_target_vis_dict = {}
    for item in radar_target_visibilities:
        key = (item["radar_id"], item["sat_id"])
        radar_target_vis_dict[key] = item["visible_windows"]

except Exception as e:
    print(f"加载可见弧段失败: {e}")
else:
    print("可见弧段数据加载成功")

x = [[pulp.LpVariable(f"x_{r}_{t}", cat="Binary") for t in range(num_targets)] for r in range(num_radars)]
y = []
for r in range(num_radars):
    y_radar = []
    for t in range(num_targets):
        key = (r, t)
        visibles = radar_target_vis_dict.get(key, [])
        y_target = [pulp.LpVariable(f"y_{r}_{t}_{a}", cat="Binary") for a in range(len(visibles))]
        y_radar.append(y_target)
    y.append(y_radar)

print("决策变量设置完成")

print("目标函数设置开始...")
priority_weights = require_data["优先级(数值越大，优先级越高)"].values
prob += pulp.lpSum(priority_weights[t] * x[r][t] for r in range(num_radars) for t in range(num_targets))
print("目标函数设置完成")

print("约束条件设置开始...")
required_stations = require_data["需要的测站数量"].values
required_observation_time = require_data["需要的观测时间(min)"].values

for t in range(num_targets):
    prob += pulp.lpSum([x[r][t] for r in range(num_radars)]) >= required_stations[t]
    total_time = []
    for r in range(num_radars):
        key = (r, t)
        visibles = radar_target_vis_dict.get(key, [])
        total_time.append(pulp.lpSum([
            y[r][t][a] * visibles[a][2] / 60 for a in range(len(visibles))
        ]))
    prob += pulp.lpSum(total_time) >= required_observation_time[t]

for r in range(num_radars):
    prob += pulp.lpSum([x[r][t] for t in range(num_targets)]) <= radar_capacities[r]

print("约束条件设置完成")

print("开始求解...")
result_status = prob.solve(pulp.PULP_CBC_CMD())
print(f"求解完成，状态: {pulp.LpStatus[result_status]}")

if pulp.LpStatus[result_status] == 'Infeasible':
    print("模型无可行解，请检查约束条件是否过于严格。")
else:
    for r in range(num_radars):
        for t in range(num_targets):
            for a in range(len(y[r][t])):
                if y[r][t][a].value() is not None and y[r][t][a].value() > 0.5:
                    print(f"雷达{r}对目标{t}在弧段{a}进行观测。")


# 获取结果
selected_assignments = [
    (r, t) for r in range(num_radars) for t in range(num_targets) if x[r][t].value() > 0.5
]

selected_observations = [
    (r, t, a) for r in range(num_radars) for t in range(num_targets)
    for a in range(len(radar_target_visibilities[(r * num_targets + t)]["visible_windows"]))
    if y[r][t][a].value() > 0.5
]

# 输出结果
print("\n--- 规划结果 ---")
for r, t in selected_assignments:
    print(f"雷达 {r+1} 分配给了目标 {t+1}")
    key = (r, t)
    if key in radar_target_vis_dict:
        visibles = radar_target_vis_dict[key]
        for a in range(len(visibles)):
            if y[r][t][a].value() > 0.5:
                start = visibles[a][0].isot
                end = visibles[a][1].isot
                dur = visibles[a][2] / 60
                print(f"  使用可见弧段: {start} ~ {end}, 持续 {dur:.2f} 分钟")

## 数据可视化
def plot_schedule(assignments, observations):
    fig, ax = plt.subplots(figsize=(14, 8))

    radar_names = [f"雷达{i+1}" for i in range(num_radars)]
    target_names = [f"目标{j+1}" for j in range(num_targets)]

    # 给不同目标分配不同颜色
    import matplotlib.cm as cm
    import numpy as np

    cmap = cm.get_cmap('tab20', num_targets)
    colors = [cmap(i) for i in range(num_targets)]

    yticks = []
    ytick_labels = []
    y_base = 0

    for r in range(num_radars):
        for t in range(num_targets):
            for a in range(len(y[r][t])):
                if y[r][t][a].value() > 0.5:
                    start_time = radar_target_vis_dict[(r, t)][a][0].datetime
                    end_time = radar_target_vis_dict[(r, t)][a][1].datetime
                    ax.barh(y_base, (end_time - start_time).total_seconds() / 60,  # 时长（分钟）
                            left=start_time,
                            height=0.8,
                            color=colors[t],
                            edgecolor='black')
                    yticks.append(y_base)
                    ytick_labels.append(f"{radar_names[r]} → {target_names[t]}")
                    y_base += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_title("雷达观测任务排程图", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

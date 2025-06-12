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

    for i in range(len(usable_arcs_data)):
        sat_id = usable_arcs_data[i][0][0][0][0]  # 卫星索引
        radar_id = usable_arcs_data[i][0][1][0][0]  # 雷达索引
        arc_chain = usable_arcs_data[i][0][2] # 弧段时间范围（列索引）
        arc_durations = usable_arcs_data[i][0][3]  # 每个弧段时长（秒）

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

        radar_target_visibilities.append({
            "radar_id": radar_id,
            "sat_id": sat_id,
            "visible_windows": visible_windows
        })

except Exception as e:
    print(f"加载可见弧段失败: {e}")
else:
    print("可见弧段数据加载成功")
    
## 决策变量设置
# 创建决策变量：x[r][t] 表示雷达r是否分配给目标t
x = pulp.LpVariable.dicts("RadarTargetAssignment", 
                          [(r, t) for r in range(num_radars) for t in range(num_targets)], 
                          cat='Binary')

pprint.pprint(radar_target_visibilities)

# y[r][t][a] 表示雷达r对目标t在第a个可见弧段是否执行探测
y = pulp.LpVariable.dicts("ObservationArcSelection",
                          [(r, t, a) for r in range(num_radars) for t in range(num_targets)
                           for a in range(len(radar_target_visibilities[(r * num_targets + t)]["visible_windows"]))],
                          cat='Binary')

## 目标函数设置
# 目标函数：最大化被探测目标的总优先级权重
prob += pulp.lpSum([
    x[r][t] * require_data.loc[t, '优先级'] 
    for r in range(num_radars) 
    for t in range(num_targets)
]), "Maximize_Total_Priority"

## 约束条件设置
# 1. 每个目标必须满足所需的最小探测次数和时间
for t in range(num_targets):
    required_obs_count = require_data.loc[t, '需要的弧段数量']
    required_obs_time = require_data.loc[t, '需要的观测时间(min)'] * 60  # 转换为秒

    prob += pulp.lpSum([
        y[r][t][a] for r in range(num_radars)
        for a in range(len(radar_target_visibilities[(r * num_targets + t)]["visible_windows"]))
    ]) >= required_obs_count, f"Min_Observations_Target_{t}"

    prob += pulp.lpSum([
        y[r][t][a] * radar_target_visibilities[(r * num_targets + t)]["visible_windows"][a][2]
        for r in range(num_radars)
        for a in range(len(radar_target_visibilities[(r * num_targets + t)]["visible_windows"]))
    ]) >= required_obs_time, f"Min_Observation_Time_Target_{t}"
    
# 2. 每个雷达最多探测其容量内的目标数
for r in range(num_radars):
    max_capacity = radar_capacities[r]
    prob += pulp.lpSum([x[r][t] for t in range(num_targets)]) <= max_capacity, f"Radar_Capacity_{r}"
    
# 3. 若雷达未分配给某目标，则不能对该目标进行探测
for r in range(num_radars):
    for t in range(num_targets):
        for a in range(len(radar_target_visibilities[(r * num_targets + t)]["visible_windows"])):
            prob += y[r][t][a] <= x[r][t], f"No_Observation_Without_Assignment_{r}_{t}_{a}"

## 问题求解
# 求解优化问题
print("开始求解...")
prob.solve()

# 输出求解状态
print("求解完成，状态:", pulp.LpStatus[prob.status])

# 获取结果
selected_assignments = [
    (r, t) for r in range(num_radars) for t in range(num_targets) if x[r][t].value() > 0.5
]

selected_observations = [
    (r, t, a) for r in range(num_radars) for t in range(num_targets)
    for a in range(len(radar_target_visibilities[(r * num_targets + t)]["visible_windows"]))
    if y[r][t][a].value() > 0.5
]

## 数据可视化
def plot_schedule(assignments, observations):
    fig, ax = plt.subplots(figsize=(12, 8))

    for r, t in assignments:
        for a in range(len(radar_target_visibilities[(r * num_targets + t)]["visible_windows"])):
            if (r, t, a) in observations:
                start_time = radar_target_visibilities[(r * num_targets + t)]["visible_windows"][a][0]
                end_time = radar_target_visibilities[(r * num_targets + t)]["visible_windows"][a][1]
                duration = (end_time - start_time).to_value(u.min)

                # 绘制时间条形图
                ax.barh(f"雷达{r+1}-目标{t+1}", width=duration,
                        left=(start_time - start_time).to_value(u.min),
                        edgecolor='black', label=f"雷达{r+1} -> 目标{t+1}")

    ax.set_xlabel("时间 (分钟)")
    ax.set_title("空间目标探测任务调度计划")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

plot_schedule(selected_assignments, selected_observations)
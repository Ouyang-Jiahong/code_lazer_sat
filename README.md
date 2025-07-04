# 空间目标探测任务规划与评估任务书

## 运行环境
本项目的运行环境为 Python 3.13.2。请确保您已经安装了该版本的 Python，再按照上述说明安装所需的库。

## 相关库安装

请使用以下命令安装所需的 Python 库。将 `XXX` 替换为下方列出的库名称：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple XXX
```

需要安装的库包括：

- `numpy`
- `pandas`
- `openpyxl`
- `deap`
- `poliastro`
- `dash`
- `plotly`
- `scipy`
- `astropy`
- `coptpy`

> **注意：** 若在安装过程中遇到网络限制问题（如大量下载行为被拦截），建议更换网络环境或尝试其他镜像源。

### COPT库的安装

本项目使用杉数科技提供的商业混合整数规划求解器 COPT（Cardinal Optimizer）。该求解器支持高性能求解大规模 MILP、MIQP 等优化问题，并支持 Python 接口与 Gurobi 接近。具体使用过程中，通过构建 COPT 模型对象、添加变量与约束、设置目标函数并调用 model.solve() 进行求解。

要安装此求解器，首先需要在``https://www.shanshu.ai/solver`` 上下载，然后进行安装。同时，要申请求解器的`license`，具体安装方式，见网站。


## 代码文件说明
### data_processing_module.py
该文件实现了空间目标探测任务的规划求解，使用 `copt` 库构建线性规划模型，包含数据导入、预处理、求解器参数设置、模型构建和求解等步骤。求解完成后会输出调度方案摘要，还可用于后续可视化扩展。

### data.py
负责数据加载和预处理，从 Excel 文件和 MATLAB 数据文件中读取测站参数、任务需求、可见弧段等数据，并构建雷达 - 目标可见性字典。

### visible_arc_visualization_app.py
使用 Dash 框架开发的可视化工具，提供一个前端页面，用户可以通过下拉框选择目标编号，查看该目标的可见弧段时序图。

### result_show_app.py
同样基于 Dash 框架开发，用于以 HTML 形式显示 `main.py` 中计算出的雷达调度结果。支持选择目标和雷达编号，展示对应的调度表格和甘特图。

### plottemp.py
弧段切割功能的可视化，用于绘制图表。

## 任务场景描述

本任务旨在利用一组雷达设备对给定的空间目标进行探测任务规划与评估。具体包括：

- 使用多种类型的雷达进行探测，包括**相控阵雷达**和**机械跟踪雷达**。
- 雷达参数详见 `sensorData.xlsx` 文件，包含部署位置、探测能力等信息。
- 每个空间目标的轨道参数及其探测需求（如所需测站数量、最小探测次数、最短探测时间等）详见 `requireData.xlsx` 文件。
- 探测时间段为：**2021年10月14日 4时0分0秒 至 2021年10月15日 4时0分0秒（UTCG时间）**

### 输入数据

- 雷达配置信息
- 空间目标轨道数据
- 探测任务需求

### 输出结果
输出应包含每部雷达在指定时间段内对各目标的详细探测安排，包括：

- 探测起止时间
- 是否执行探测任务
- 探测顺序安排

### 运行步骤
1. 安装所需的 Python 库。
2. 确保 `simData` 目录下的 `sensorData.xlsx`、`requireData.xlsx`、`usableArcs.mat` 和 `simDate.mat` 等文件存在。
3. 运行 `data_processing_module.py` 进行任务规划求解。
4. 也可以运行 `visible_arc_visualization_app.py` 查看目标可见弧段时序图。
5. 也可以运行 `result_show_app.py` 查看雷达调度结果（若运行这个，则需要重新运行一次`data_processing_module.py`，所以建议直接运行这个程序）。
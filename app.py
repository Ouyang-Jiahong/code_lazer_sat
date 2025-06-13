import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

from main import radar_target_vis_dict  # 从主程序中导入雷达-目标-弧段字典

# 构造可视化数据表：遍历每一个 (雷达, 目标) 的可见弧段列表，生成记录
records = []
for (r, s), arc_list in radar_target_vis_dict.items():
    for a_idx, (start, end, duration) in enumerate(arc_list):
        records.append({
            "radar": r,                      # 雷达编号
            "target": s,                     # 目标编号
            "arc_index": a_idx,              # 弧段编号（每对 r,s 内部编号）
            "start": start.utc.datetime,     # 弧段起始时间（UTC时间戳）
            "end": end.utc.datetime,         # 弧段终止时间（UTC时间戳）
            "duration_min": duration         # 弧段持续时间（单位可根据需要处理）
        })

# 将所有记录构建为 DataFrame，便于后续绘图
df_arcs = pd.DataFrame(records)

# 启动 Dash 应用
app = dash.Dash(__name__)

# 定义前端页面布局：包含标题、下拉框和图表区域
app.layout = html.Div([
    html.H2("目标任务可见弧段可视化工具"),  # 页面主标题

    html.Label("选择目标编号："),           # 下拉选择说明

    dcc.Dropdown(
        id='target-dropdown',              # 下拉框控件 ID
        options=[
            {'label': f"目标 {s}", 'value': s}
            for s in sorted(df_arcs['target'].unique())  # 按目标编号构建选项
        ],
        value=sorted(df_arcs['target'].unique())[0]      # 默认选择第一个目标
    ),

    dcc.Graph(id='arc-plot')               # 图形展示区域
])

# 定义回调函数：响应目标编号选择事件，更新时间轴图
@app.callback(
    Output('arc-plot', 'figure'),          # 输出到图表组件
    Input('target-dropdown', 'value')      # 输入来自下拉框组件
)
def update_arc_plot(selected_target):
    # 根据选择的目标筛选数据
    df = df_arcs[df_arcs['target'] == selected_target]

    # 若该目标无可视弧段，则返回空图提示
    if df.empty:
        return px.scatter(title=f"目标 {selected_target} 无可用弧段")

    # 使用 Plotly Express 绘制时间轴图（水平条形图）
    fig = px.timeline(
        df,
        x_start="start",       # 时间轴起点
        x_end="end",           # 时间轴终点
        y="radar",             # Y轴为雷达编号
        color="radar",         # 不同雷达上色
        hover_data=["duration_min", "arc_index"],  # 鼠标悬停显示信息
        title=f"目标 {selected_target} 可见弧段时序图"
    )

    # 设置纵轴方向为“从上到下”，符合时间轴习惯
    fig.update_yaxes(autorange="reversed", title="测站编号")

    # 调整整体布局：高度、边距、标题
    fig.update_layout(
        height=600,
        xaxis_title="时间（UTC）",
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig

# 启动本地服务器运行 Dash 应用（调试模式）
if __name__ == '__main__':
    app.run(
        host='127.0.0.1',     # 设置主机地址（默认也是这个）
        port=25525,           # 设置自定义端口
        debug=False           # 关闭调试模式
    )

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 从 main.py 导入必要的变量和函数
try:
    from main import require_data, sensor_data, radar_target_vis_dict, index_to_utc, radars, targets, arc_indices, x
except ImportError as e:
    raise ImportError("请确保 main.py 中已定义并导出以下变量：radars, targets, arc_indices, x") from e

# 初始化 Dash 应用
app = dash.Dash(__name__)

# ----------------------------
# 构建可视化数据（模拟求解结果）
# ----------------------------
def build_schedule_data():
    schedule = []
    for r in radars:
        for s in targets:
            arcs = arc_indices.get((r, s), [])
            if arcs == [-1]:
                continue
            for a in arcs:
                try:
                    # 假设这是调度选择条件
                    if x[r][s][a].x > 0.5:
                        radar_name = sensor_data.iloc[r]["雷达编号"]
                        target_name = require_data.iloc[s]["目标编号"]
                        window = radar_target_vis_dict[(radar_name, target_name)][a]
                        schedule.append({
                            "雷达": radar_name,
                            "目标": target_name,
                            "弧段": a,
                            "开始时间": index_to_utc(window[0]),
                            "结束时间": index_to_utc(window[1]),
                            "持续时间(min)": window[2],
                        })
                except Exception as e:
                    print(f"解析弧段 ({r}, {s}, {a}) 时出错: {e}")
    return pd.DataFrame(schedule)

# ----------------------------
# 页面布局
# ----------------------------

unique_targets = ['全部'] + require_data["目标编号"].unique().tolist()

app.layout = html.Div([
    html.H2("雷达目标观测调度方案结果展示", style={'textAlign': 'center'}),

    html.Div([
        html.Label("选择目标编号："),
        dcc.Dropdown(
            id='target-select',
            options=[{'label': target, 'value': target} for target in unique_targets],
            value='全部'
        ),
    ], style={'width': '80%', 'margin': 'auto', 'padding': '10px'}),

    html.Div(id='output', children=[
        dcc.Graph(id='result-table'),
        dcc.Graph(id='gantt-chart')
    ])
])

# ----------------------------
# 回调函数：根据选择更新表格和甘特图
# ----------------------------
@app.callback(
    [Output('result-table', 'figure'),
     Output('gantt-chart', 'figure')],
    Input('target-select', 'value')
)
def update_output(selected_target):
    df = build_schedule_data()
    if df.empty:
        empty_fig = go.Figure(layout=go.Layout(title="无可用数据"))
        return empty_fig, empty_fig

    if selected_target != '全部':
        df = df[df['目标'] == selected_target]

    # 表格图表
    table_fig = go.Figure(data=[
        go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=[df[col] for col in df.columns])
        )
    ])

    # 甘特图
    gantt_fig = px.timeline(
        df, x_start="开始时间", x_end="结束时间", y="雷达", color="目标",
        title=f"目标 {selected_target} 的雷达观测调度甘特图"
    )
    gantt_fig.update_yaxes(autorange="reversed")
    gantt_fig.update_layout(xaxis_title="时间（UTC）", yaxis_title="雷达")

    return table_fig, gantt_fig

# ----------------------------
# 运行应用
# ----------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=25526, debug=False)
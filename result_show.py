import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# 从 main.py 导入变量
try:
    from main import require_data, sensor_data, radar_target_vis_dict, index_to_utc, radars, targets, arc_indices, x
except ImportError as e:
    raise ImportError("请确保 main.py 中已定义并导出以下变量：radars, targets, arc_indices, x") from e

# 初始化 Dash 应用
app = dash.Dash(__name__)
app.title = "雷达调度可视化"

# ----------------------------
# 构建调度数据（模拟求解结果）
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

# 获取所有雷达和目标编号
unique_radars = ['全部'] + [sensor_data.iloc[r]["雷达编号"] for r in radars]
unique_targets = ['全部'] + list(require_data["目标编号"].unique())

# ----------------------------
# 页面布局
# ----------------------------
app.layout = html.Div([
    html.H2("雷达目标观测调度方案结果展示", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label("选择目标编号："),
            dcc.Dropdown(
                id='target-select',
                options=[{'label': target, 'value': target} for target in unique_targets],
                value='全部'
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("选择雷达编号："),
            dcc.Dropdown(
                id='radar-select',
                options=[{'label': radar, 'value': radar} for radar in unique_radars],
                value='全部'
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    html.Div(id='output', children=[
        dcc.Graph(id='result-table'),
        dcc.Graph(id='gantt-chart')
    ])
])

# ----------------------------
# 回调函数：更新图表
# ----------------------------
@app.callback(
    [Output('result-table', 'figure'),
     Output('gantt-chart', 'figure')],
    [Input('target-select', 'value'),
     Input('radar-select', 'value')]
)
def update_output(selected_target, selected_radar):
    df = build_schedule_data()
    if df.empty:
        empty_fig = go.Figure(layout=go.Layout(title="无可用数据"))
        return empty_fig, empty_fig

    # 筛选数据
    if selected_target != '全部':
        df = df[df['目标'] == selected_target]
    if selected_radar != '全部':
        df = df[df['雷达'] == selected_radar]

    df["目标"] = df["目标"].astype(str)  # 关键：防止编号被压缩

    # 构建表格
    table_fig = go.Figure(data=[
        go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=[df[col] for col in df.columns])
        )
    ])

    # 构建甘特图
    target_order = sorted(df["目标"].unique(), reverse=True)
    gantt_fig = px.timeline(
        df,
        x_start="开始时间",
        x_end="结束时间",
        y="目标",
        color="雷达",
        title="观测任务甘特图"
    )

    gantt_fig.update_yaxes(
        autorange="reversed",
        tickmode='array',
        tickvals=target_order,
        ticktext=target_order
    )

    gantt_fig.update_layout(
        xaxis_title="时间（UTC）",
        yaxis_title="目标编号",
        title_x=0.5,
        height=max(600, 40 * len(target_order)),
        margin=dict(l=60, r=40, t=60, b=40)
    )

    return table_fig, gantt_fig

# ----------------------------
# 启动应用
# ----------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=25526, debug=False)
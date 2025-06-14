import dash
from dash import dcc, html, Input, Output, callback_context, no_update
from dash.dcc import Download
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# 从 main.py 导入变量
try:
    from data_processing_module import require_data, sensor_data, radar_target_vis_dict, index_to_utc, radars, targets, arc_indices, x
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
                            "radar": radar_name,
                            "target": target_name,
                            "arc_index": a,
                            "start": index_to_utc(window[0]),
                            "end": index_to_utc(window[1]),
                            "duration_min": window[2],
                        })
                except Exception as e:
                    print(f"解析弧段 ({r}, {s}, {a}) 时出错: {e}")
    return pd.DataFrame(schedule)

# 获取所有雷达和目标编号（用于下拉框显示中文）
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

    # 导出按钮
    html.Div([
        html.Button("导出 CSV", id="export-csv-btn", n_clicks=0, style={'margin': '10px'}),
        Download(id="download-csv")
    ], style={'textAlign': 'center'}),

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
        df = df[df['target'] == selected_target]
    if selected_radar != '全部':
        df = df[df['radar'] == selected_radar]

    df["target"] = df["target"].astype(str)  # 关键：防止编号被压缩

    # 构建表格（使用英文列名）
    table_fig = go.Figure(data=[
        go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=[df[col] for col in df.columns])
        )
    ])

    # 构建甘特图（y轴为 target，颜色为 radar）
    target_order = sorted(df["target"].unique(), reverse=True)
    gantt_fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="target",
        color="radar",
        hover_data=["duration_min", "arc_index"],
        title="Observation Task Gantt Chart"
    )

    gantt_fig.update_yaxes(
        autorange="reversed",
        tickmode='array',
        tickvals=target_order,
        ticktext=target_order
    )

    gantt_fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Target ID",
        title_x=0.5,
        height=max(600, 40 * len(target_order)),
        margin=dict(l=60, r=40, t=60, b=40)
    )

    return table_fig, gantt_fig


# ----------------------------
# 回调函数：导出 CSV 文件
# ----------------------------
@app.callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    [Input('target-select', 'value'),
     Input('radar-select', 'value')],
    prevent_initial_call=True
)
def export_csv(n_clicks, selected_target, selected_radar):
    if n_clicks <= 0:
        return no_update

    df = build_schedule_data()

    # 筛选目标
    if selected_target != '全部':
        df = df[df['target'] == selected_target]

    # 筛选雷达
    if selected_radar != '全部':
        df = df[df['radar'] == selected_radar]

    if df.empty:
        return no_update

    return dict(content=df.to_csv(index=False), filename="schedule_tasks.csv")


# 启动本地服务器运行 Dash 应用
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=25526, debug=False)
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io
import base64

# ----------------------------
# 数据导入
# ----------------------------
try:
    from main import require_data, sensor_data, radar_target_vis_dict, index_to_utc, radars, targets, arc_indices, x
except ImportError as e:
    raise ImportError("请确保 main.py 中已定义并导出以下变量：radars, targets, arc_indices, x") from e

# ----------------------------
# 初始化应用
# ----------------------------
app = dash.Dash(__name__)
app.title = "雷达目标调度可视化"

# ----------------------------
# 构建调度数据
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

# ----------------------------
# 获取选项
# ----------------------------
unique_radars = ['全部'] + [sensor_data.iloc[r]["雷达编号"] for r in radars]
unique_targets = ['全部'] + list(require_data["目标编号"].unique())

# ----------------------------
# 页面布局
# ----------------------------
app.layout = html.Div([
    html.H2("雷达目标观测调度方案结果展示", style={
        'textAlign': 'center',
        'marginTop': '20px',
        'fontFamily': 'Microsoft YaHei'
    }),

    html.Div([
        html.Div([
            html.Label("选择目标编号：", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='target-select',
                options=[{'label': t, 'value': t} for t in unique_targets],
                value='全部',
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("选择雷达编号：", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='radar-select',
                options=[{'label': r, 'value': r} for r in unique_radars],
                value='全部',
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'marginBottom': '10px'}),

    html.Div([
        html.Button("导出调度结果 CSV", id='download-btn', n_clicks=0, style={
            'backgroundColor': '#007ACC',
            'color': 'white',
            'border': 'none',
            'padding': '10px 20px',
            'fontSize': '14px',
            'marginBottom': '20px',
            'cursor': 'pointer'
        }),
        dcc.Download(id="download-dataframe-csv")
    ], style={'textAlign': 'right'}),

    dcc.Graph(id='result-table'),

    html.Div([
        dcc.Graph(id='gantt-chart')
    ], style={'marginTop': '30px'})
], style={'padding': '20px', 'fontFamily': 'Microsoft YaHei'})

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
        empty_fig = go.Figure(layout=go.Layout(title="无可用数据", title_x=0.5))
        return empty_fig, empty_fig

    if selected_target != '全部':
        df = df[df['目标'] == selected_target]
    if selected_radar != '全部':
        df = df[df['雷达'] == selected_radar]

    if df.empty:
        empty_fig = go.Figure(layout=go.Layout(title="筛选条件下无数据", title_x=0.5))
        return empty_fig, empty_fig

    # 表格图
    table_fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='lightblue',
                align='center',
                font=dict(color='black', size=12)
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color='lavender',
                align='center',
                font=dict(size=11)
            )
        )
    ])
    table_fig.update_layout(title="调度方案表格", title_x=0.5)

    # 甘特图
    gantt_fig = px.timeline(
    df, x_start="开始时间", x_end="结束时间", y="目标", color="雷达",
    title="观测任务甘特图"
    )

    gantt_fig.update_yaxes(
        autorange="reversed",
        tickformat='d'  # 禁止69k缩写
    )
    gantt_fig.update_layout(
        xaxis_title="时间（UTC）",
        yaxis_title="目标编号",
        title_x=0.5,
        height=max(600, 40 * len(df['目标'].unique())),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return table_fig, gantt_fig

# ----------------------------
# 回调函数：导出CSV文件
# ----------------------------
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("target-select", "value"),
    State("radar-select", "value"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, selected_target, selected_radar):
    df = build_schedule_data()
    if selected_target != '全部':
        df = df[df['目标'] == selected_target]
    if selected_radar != '全部':
        df = df[df['雷达'] == selected_radar]
    return dcc.send_data_frame(df.to_csv, filename="调度结果.csv", index=False)

# ----------------------------
# 启动服务
# ----------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=25526, debug=False)

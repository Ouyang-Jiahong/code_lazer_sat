import dash
from dash import dcc, html, Input, Output, callback_context, no_update
from dash.dcc import Download  # 用于下载文件
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from data import radar_target_vis_dict, simDate


def index_to_utc(idx):
    """将 simDate 中的时间索引转为 ISO 格式的 UTC 时间字符串"""
    t = simDate[:, idx]
    return f"{int(t[0])}-{int(t[1]):02d}-{int(t[2]):02d}T{int(t[3]):02d}:{int(t[4]):02d}:{int(t[5]):02d}"


# 构造可视化数据表：遍历每一个 (雷达, 目标) 的可见弧段列表，生成记录
records = []
for (r, s), arc_list in radar_target_vis_dict.items():
    for a_idx, (s_idx, e_idx, duration) in enumerate(arc_list):
        start_time = index_to_utc(s_idx)
        end_time = index_to_utc(e_idx)

        records.append({
            "radar": r,
            "target": s,
            "arc_index": a_idx,
            "duration_min": duration,
            "start": start_time,
            "end": end_time
        })

# 将所有记录构建为 DataFrame
df_arcs = pd.DataFrame(records)

# 获取唯一的目标和雷达编号，用于下拉框选项
unique_targets = sorted(df_arcs['target'].unique())
unique_radars = sorted(df_arcs['radar'].unique())

# 添加 '全部' 选项
dropdown_targets = [{'label': '全部目标', 'value': 'all'}] + \
                   [{'label': f"目标 {s}", 'value': s} for s in unique_targets]

dropdown_radars = [{'label': '全部雷达', 'value': 'all'}] + \
                  [{'label': f"雷达 {r}", 'value': r} for r in unique_radars]

# 启动 Dash 应用
app = dash.Dash(__name__)
app.title = "可见弧段可视化"

# ----------------------------
# 页面布局
# ----------------------------
app.layout = html.Div([
    html.H2("目标任务可见弧段可视化工具", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label("选择目标编号："),
            dcc.Dropdown(
                id='target-select',
                options=dropdown_targets,
                value='all'
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("选择雷达编号："),
            dcc.Dropdown(
                id='radar-select',
                options=dropdown_radars,
                value='all'
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
    # 复制原始数据以避免修改原数据
    df = df_arcs.copy()

    # 筛选目标
    if selected_target != 'all':
        df = df[df['target'] == selected_target]

    # 筛选雷达
    if selected_radar != 'all':
        df = df[df['radar'] == selected_radar]

    # 如果筛选后无数据
    if df.empty:
        empty_fig = go.Figure(layout=go.Layout(title="无可用数据"))
        return empty_fig, empty_fig

    # ----------------------------
    # 构建表格
    # ----------------------------
    table_fig = go.Figure(data=[
        go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=[df[col] for col in df.columns])
        )
    ])

    # ----------------------------
    # 构建甘特图
    # ----------------------------
    if selected_target != 'all' and selected_radar != 'all':
        y_col = "arc_index"
        title = f"目标 {selected_target}, 雷达 {selected_radar} 可见弧段时序图"
        yaxis_title = "弧段编号"
    else:
        y_col = "radar" if selected_target != 'all' else "target"
        title = "多目标/雷达可见弧段时序图"
        yaxis_title = "雷达编号" if selected_target != 'all' else "目标编号"

    # 构建甘特图
    gantt_fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y=y_col,
        color="radar" if selected_target != 'all' else "target",
        hover_data=["duration_min", "arc_index"]
    )

    # 获取 Y 轴顺序并倒序显示
    y_order = sorted(df[y_col].unique(), key=lambda x: int(x) if isinstance(x, (int, float)) or str(x).replace('.', '', 1).isdigit() else x,
                     reverse=True)

    gantt_fig.update_yaxes(
        autorange="reversed",
        tickmode='array',
        tickvals=y_order,
        ticktext=[str(int(float(y))) if isinstance(y, (int, float)) or str(y).replace('.', '', 1).isdigit() else y for y
                  in y_order],
        title=yaxis_title
    )

    gantt_fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="时间（UTC）",
        yaxis_title=yaxis_title,
        height=max(600, 40 * len(y_order)),
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

    # 复制原始数据以避免修改原数据
    df = df_arcs.copy()

    # 筛选目标
    if selected_target != 'all':
        df = df[df['target'] == selected_target]

    # 筛选雷达
    if selected_radar != 'all':
        df = df[df['radar'] == selected_radar]

    # 如果筛选后无数据
    if df.empty:
        return no_update

    # 返回 CSV 下载
    return dict(content=df.to_csv(index=False), filename="visible_arcs.csv")


# 启动本地服务器运行 Dash 应用
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=25525, debug=False)
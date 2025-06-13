import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

from main import radar_target_vis_dict

# 构造可视化数据表
records = []
for (r, s), arc_list in radar_target_vis_dict.items():
    for a_idx, (start, end, duration) in enumerate(arc_list):
        records.append({
            "radar": r,
            "target": s,
            "arc_index": a_idx,
            "start": start.utc.datetime,
            "end": end.utc.datetime,
            "duration_min": duration
        })

df_arcs = pd.DataFrame(records)

# 启动 Dash 应用
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("目标任务可见弧段可视化工具"),
    html.Label("选择目标编号："),
    dcc.Dropdown(
        id='target-dropdown',
        options=[{'label': f"目标 {s}", 'value': s} for s in sorted(df_arcs['target'].unique())],
        value=sorted(df_arcs['target'].unique())[0]
    ),
    dcc.Graph(id='arc-plot')
])


@app.callback(
    Output('arc-plot', 'figure'),
    Input('target-dropdown', 'value')
)
def update_arc_plot(selected_target):
    df = df_arcs[df_arcs['target'] == selected_target]

    if df.empty:
        return px.scatter(title=f"目标 {selected_target} 无可用弧段")

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="radar",
        color="radar",
        hover_data=["duration_min", "arc_index"],
        title=f"目标 {selected_target} 可见弧段时序图"
    )
    fig.update_yaxes(autorange="reversed", title="测站编号")
    fig.update_layout(
        height=600,
        xaxis_title="时间（UTC）",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)

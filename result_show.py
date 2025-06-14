import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from main import df

# 启动 Dash 应用
app = dash.Dash(__name__)

# 定义前端页面布局：包含标题和表格区域
app.layout = html.Div([
    html.H2("雷达目标观测调度方案结果展示"),  # 页面主标题
    dcc.Graph(id='result-table', figure=go.Figure(data=[go.Table(header=dict(values=list(df.columns)),
                 cells=dict(values=[df[col] for col in df.columns]))]))
])

# 启动本地服务器运行 Dash 应用（调试模式）
if __name__ == '__main__':
    app.run(
        host='127.0.0.1',     # 设置主机地址（默认也是这个）
        port=25526,           # 设置自定义端口
        debug=False           # 关闭调试模式
    )
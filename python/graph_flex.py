import plotly.graph_objs as go
import plotly.express as px  # plotly에서 다양한 색상 팔레트를 제공
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
def to_rgba(color, alpha=0.7):
    if color.startswith('#'):
        # Convert #RRGGBB to rgba
        return f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})"
    elif color.startswith('rgb('):
        # Convert rgb(R, G, B) to rgba
        return color.replace('rgb', 'rgba').replace(')', f', {alpha})')
    else:
        raise ValueError(f"Unsupported color format: {color}")
colors = px.colors.qualitative.T10
# colors = [to_rgba(color) for color in colorss]


def flex_graph(df, FLEX_SELECT, selected_chart_names):
    color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(selected_chart_names)}
    FLEX_GRAPH = FLEX_SELECT.copy()
    FLEX_GRAPH["short_name"] = FLEX_GRAPH["chart_name"].apply(lambda x: f"{x.split('|')[0].strip()} | {x.split('|')[1].strip()}")

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Torque (N/mm) - Angle (degree)", "Torque", "Energy Return"))


    # 첫 번째 플롯: Scatter 플롯 (Torque vs. Angle)
    for idx, key in enumerate(selected_chart_names):
        x_data = df[key][0]  # x축 (Angle)
        y_data = df[key][1]  # y축 (Torque)
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines+markers",
                name=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 모델명만 표시
                line=dict(width=2, dash="dot", color=color_map[key]),
                marker=dict(size=3),
                legendgroup = f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}"

            ),
            row=1,
            col=1
        )

    # 빨간색 점선 추가 (y=0 기준선)
    fig.add_shape(
        type="line",
        x0=0,  # x축 시작점
        x1=45,  # x축 끝점
        y0=0,  # y축 값 (고정)
        y1=0,  # y축 값 (고정)
        line=dict(color="red", width=2, dash="dot"),  # 빨간색 점선 스타일
        row=1,
        col=1
    )
    
    for key in selected_chart_names:
        fig.add_trace(
            go.Bar(
                x=FLEX_GRAPH[FLEX_GRAPH["chart_name"] == key]["short_name"],
                y=FLEX_GRAPH[FLEX_GRAPH["chart_name"] == key]["Torque (N/mm)"],
                name=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 범례에 모델명만 표시
                legendgroup=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 동일한 모델은 같은 legendgroup
                marker_color=color_map[key],
                text=FLEX_GRAPH[FLEX_GRAPH["chart_name"] == key]["Torque (N/mm)"],  # 데이터 값 표시
                showlegend=False  # 이미 첫 번째 플롯에서 범례 표시
            ),
            row=1,
            col=2
        )

    # 세 번째 플롯: Bar 플롯 (Energy Return 비교)
    for key in selected_chart_names:    
        fig.add_trace(
            go.Bar(
                x=FLEX_GRAPH[FLEX_GRAPH["chart_name"] == key]["short_name"],
                y=FLEX_GRAPH[FLEX_GRAPH["chart_name"] == key]["Energy Return"],
                name=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 범례에 모델명만 표시
                legendgroup=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 동일한 모델은 같은 legendgroup
                marker_color=color_map[key],
                text=FLEX_GRAPH[FLEX_GRAPH["chart_name"] == key]["Energy Return"],  # 데이터 값 표시
                showlegend=False  # 이미 첫 번째 플롯에서 범례 표시
            ),
            row=1,
            col=3
        )

    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="",
            x=0.5,  # 제목 가운데 정렬
            xanchor="center"
        ),
        template="plotly_white",
        height=500,
        width=1200,
        showlegend=True,  # 범례 표시 제거 (필요시 True로 설정)
        legend=dict(
            orientation="h",  
            yanchor="bottom",    
            y=-0.5,           
            xanchor="center",
            x=0.5,
            font=dict(
                family="Calibri",
                size=12,
                color="black"
            ),
            bgcolor="white",
            bordercolor="black",
            borderwidth=2
        ),
    )

    # 각 플롯의 축 제목 설정
    fig.update_xaxes(title_text="Angle (degree)", row=1, col=1)
    fig.update_yaxes(title_text="Torque (N/mm)", row=1, col=1)
    fig.update_yaxes(title_text="Torque (N/mm)", row=1, col=2)
    fig.update_yaxes(title_text="Energy Return (%)", row=1, col=3)
    
    return fig
        
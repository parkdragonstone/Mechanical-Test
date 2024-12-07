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

def static_friction_graph(df, selected_chart_names):
    color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(selected_chart_names)}
    TRACTION_GRAPH = df.copy()
    TRACTION_GRAPH["short_name"] = TRACTION_GRAPH["chart_name"].apply(lambda x: f"{x.split('|')[0].strip()} | {x.split('|')[1].strip()}")

    # 서브플롯 생성
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("FRONT", "LATERAL", "ROTATION")
    )

    for key in selected_chart_names:

        # Front P 데이터 (row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['short_name'],
                y=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['front_p_mean'],
                name=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",
                marker_color=color_map[key],
                legendgroup=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 동일한 모델은 같은 legendgroup
                text = f"{TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['front_p_mean'].values[0]} ({TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['front_p_std'].values[0]})",
                # textposition="outside",  # 데이터 값을 바 위에 표시
                # error_y=dict(
                #     type='data',
                #     array=TRACTION_GRAPH['front_p_std'],  # 표준편차
                #     visible=True
                # ),
                showlegend=True
            ),
            row=1,
            col=1
        )

        # Lateral P 데이터 (row=1, col=2)
        fig.add_trace(
            go.Bar(
                x=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['short_name'],
                y=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['lateral_p_mean'],
                name=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",
                marker_color=color_map[key],
                legendgroup=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 동일한 모델은 같은 legendgroup
                text = f"{TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['lateral_p_mean'].values[0]} ({TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['lateral_p_std'].values[0]})",
                # textposition="outside",  # 데이터 값을 바 위에 표시
                # error_y=dict(
                #     type='data',
                #     array=TRACTION_GRAPH['lateral_p_std'],  # 표준편차
                #     visible=True
                # ),
                showlegend=False
            ),
            row=1,
            col=2
        )

        # Rotation P 데이터 (row=1, col=3)
        fig.add_trace(
            go.Bar(
                x=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['short_name'],
                y=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['rotation_p_mean'],
                name=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",
                marker_color=color_map[key],
                legendgroup=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 동일한 모델은 같은 legendgroup
                text = f"{TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['rotation_p_mean'].values[0]} ({TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['rotation_p_std'].values[0]})",
                # textposition="outside",  # 데이터 값을 바 위에 표시
                # error_y=dict(
                #     type='data',
                #     array=TRACTION_GRAPH['rotation_p_std'],  # 표준편차
                #     visible=True
                # ),
                showlegend=False
            ),
            row=1,
            col=3
        )

    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="STATIC FRICTION",
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
            y=-0.4,           
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
    fig.update_yaxes(title_text="Friction (N)", row=1, col=1)
    fig.update_yaxes(title_text="Rotation (N/m)", row=1, col=3)

    return fig


def kinetic_friction_graph(df, selected_chart_names):
    color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(selected_chart_names)}
    TRACTION_GRAPH = df.copy()
    TRACTION_GRAPH["short_name"] = TRACTION_GRAPH["chart_name"].apply(lambda x: f"{x.split('|')[0].strip()} | {x.split('|')[1].strip()}")

    # 서브플롯 생성
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("FRONT", "LATERAL")
    )

    for key in selected_chart_names:

        # Front P 데이터 (row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['short_name'],
                y=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['front_k_mean'],
                name=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",
                marker_color=color_map[key],
                legendgroup=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 동일한 모델은 같은 legendgroup
                text = f"{TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['front_k_mean'].values[0]} ({TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['front_k_std'].values[0]})",
                # error_y=dict(
                #     type='data',
                #     array=TRACTION_GRAPH['front_k_std'],  # 표준편차
                #     visible=True
                # ),
                showlegend=True
            ),
            row=1,
            col=1
        )

        # Lateral P 데이터 (row=1, col=2)
        fig.add_trace(
            go.Bar(
                x=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['short_name'],
                y=TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['lateral_k_mean'],
                name=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",
                marker_color=color_map[key],
                legendgroup=f"{key.split('|')[0].strip()} | {key.split('|')[1].strip()}",  # 동일한 모델은 같은 legendgroup
                text = f"{TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['lateral_k_mean'].values[0]} ({TRACTION_GRAPH[TRACTION_GRAPH['chart_name'] == key]['lateral_k_std'].values[0]})",
                # error_y=dict(
                #     type='data',
                #     array=TRACTION_GRAPH['lateral_k_std'],  # 표준편차
                #     visible=True
                # ),
                showlegend=False
            ),
            row=1,
            col=2
        )


    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="KINETIC FRICTION",
            x=0.5,  # 제목 가운데 정렬
            xanchor="center"
        ),
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True,  # 범례 표시 제거 (필요시 True로 설정)
        legend=dict(
            orientation="h",  
            yanchor="bottom",    
            y=-0.4,           
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
    fig.update_yaxes(title_text="Friction (N)", row=1, col=1)


    return fig
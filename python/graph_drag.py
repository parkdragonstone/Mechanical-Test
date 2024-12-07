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

def drag_graph(DRAG, selected_chart_names, title):
    df = DRAG[DRAG['chart_name'].isin(selected_chart_names)].copy()
    # TEST 컬럼 추출
    test_columns = [col for col in df.columns if col.startswith("TEST")]

    if title == 'Drag Test (Set Zero)':
        # TEST0000 값을 기준으로 모든 TEST 값에서 차이값 계산
        df_difference = df[test_columns].subtract(df["TEST0000"], axis=0)
        df_diff = df.copy()
        df_diff[test_columns] = df_difference
        df = df_diff
    
    elif title == 'Drag Teset (Differential)':
        df[test_columns] = -df[test_columns].diff(axis=1)
    
    # x축 생성 (숫자 변환)
    x_values = [int(col.replace("TEST", "")) for col in test_columns]

    # Plotly 스캐터 플롯 생성
    fig = go.Figure()

    # 각 모델에 대해 스캐터 추가
    for idx, (index,row) in enumerate(df.iterrows()):
        fig.add_trace(go.Scatter(
            x=x_values,
            y=row[test_columns],  # TEST 데이터 값
            mode='lines+markers',
            name=f"{row['chart_name'].split('|')[0].strip()} | {row['chart_name'].split('|')[1].strip()}",  # 날짜 제외
            marker=dict(size=8, color=colors[idx % len(colors)]),  # 고유 색상 적용
            line=dict(width=2, dash="solid", color=colors[idx % len(colors)])  # 고유 색상 적용
        ))

    # x축 설정: 데이터의 최소값부터 최대값으로 설정
    fig.update_xaxes(
        range=[min(x_values)-30, max(x_values)+30],
        tickmode="linear",
        tick0=min(x_values),
        dtick=50,  # 50 단위로 설정
        title="Drag Trials (50 steps)"
    )

    # y축 설정: 데이터의 최소값과 최대값에 맞게 조정
    fig.update_yaxes(
        # range=[df[test_columns].min().min() - 40, df[test_columns].max().max() + 40],  # 데이터 범위에 여유 추가
        title="Shoe & Last Mass (g)",
        showgrid=False  # y축의 그리드 제거
    )

    # Legend 설정: 아래에 표시, 검은색 테두리 추가
    fig.update_layout(
        title=dict(
            text=title, # DRAG Test (Real Mass)
            x=0.5,  # 제목 가운데 정렬
            xanchor="center"
        ),
        template="plotly_white",
        width=1200,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,  # 아래로 위치
            xanchor="center",
            x=0.5,
            bgcolor="white",  # 흰색 배경
            bordercolor="black",  # 검은색 테두리
            borderwidth=1
        )
    )

    # 그래프 표시
    return fig
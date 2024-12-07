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

def slope_graph(all_filtered_data, selected_models, slope_col, title_1, title_2, xlabel):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>{title_1}</b>", f"<b>{title_2}</b>"))
        # 첫 번째 서브플롯 - ForeFoot 데이터
        
    for i in range(len(selected_models)):
        ff = all_filtered_data[selected_models[i]]['ff_series'][slope_col].values[0]
        
        # NaN 처리: NaN이 아닌 데이터에 대해서만 막대를 생성
        if not np.isnan(ff):
            fig.add_trace(go.Bar(
                x=[ff],  # Force 값
                y=[f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}"],  # Brand | Model 문자열
                orientation='h',  # 수평 막대
                text=f"{ff:.2f}",
                textposition='auto',
                marker_color=colors[i % len(colors)],  # 막대 색
                name = f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}",
                showlegend=True,
                legendgroup = f"group_{i}",
            ),
            row=1, col=1)

    
    for i in range(len(selected_models)):
        
        if all_filtered_data[selected_models[i]]['rf_series'].empty:
            continue
        
        rf = all_filtered_data[selected_models[i]]['rf_series'][slope_col].values[0]
        
        # NaN 처리: NaN이 아닌 데이터에 대해서만 막대를 생성
        if not np.isnan(rf):
            fig.add_trace(go.Bar(
                x=[rf],  # Force 값
                y=[f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}"],  # Brand | Model 문자열
                orientation='h',  # 수평 막대
                text=f"{rf:.2f}",  # 텍스트: 평균 (표준편차)
                textposition='auto',
                marker_color=colors[i % len(colors)],  # 막대 색
                name = f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}",
                showlegend=False,
                legendgroup = f"group_{i}"
            ),
            row=1, col=2)
            
    # 레이아웃 설정
    fig.update_layout(
        # title={
        #     'text': title,
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'
        # },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Calibri", size=14),
        yaxis=dict(automargin=True),  # Y축 여백 자동 조정
        legend=dict(
            orientation="h",  
            yanchor="bottom",    
            y=-0.7,           
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
        showlegend=True,
        width=1200,
        height=400,
        margin=dict(l=0, r=0, b=100, t=50)
    )

    # 축 이름 설정
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=1, col=2)
    # 두 번째 y축 이름 제거
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2)
    
    return fig

def for_dis_rel_graph(filterd_data, selected_models, title_1, title_2, xlabel, ylabel, selected_shoe_type):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>{title_1}</b>", f"<b>{title_2}</b>"))

    if selected_shoe_type == 'TENNIS':
        rf_ds1_s, rf_ds1_e, rf_ds2_s, rf_ds2_e = 250, 400, 600, 750
        ff_ds1_s, ff_ds1_e, ff_ds2_s, ff_ds2_e = 400, 600, 900, 1100
            
    else:
        rf_ds1_s, rf_ds1_e, rf_ds2_s, rf_ds2_e = 350, 500, 750, 900
        ff_ds1_s, ff_ds1_e, ff_ds2_s, ff_ds2_e = 400, 600, 900, 1100
    
    ff_x_max = 0  
    ff_x_min = 0  
    # 선택된 각 모델에 대해 데이터 불러오기
    for i, model in enumerate(selected_models):
        model_data = filterd_data.get(model, {})

        # RF Series 데이터가 있는 경우 그래프에 추가
        if not model_data.get('ff_series', pd.DataFrame()).empty:
            rf_series_data = model_data['ff_series']
            if ff_x_max < rf_series_data.max()['Disp (mm)']:
                ff_x_max = rf_series_data.max()['Disp (mm)']
            if ff_x_min > rf_series_data.min()['Disp (mm)']:
                ff_x_min = rf_series_data.min()['Disp (mm)']
                
            trace_rf = go.Scatter(
                x=rf_series_data['Disp (mm)'], 
                y=rf_series_data['Force (N)'], 
                mode='lines+markers', 
                name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                line=dict(color=colors[i % len(colors)], width=2, dash='dot'),  # 색상 리스트에서 순차적으로 선택
                showlegend=True,
                legendgroup = f"group_{i}"
            )
            fig.add_trace(trace_rf, row = 1, col = 1)
        

            # ds1, ds2 구간에서 올라가는 부분의 끝 지점과 내려가는 부분의 끝 지점 좌표
            ds1_up_f1 = rf_series_data['ds1_upward_f1'].iloc[0]
            ds1_up_f2 = rf_series_data['ds1_upward_f2'].iloc[0]
            ds1_up_d1 = rf_series_data['ds1_upward_d1'].iloc[0]
            ds1_up_d2 = rf_series_data['ds1_upward_d2'].iloc[0]

            ds2_up_f1 = rf_series_data['ds2_upward_f1'].iloc[0]
            ds2_up_f2 = rf_series_data['ds2_upward_f2'].iloc[0]
            ds2_up_d1 = rf_series_data['ds2_upward_d1'].iloc[0]
            ds2_up_d2 = rf_series_data['ds2_upward_d2'].iloc[0]
            
            # ds1 구간 점선
            fig.add_trace(
                go.Scatter(
                    x=[ds1_up_d1, ds1_up_d2],
                    y=[ds1_up_f1, ds1_up_f2],
                    mode="lines",
                    name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                    line=dict(color='red', width=4),
                    showlegend=False,
                    legendgroup=f"group_{i}"
                ),
                row=1, col=1
            )

            # ds2 구간 점선
            fig.add_trace(
                go.Scatter(
                    x=[ds2_up_d1, ds2_up_d2],
                    y=[ds2_up_f1, ds2_up_f2],
                    mode="lines",
                    line=dict(color='red', width=4),
                    name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                    showlegend=False,
                    legendgroup=f"group_{i}"
                ),
                row=1, col=1
            )
     
            
    # Shaded regions between 400-600 N and 900-1100 N
    fig.add_shape(
        type="rect",
        xref="x", yref="y",  # X-axis uses paper (full chart width), Y uses y-axis values
        x0=ff_x_min, x1=ff_x_max+2, y0=ff_ds1_s, y1=ff_ds1_e,
        fillcolor="rgba(255, 182, 193, 0.4)",  # Light red, semi-transparent
        line_width=0,
        layer="below",  # Place the shading behind the traces,
        row=1, col=1
    )

    fig.add_shape(
        type="rect",
        xref="x", yref="y",
        x0=ff_x_min, x1=ff_x_max+2, y0=ff_ds2_s, y1=ff_ds2_e,
        fillcolor="rgba(255, 182, 193, 0.4)",  # Light red, semi-transparent
        line_width=0,
        layer="below",
        row=1, col=1
    )

    rf_x_max = 0
    rf_x_min = 0
    # 선택된 각 모델에 대해 데이터 불러오기
    for i, model in enumerate(selected_models):
        model_data = filterd_data.get(model, {})

        # RF Series 데이터가 있는 경우 그래프에 추가
        if not model_data.get('rf_series', pd.DataFrame()).empty:
            rf_series_data = model_data['rf_series']
            if rf_x_max < rf_series_data.max()['Disp (mm)']:
                rf_x_max = rf_series_data.max()['Disp (mm)']
            if rf_x_min > rf_series_data.min()['Disp (mm)']:
                rf_x_min = rf_series_data.min()['Disp (mm)']
                
            trace_rf = go.Scatter(
                x=rf_series_data['Disp (mm)'], 
                y=rf_series_data['Force (N)'], 
                mode='lines+markers', 
                name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                line=dict(color=colors[i % len(colors)], width=2),  # 색상 리스트에서 순차적으로 선택
                showlegend=False,
                legendgroup = f"group_{i}"
            )
            fig.add_trace(trace_rf, row = 1, col = 2)

            # ds1, ds2 구간에서 올라가는 부분의 끝 지점과 내려가는 부분의 끝 지점 좌표
            ds1_up_f1 = rf_series_data['ds1_upward_f1'].iloc[0]
            ds1_up_f2 = rf_series_data['ds1_upward_f2'].iloc[0]
            ds1_up_d1 = rf_series_data['ds1_upward_d1'].iloc[0]
            ds1_up_d2 = rf_series_data['ds1_upward_d2'].iloc[0]

            ds2_up_f1 = rf_series_data['ds2_upward_f1'].iloc[0]
            ds2_up_f2 = rf_series_data['ds2_upward_f2'].iloc[0]
            ds2_up_d1 = rf_series_data['ds2_upward_d1'].iloc[0]
            ds2_up_d2 = rf_series_data['ds2_upward_d2'].iloc[0]

            
            # ds1 구간 점선
            fig.add_trace(
                go.Scatter(
                    x=[ds1_up_d1, ds1_up_d2],
                    y=[ds1_up_f1, ds1_up_f2],
                    mode="lines",
                    name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                    line=dict(color='red', width=4),
                    showlegend=False,
                    legendgroup=f"group_{i}"
                ),
                row=1, col=2
            )

            # ds2 구간 점선
            fig.add_trace(
                go.Scatter(
                    x=[ds2_up_d1, ds2_up_d2],
                    y=[ds2_up_f1, ds2_up_f2],
                    mode="lines",
                    line=dict(color='red', width=4),
                    name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                    showlegend=False,
                    legendgroup=f"group_{i}"
                ),
                row=1, col=2
            )
   
    # Shaded regions between 400-600 N and 900-1100 N
    fig.add_shape(
        type="rect",
        xref="x", yref="y",  # X-axis uses paper (full chart width), Y uses y-axis values
        x0=rf_x_min, x1=rf_x_max+2, y0=rf_ds1_s, y1=rf_ds1_e,
        fillcolor="rgba(255, 182, 193, 0.4)",  # Light red, semi-transparent
        line_width=0,
        layer="below",  # Place the shading behind the traces,
        row=1, col=2
    )

    fig.add_shape(
        type="rect",
        xref="x", yref="y",
        x0=rf_x_min, x1=rf_x_max+2, y0=rf_ds2_s, y1=rf_ds2_e,
        fillcolor="rgba(255, 182, 193, 0.4)",  # Light red, semi-transparent
        line_width=0,
        layer="below",
        row=1, col=2
    )

    # 배경색을 하얀색으로 설정하고, legend 위치와 제목 중앙 정렬
    fig.update_layout(
        # title={
        #     'text': title,
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'
        # },       
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=500,
        xaxis=dict(
            # range=[0, 25], 
            showgrid=True, 
            gridcolor='lightgray'),
        yaxis=dict(
            # range=[0, 3000], 
            showgrid=True, 
            gridcolor='lightgray'),
        legend=dict(
            orientation="h",  
            yanchor="bottom",    
            y=-0.48,           
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
        margin=dict(l=0, r=0, b=100, t=50)
    )

    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_yaxes(title_text=ylabel, row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text=xlabel, row=1, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(row=1, col=2, showgrid=True, gridcolor='lightgray')
    return fig

def bar_graph(FF, FF_SD, RF, RF_SD, col, title_1, title_2, xlabel, selected_models):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>{title_1}</b>", f"<b>{title_2}</b>"))
        # 첫 번째 서브플롯 - ForeFoot 데이터

    if col == 'Thickness':
        col = 'Thickness ForeFoot (mm)'
        
    for i in range(len(selected_models)):
        mean = FF[FF['chart_name'] == selected_models[i]][col].values[0]
        std = FF_SD[FF_SD['chart_name'] == selected_models[i]][col].values[0]
        
        # NaN 처리: NaN이 아닌 데이터에 대해서만 막대를 생성
        if not np.isnan(mean):
            fig.add_trace(go.Bar(
                x=[mean],  # Force 값
                y=[f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}"],  # Brand | Model 문자열
                orientation='h',  # 수평 막대
                text=f"{mean:.2f} ({std:.2f})" if not np.isnan(std) else f"{mean:.2f} (N/A)",  # 텍스트: 평균 (표준편차)
                textposition='auto',
                marker_color=colors[i % len(colors)],  # 막대 색
                name = f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}",
                showlegend=True,
                legendgroup = f"group_{i}"
            ),
            row=1, col=1)

    if col == 'Thickness ForeFoot (mm)':
        col = 'Thickness RearFoot (mm)'
        
    for i in range(len(selected_models)):
        mean = RF[RF['chart_name'] == selected_models[i]][col].values[0]
        std = RF_SD[RF_SD['chart_name'] == selected_models[i]][col].values[0]
        
        # NaN 처리: NaN이 아닌 데이터에 대해서만 막대를 생성
        if not np.isnan(mean):
            fig.add_trace(go.Bar(
                x=[mean],  # Force 값
                y=[f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}"],  # Brand | Model 문자열
                orientation='h',  # 수평 막대
                text=f"{mean:.2f} ({std:.2f})" if not np.isnan(std) else f"{mean:.2f} (N/A)",  # 텍스트: 평균 (표준편차)
                textposition='auto',
                marker_color=colors[i % len(colors)],  # 막대 색
                name = f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}",
                showlegend=False,
                legendgroup = f"group_{i}"
            ),
            row=1, col=2)
            
    # 레이아웃 설정
    fig.update_layout(
        # title={
        #     'text': title,
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'
        # },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Calibri", size=14),
        yaxis=dict(automargin=True),  # Y축 여백 자동 조정
        legend=dict(
            orientation="h",  
            yanchor="bottom",    
            y=-0.7,           
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
        showlegend=True,
        width=1200,
        height=400,
        margin=dict(l=0, r=0, b=100, t=50)
    )

    # 축 이름 설정
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=1, col=2)
    # 두 번째 y축 이름 제거
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2)
    
    
    return fig

def scatter_plotly(title_1, title_2, x, y, selected_chart_names, FILT_FF, FILT_RF, xlabel, ylabel):
    # 두 개의 서브플롯 생성
    if y == 'Thickness (mm)':
        y = 'Thickness ForeFoot (mm)'
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>{title_1}</b>", f"<b>{title_2}</b>"))

    # ForeFoot 데이터
    x_avg_ff = FILT_FF[x].mean()
    y_avg_ff = FILT_FF[y].mean()
    
    # 첫 번째 서브플롯 - ForeFoot 데이터
    fig.add_trace(
        go.Scatter(
            x=FILT_FF[x],
            y=FILT_FF[y],
            mode='markers',
            marker=dict(size=6, color='rgba(0, 0, 255, 0.1)'),  # 연한 파란색으로 설정
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 빨간색 선 추가 (ForeFoot)
    fig.add_trace(
        go.Scatter(
            x=[x_avg_ff, x_avg_ff],
            y=[FILT_FF[y].min(), FILT_FF[y].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[FILT_FF[x].min(), FILT_FF[x].max()],
            y=[y_avg_ff, y_avg_ff],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    # ForeFoot 텍스트 추가 (x축 평균)
    fig.add_annotation(
        text=f"AVG: {x_avg_ff:.2f}",
        x=x_avg_ff,
        y=FILT_FF[y].min(),
        xref="x1",
        yref="y1",
        showarrow=False,
        # textangle=-90,  # 텍스트 90도 회전
        font=dict(
            family="Calibri",
            size=10,
            color="black"
        ),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    # ForeFoot 텍스트 추가 (y축 평균)
    fig.add_annotation(
        text=f"AVG: {y_avg_ff:.2f}",
        x=FILT_FF[x].max(),
        y=y_avg_ff,
        xref="x1",
        yref="y1",
        showarrow=False,
        font=dict(
            family="Calibri",
            size=10,
            color="black"
        ),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # 선택된 모델에 대해 ForeFoot에 점 추가
    for i, select in enumerate(selected_chart_names):
        selected_data = FILT_FF[FILT_FF['chart_name'] == select]
        fig.add_trace(
            go.Scatter(
                x=selected_data[x],
                y=selected_data[y],
                mode='markers',
                marker=dict(size=12, color=colors[i % len(colors)]),  # 진한 색상
                name=f"{select.split('|')[0]} - {select.split('|')[1]}",
                showlegend=True,
                legendgroup = f"group_{i}"
            ),
            row=1, col=1
        )

    if y == 'Thickness ForeFoot (mm)':
        y = 'Thickness RearFoot (mm)'
    
    # RearFoot 데이터
    x_avg_rf = FILT_RF[x].mean()
    y_avg_rf = FILT_RF[y].mean()
    # 두 번째 서브플롯 - RearFoot 데이터
    fig.add_trace(
        go.Scatter(
            x=FILT_RF[x],
            y=FILT_RF[y],
            mode='markers',
            marker=dict(size=6, color='rgba(0, 0, 255, 0.1)'),  # 연한 파란색으로 설정
            showlegend=False,
        ),
        row=1, col=2
    )
    
    # 빨간색 선 추가 (RearFoot)
    fig.add_trace(
        go.Scatter(
            x=[x_avg_rf, x_avg_rf],
            y=[FILT_RF[y].min(), FILT_RF[y].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[FILT_RF[x].min(), FILT_RF[x].max()],
            y=[y_avg_rf, y_avg_rf],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )

    # RearFoot 텍스트 추가 (x축 평균)
    fig.add_annotation(
        text=f"AVG: {x_avg_rf:.2f}",
        x=x_avg_rf,
        y=FILT_RF[y].min(),
        xref="x2",
        yref="y2",
        showarrow=False,
        # textangle=-90,  # 텍스트 90도 회전
        font=dict(
            family="Calibri",
            size=10,
            color="black"
        ),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    # RearFoot 텍스트 추가 (y축 평균)
    fig.add_annotation(
        text=f"AVG: {y_avg_rf:.2f}",
        x=FILT_RF[x].max(),
        y=y_avg_rf,
        xref="x2",
        yref="y2",
        showarrow=False,
        font=dict(
            family="Calibri",
            size=10,
            color="black"
        ),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # 선택된 모델에 대해 RearFoot에 점 추가
    for i, select in enumerate(selected_chart_names):
        selected_data = FILT_RF[FILT_RF['chart_name'] == select]
        fig.add_trace(
            go.Scatter(
                x=selected_data[x],
                y=selected_data[y],
                mode='markers',
                marker=dict(size=12, color=colors[i % len(colors)]),  # 진한 색상
                name=f"{select.split('|')[0]} - {select.split('|')[1]}",
                showlegend=False,
                legendgroup = f"group_{i}"
            ),
            row=1, col=2
        )

    # 전체 레이아웃 설정
    fig.update_layout(
        # title="Force (N) - Energy Return (%)",
        showlegend=True,
        # autosize=True,
        width=1200,
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.7,
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
            margin=dict(l=0, r=0, b=100, t=50)
    )

    # 축 이름 설정
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=1, col=2)
    
    return fig

def scatter__specimen_plotly(title_1, title_2, x, y1, y2, selected_chart_names, FILT_FF, xlabel, ylabel1, ylabel2):
    # 두 개의 서브플롯 생성
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>{title_1}</b>", f"<b>{title_2}</b>"))
    
    # ForeFoot 데이터
    x_avg_ff = FILT_FF[x].mean()
    y_avg_ff = FILT_FF[y1].mean()
    
    # 첫 번째 서브플롯 - ForeFoot 데이터
    fig.add_trace(
        go.Scatter(
            x=FILT_FF[x],
            y=FILT_FF[y1],
            mode='markers',
            marker=dict(size=6, color='rgba(0, 0, 255, 0.1)'),  # 연한 파란색으로 설정
            showlegend=False
        ),
        row=1, col=1
    )
    # 빨간색 선 추가 (ForeFoot)
    fig.add_trace(
        go.Scatter(
            x=[x_avg_ff, x_avg_ff],
            y=[FILT_FF[y1].min(), FILT_FF[y1].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[FILT_FF[x].min(), FILT_FF[x].max()],
            y=[y_avg_ff, y_avg_ff],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    # ForeFoot 텍스트 추가 (x축 평균)
    fig.add_annotation(
        text=f"AVG: {x_avg_ff:.2f}",
        x=x_avg_ff,
        y=FILT_FF[y1].min(),
        xref="x1",
        yref="y1",
        showarrow=False,
        # textangle=-90,  # 텍스트 90도 회전
        font=dict(
            family="Calibri",
            size=10,
            color="black"
        ),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    # ForeFoot 텍스트 추가 (y축 평균)
    fig.add_annotation(
        text=f"AVG: {y_avg_ff:.2f}",
        x=FILT_FF[x].max(),
        y=y_avg_ff,
        xref="x1",
        yref="y1",
        showarrow=False,
        font=dict(
            family="Calibri",
            size=10,
            color="black"
        ),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # 선택된 모델에 대해 ForeFoot에 점 추가
    for i, select in enumerate(selected_chart_names):
        selected_data = FILT_FF[FILT_FF['chart_name'] == select]
        fig.add_trace(
            go.Scatter(
                x=selected_data[x],
                y=selected_data[y2],
                mode='markers',
                marker=dict(size=12, color=colors[i % len(colors)]),  # 진한 색상
                name=f"{select.split('|')[0]} - {select.split('|')[1]}",
                showlegend=True,
                legendgroup = f"group_{i}"
            ),
            row=1, col=1
        )

        
    # ForeFoot 데이터
    x_avg_rf = FILT_FF[x].mean()
    y_avg_rf = FILT_FF[y2].mean()
    
    # 두 번째 서브플롯 - RearFoot 데이터
    fig.add_trace(
        go.Scatter(
            x=FILT_FF[x],
            y=FILT_FF[y2],
            mode='markers',
            marker=dict(size=6, color='rgba(0, 0, 255, 0.1)'),  # 연한 파란색으로 설정
            showlegend=False,
        ),
        row=1, col=2
    )
    # 빨간색 선 추가 (RearFoot)
    fig.add_trace(
        go.Scatter(
            x=[x_avg_rf, x_avg_rf],
            y=[FILT_FF[y2].min(), FILT_FF[y2].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[FILT_FF[x].min(), FILT_FF[x].max()],
            y=[y_avg_rf, y_avg_rf],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )

    # RearFoot 텍스트 추가 (x축 평균)
    fig.add_annotation(
        text=f"AVG: {x_avg_rf:.2f}",
        x=x_avg_rf,
        y=FILT_FF[y2].min(),
        xref="x2",
        yref="y2",
        showarrow=False,
        # textangle=-90,  # 텍스트 90도 회전
        font=dict(
            family="Calibri",
            size=10,
            color="black"
        ),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    # RearFoot 텍스트 추가 (y축 평균)
    fig.add_annotation(
        text=f"AVG: {y_avg_rf:.2f}",
        x=FILT_FF[x].max(),
        y=y_avg_rf,
        xref="x2",
        yref="y2",
        showarrow=False,
        font=dict(
            family="Calibri",
            size=10,
            color="black"
        ),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # 선택된 모델에 대해 RearFoot에 점 추가
    for i, select in enumerate(selected_chart_names):
        selected_data = FILT_FF[FILT_FF['chart_name'] == select]
        fig.add_trace(
            go.Scatter(
                x=selected_data[x],
                y=selected_data[y2],
                mode='markers',
                marker=dict(size=12, color=colors[i % len(colors)]),  # 진한 색상
                name=f"{select.split('|')[0]} - {select.split('|')[1]}",
                showlegend=False,
                legendgroup = f"group_{i}"
            ),
            row=1, col=2
        )

    # 전체 레이아웃 설정
    fig.update_layout(
        # title="Force (N) - Energy Return (%)",
        showlegend=True,
        width=1200,
        height=500,
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
            margin=dict(l=0, r=0, b=100, t=50)
    )

    # 축 이름 설정
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_yaxes(title_text=ylabel1, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=1, col=2)
    fig.update_xaxes(title_text=ylabel2, row=1, col=2)
    return fig

def slope_speciment_graph(all_filtered_data, selected_models, slope_col1, slope_col2, title_1, title_2, xlabel):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>{title_1}</b>", f"<b>{title_2}</b>"))
        # 첫 번째 서브플롯 - ForeFoot 데이터
        
    for i in range(len(selected_models)):
        ff = all_filtered_data[selected_models[i]]['ff_series'][slope_col1].values[0]
        # NaN 처리: NaN이 아닌 데이터에 대해서만 막대를 생성
        if not np.isnan(ff):
            fig.add_trace(go.Bar(
                x=[ff],  # Force 값
                y=[f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}"],  # Brand | Model 문자열
                orientation='h',  # 수평 막대
                text=f"{ff:.2f}",
                textposition='auto',
                marker_color=colors[i % len(colors)],  # 막대 색
                name = f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}",
                showlegend=True,
                legendgroup = f"group_{i}"
            ),
            row=1, col=1)

        ff = all_filtered_data[selected_models[i]]['ff_series'][slope_col2].values[0]
        # NaN 처리: NaN이 아닌 데이터에 대해서만 막대를 생성
        if not np.isnan(ff):
            fig.add_trace(go.Bar(
                x=[ff],  # Force 값
                y=[f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}"],  # Brand | Model 문자열
                orientation='h',  # 수평 막대
                text=f"{ff:.2f}",
                textposition='auto',
                marker_color=colors[i % len(colors)],  # 막대 색
                name = f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}",
                showlegend=False,
                legendgroup = f"group_{i}"
            ),
            row=1, col=2)
            
    # 레이아웃 설정
    fig.update_layout(
        # title={
        #     'text': title,
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'
        # },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Calibri", size=14),
        yaxis=dict(automargin=True),  # Y축 여백 자동 조정
        legend=dict(
            orientation="h",  
            yanchor="bottom",    
            y=-0.7,           
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
        showlegend=True,
        width=1200,
        height=400,
        margin=dict(l=0, r=0, b=100, t=50)
    )

    # 축 이름 설정
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=1, col=2)
    # 두 번째 y축 이름 제거
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2)
    
    return fig

def bar_speciment_graph(FF, FF_SD, col1, col2, title_1, title_2, xlabel1, xlabel2, selected_models):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>{title_1}</b>", f"<b>{title_2}</b>"))
        # 첫 번째 서브플롯 - ForeFoot 데이터
        
    for i in range(len(selected_models)):
        mean = FF[FF['chart_name'] == selected_models[i]][col1].values[0]
        std = FF_SD[FF_SD['chart_name'] == selected_models[i]][col1].values[0]
        
        # NaN 처리: NaN이 아닌 데이터에 대해서만 막대를 생성
        if not np.isnan(mean):
            fig.add_trace(go.Bar(
                x=[mean],  # Force 값
                y=[f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}"],  # Brand | Model 문자열
                orientation='h',  # 수평 막대
                text=f"{mean:.2f} ({std:.2f})" if not np.isnan(std) else f"{mean:.2f} (N/A)",  # 텍스트: 평균 (표준편차)
                textposition='auto',
                marker_color=colors[i % len(colors)],  # 막대 색
                name = f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}",
                showlegend=True,
                legendgroup = f"group_{i}"
            ),
            row=1, col=1)
        
    for i in range(len(selected_models)):
        mean = FF[FF['chart_name'] == selected_models[i]][col2].values[0]
        std = FF_SD[FF_SD['chart_name'] == selected_models[i]][col2].values[0]
        
        # NaN 처리: NaN이 아닌 데이터에 대해서만 막대를 생성
        if not np.isnan(mean):
            fig.add_trace(go.Bar(
                x=[mean],  # Force 값
                y=[f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}"],  # Brand | Model 문자열
                orientation='h',  # 수평 막대
                text=f"{mean:.2f} ({std:.2f})" if not np.isnan(std) else f"{mean:.2f} (N/A)",  # 텍스트: 평균 (표준편차)
                textposition='auto',
                marker_color=colors[i % len(colors)],  # 막대 색
                name = f"{selected_models[i].split('|')[0]} - {selected_models[i].split('|')[1]}",
                showlegend=False,
                legendgroup = f"group_{i}"
            ),
            row=1, col=2)
            
    # 레이아웃 설정
    fig.update_layout(
        # title={
        #     'text': title,
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'
        # },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Calibri", size=14),
        yaxis=dict(automargin=True),  # Y축 여백 자동 조정
        legend=dict(
            orientation="h",  
            yanchor="bottom",    
            y=-0.7,           
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
        showlegend=True,
        width=1200,
        height=400,
        margin=dict(l=0, r=0, b=100, t=50)
    )

    # 축 이름 설정
    fig.update_xaxes(title_text=xlabel1, row=1, col=1)
    fig.update_xaxes(title_text=xlabel2, row=1, col=2)
    # 두 번째 y축 이름 제거
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2)
    
    
    return fig

def for_dis_rel_specimen_graph(filterd_data, selected_models, title, xlabel, ylabel, selected_shoe_type):
    fig = go.Figure()

    if selected_shoe_type == 'TENNIS':
        ff_ds1_s, ff_ds1_e, ff_ds2_s, ff_ds2_e = 400, 600, 900, 1100
    
    elif selected_shoe_type == 'SPECIMEN':
        ff_ds1_s, ff_ds1_e, ff_ds2_s, ff_ds2_e = 250, 400, 600, 750    
    else:
        ff_ds1_s, ff_ds1_e, ff_ds2_s, ff_ds2_e = 400, 600, 900, 1100
    
    ff_x_max = 0  
    ff_x_min = 0  
    # 선택된 각 모델에 대해 데이터 불러오기
    for i, model in enumerate(selected_models):
        model_data = filterd_data.get(model, {})

        # RF Series 데이터가 있는 경우 그래프에 추가
        if not model_data.get('ff_series', pd.DataFrame()).empty:
            rf_series_data = model_data['ff_series']
            if ff_x_max < rf_series_data.max()['Disp (mm)']:
                ff_x_max = rf_series_data.max()['Disp (mm)']
            if ff_x_min > rf_series_data.min()['Disp (mm)']:
                ff_x_min = rf_series_data.min()['Disp (mm)']
                
            trace_rf = go.Scatter(
                x=rf_series_data['Disp (mm)'], 
                y=rf_series_data['Force (N)'], 
                mode='lines+markers', 
                name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                line=dict(color=colors[i % len(colors)], width=2, dash='dot'),  # 색상 리스트에서 순차적으로 선택
                showlegend=True,
                legendgroup = f"group_{i}"
            )
            fig.add_trace(trace_rf)
        

            # ds1, ds2 구간에서 올라가는 부분의 끝 지점과 내려가는 부분의 끝 지점 좌표
            ds1_up_f1 = rf_series_data['ds1_upward_f1'].iloc[0]
            ds1_up_f2 = rf_series_data['ds1_upward_f2'].iloc[0]
            ds1_up_d1 = rf_series_data['ds1_upward_d1'].iloc[0]
            ds1_up_d2 = rf_series_data['ds1_upward_d2'].iloc[0]

            ds2_up_f1 = rf_series_data['ds2_upward_f1'].iloc[0]
            ds2_up_f2 = rf_series_data['ds2_upward_f2'].iloc[0]
            ds2_up_d1 = rf_series_data['ds2_upward_d1'].iloc[0]
            ds2_up_d2 = rf_series_data['ds2_upward_d2'].iloc[0]
            
            # ds1 구간 점선
            fig.add_trace(
                go.Scatter(
                    x=[ds1_up_d1, ds1_up_d2],
                    y=[ds1_up_f1, ds1_up_f2],
                    mode="lines",
                    name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                    line=dict(color='red', width=4),
                    showlegend=False,
                    legendgroup=f"group_{i}"
                ),
            )

            # ds2 구간 점선
            fig.add_trace(
                go.Scatter(
                    x=[ds2_up_d1, ds2_up_d2],
                    y=[ds2_up_f1, ds2_up_f2],
                    mode="lines",
                    line=dict(color='red', width=4),
                    name=f"{model.split('|')[0]} - {model.split('|')[1]}", 
                    showlegend=False,
                    legendgroup=f"group_{i}"
                ),
            )
     
            
    # Shaded regions between 400-600 N and 900-1100 N
    fig.add_shape(
        type="rect",
        xref="x", yref="y",  # X-axis uses paper (full chart width), Y uses y-axis values
        x0=ff_x_min, x1=ff_x_max+2, y0=ff_ds1_s, y1=ff_ds1_e,
        fillcolor="rgba(255, 182, 193, 0.4)",  # Light red, semi-transparent
        line_width=0,
        layer="below",  # Place the shading behind the traces,
    )

    fig.add_shape(
        type="rect",
        xref="x", yref="y",
        x0=ff_x_min, x1=ff_x_max+2, y0=ff_ds2_s, y1=ff_ds2_e,
        fillcolor="rgba(255, 182, 193, 0.4)",  # Light red, semi-transparent
        line_width=0,
        layer="below",
    )

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },       
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=500,
        xaxis=dict(
            showgrid=True, 
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation="v",   # 세로로 표시
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,  # 오른쪽에 위치
            font=dict(
                family="Calibri",
                size=12,
                color="black"
            ),
            bgcolor="white",
            bordercolor="black",
            borderwidth=2
        )
    )

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel, showgrid=True, gridcolor='lightgray')
    return fig

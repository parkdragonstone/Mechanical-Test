import data_concat
import streamlit as st
from DASHBOARD.dashboard_impact import dashboard_impact
from DASHBOARD.dashboard_flex import dashboard_flex
from DASHBOARD.dashboard_traction import dashboard_traction
from DASHBOARD.dashboard_drag import dashboard_drag

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    chart_to_images, all_filtered_data, FF, RF, FF_SD, RF_SD, label = data_concat.impact_data_concat()
    flex_data, FLEX_DF, flex_labels = data_concat.flex_data_concat()
    TRACTION_DF, TRACTION_FILT = data_concat.tracion_data_concat()
    DRAG = data_concat.drag_data_concat()
    
    return chart_to_images, all_filtered_data, FF, RF, FF_SD, RF_SD, label, \
              flex_data, FLEX_DF, flex_labels, \
                  TRACTION_DF, TRACTION_FILT, \
                      DRAG

impact_chart_to_images, impact_all_filtered_data, impact_FF, impact_RF, impact_FF_SD, impact_RF_SD, impact_label, \
    flex_data, FLEX_DF, flex_labels, \
        TRACTION_DF, TRACTION_FILT,\
            DRAG = load_data()
    
col1, col2, col3 = st.columns([0.2,2,0.2])

with col2:
    # 페이지 전체 폰트를 Calibri로 설정하는 CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Calibri&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Calibri', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)

    # # Secret Manager에서 불러오기
    # VALID_USERNAME = st.secrets["VALID_USERNAME"]
    # VALID_PASSWORD = st.secrets["VALID_PASSWORD"]

    # if st.session_state['logged_in']:
    #     # 로그인된 사용자만 접근할 수 있는 기능
    #     st.write("This is a protected page.")
    # else:
    #     st.write("Please login to access this page.")
    
    # 사이드바에 모델 선택 메뉴 및 선택된 항목 표시
    with st.sidebar:
        selected_data_type = st.radio('Select Mechanical Test', ['IMPACT','FLEX', 'TRACTION' ,'DRAG'])
    
    if selected_data_type == 'IMPACT':
        dashboard_impact(impact_label,impact_FF, impact_RF, impact_FF_SD, impact_RF_SD, impact_chart_to_images,impact_all_filtered_data)

    elif selected_data_type == 'FLEX':
        dashboard_flex(flex_data, flex_labels, FLEX_DF)
        
    elif selected_data_type == 'TRACTION':
        dashboard_traction(TRACTION_DF, TRACTION_FILT)
        
    elif selected_data_type == 'DRAG':
        dashboard_drag(DRAG)
            
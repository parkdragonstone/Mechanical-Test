import streamlit as st
import pandas as pd
import graph_flex

def dashboard_flex(flex_data, flex_labels, FLEX_DF):
    with st.sidebar:
        selected_shoe_type = st.multiselect('Select Type of Footwear', flex_labels['category'].unique())
        selected_chart_names = st.multiselect('Select Models',sorted(flex_labels[flex_labels['category'].isin(selected_shoe_type)]['chart_name']))

        if selected_chart_names:
            st.markdown("<ul>", unsafe_allow_html=True)
            for chart_name in selected_chart_names:
                st.markdown(f"<li><strong>{chart_name}</strong>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            st.write('No models selected.')
    
    FLEX_SELECT = FLEX_DF[FLEX_DF['chart_name'].isin(selected_chart_names)]
    flex_fig = graph_flex.flex_graph(flex_data, FLEX_SELECT, selected_chart_names)
    
    # 페이지 제목 설정
    st.title("oooo CENTER - FLEX TEST")
    
    with st.container(border=True):
        st.markdown("""
            <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">TEST PROTOCOL</h3>
            </div>
            """, unsafe_allow_html=True)
        st.write(f"**Flexion Angle Range** : 45 degree")
        st.write("**Flexion Number**: Pre-Flex : 55 / Flex for data collection : 5")
        
    with st.container(border=True):
        st.markdown("""
                <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                    <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DATA OVERALL</h3>
                </div>
                """, unsafe_allow_html=True)

        st.dataframe(FLEX_SELECT.iloc[:,:-1], use_container_width=True)
        
    with st.container(border=True):
        st.markdown("""
                <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                    <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DATA FIGURE</h3>
                </div>
                """, unsafe_allow_html=True)

        st.plotly_chart(flex_fig, use_container_width=True, key="flex_graph")
    
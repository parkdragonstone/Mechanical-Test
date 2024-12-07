import streamlit as st
import pandas as pd
import graph_drag

def dashboard_drag(DRAG):
    with st.sidebar:
        selected_shoe_type = st.selectbox('Select Type of Footwear', DRAG['category'].unique())
        selected_chart_names = st.multiselect('Select Models',sorted(DRAG[DRAG['category'] == selected_shoe_type]['chart_name']))

        if selected_chart_names:
            st.markdown("<ul>", unsafe_allow_html=True)
            for chart_name in selected_chart_names:
                st.markdown(f"<li><strong>{chart_name}</strong>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            st.write('No models selected.')
    
    
    real_fig = graph_drag.drag_graph(DRAG, selected_chart_names, "DRAG Test (Real Mass)")
    zero_fig = graph_drag.drag_graph(DRAG, selected_chart_names, "Drag Test (Set Zero)")
    diff_fig = graph_drag.drag_graph(DRAG, selected_chart_names, "Drag Teset (Differential)")
    st.title("FAST CENTER - DRAG TEST")
    
    with st.container(border=True):
        st.markdown("""
            <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DRAG TEST RESULT</h3>
            </div>
            """, unsafe_allow_html=True)
        st.plotly_chart(real_fig, use_container_width=True, key=f"real_fig")
        st.plotly_chart(zero_fig, use_container_width=True, key=f"zero_fig")
        st.plotly_chart(diff_fig, use_container_width=True, key=f"diff_fig")

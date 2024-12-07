import streamlit as st
import pandas as pd
import graph_traction

def dashboard_traction(TRACTION_DF, TRACTION_FILT):
    
    with st.sidebar:
        selected_shoe_type = st.selectbox('Select Type of Footwear', TRACTION_DF['category'].unique())
        selected_chart_names = st.multiselect('Select Models',sorted(TRACTION_DF[TRACTION_DF['category'] == selected_shoe_type]['chart_name'].unique()))

        if selected_chart_names:
            st.markdown("<ul>", unsafe_allow_html=True)
            for chart_name in selected_chart_names:
                st.markdown(f"<li><strong>{chart_name}</strong>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            st.write('No models selected.')
            
            
    st.title("oooo CENTER - TRACTION TEST")
    
    TRACTION_SELECT = TRACTION_DF[TRACTION_DF['chart_name'].isin(selected_chart_names)]
    SURFACES = TRACTION_SELECT['surface_no'].unique()
    traction_df = {surface : TRACTION_SELECT[TRACTION_SELECT['surface_no'] == surface] for surface in SURFACES}
    figures_static = {surface : graph_traction.static_friction_graph(traction_df[surface], selected_chart_names) for surface in SURFACES}
    figures_kinetic = {surface : graph_traction.kinetic_friction_graph(traction_df[surface], selected_chart_names) for surface in SURFACES}
    
    TRACTION_FILT_SELECT = TRACTION_FILT[TRACTION_FILT[('','chart_name')].isin(selected_chart_names)]
    traction_filt_df = {surface : TRACTION_FILT_SELECT[TRACTION_FILT_SELECT[('','Surface')] == surface].iloc[:,:-1] for surface in SURFACES}
    
    with st.container(border=True):
        st.markdown("""
                <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                    <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DATA OVERALL</h3>
                </div>
                """, unsafe_allow_html=True)
        
        for surface in SURFACES:
            st.write(surface)
            st.dataframe(traction_filt_df[surface], use_container_width=True)
    
    for surface in SURFACES:
        with st.container(border=True):
            st.markdown(f"""
                    <style>
                        .surface-header {{
                            background-color: #f5f5f5;
                            padding: 0px;
                            border-radius: 0px;
                            margin-bottom: 10px;
                        }}
                        .surface-header h3 {{
                            color: #4F60AF;
                            font-size: 17px;
                            margin: 10px;
                        }}
                    </style>
                    <div class="surface-header">
                        <h3>{surface}</h3>
                    </div>
                """, unsafe_allow_html=True)

            st.plotly_chart(figures_static[surface], use_container_width=True, key=f"figures_static_{surface}")
            col1, col2, col3 = st.columns([1,3,1])
            with col2:
                st.plotly_chart(figures_kinetic[surface], use_container_width=True, key=f"figures_kinetic_{surface}")
            
            

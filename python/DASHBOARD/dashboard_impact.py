import streamlit as st
import pandas as pd
import graph_impact

def dashboard_impact(label,FF, RF, FF_SD, RF_SD, chart_to_images,all_filtered_data) :
    with st.sidebar:
        selected_shoe_type = st.multiselect('Select Type of Footwear', label['category'].unique())
        selected_chart_names = st.multiselect('Select Models',sorted(label[label['category'].isin(selected_shoe_type)]['chart_name']))

        if selected_chart_names:
            st.markdown("<ul>", unsafe_allow_html=True)
            for chart_name in selected_chart_names:
                st.markdown(f"<li><strong>{chart_name}</strong>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            st.write('No models selected.')
            
    # 페이지 제목 설정
    st.title("FAST CENTER - IMPACT TEST")

    # 선택된 모델 중 하나를 다시 선택하는 selectbox
    if selected_chart_names:
        selected_model = st.selectbox('Target Model', sorted(selected_chart_names))
        # 왼쪽 텍스트
        info_table = FF.loc[(selected_model.split('|')[0][:-1], selected_model.split('|')[1][1:-1])]
        
        with st.container(height=400, border=True):
            st.markdown("""
                <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                    <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">TEST PROTOCOL</h3>
                </div>
                """, unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])  # 왼쪽이 더 넓도록 비율 설정

            with col1:
                st.markdown("**ASTM F 1976-13 (2013)**")
                st.write(f"**Drop Mass** : {info_table['Missile Mass (kg)'].values[0]} kg")
                st.write(f"**Drop Height** : {int(info_table['Drop Height (mm)'].values[0])} mm")
                st.write(f"**Missile**: {int(info_table['Missile Head Diameter (mm)'].values[0])}mm diameter ({info_table['Missile Head Type'].values[0]} tup)")
                st.write("**Impact Number**: Pre-Impact : 25 / Impact for data collection : 5")

            # 오른쪽 이미지
            with col2:
                
                # 선택된 모델에 해당하는 이미지 목록 가져오기
                if selected_model:
                    model_images = chart_to_images.get(selected_model, [])[0]
                    
                    if len(model_images) == 1:
                        st.image(model_images[0], caption=f'{selected_model}',use_column_width=True)
                        
                    # 선택된 모델에 이미지가 있는 경우
                    elif model_images:
                        
                        # select_slider로 이미지 변경
                        selected_image = st.select_slider(
                            label='dd',
                            options=model_images,
                            format_func=lambda x: '',
                            label_visibility='collapsed'
                        )

                        # 선택된 이미지를 화면에 출력
                        st.image(selected_image, caption=f'{selected_model}')

                    else:
                        st.write(f"No images available for {selected_model}.")
    ## THICKNESS VERSION            
    # FF_COLS = ['Test Date' , 'Mass (g)', 'Thickness ForeFoot (mm)', 'Force (N)', 'Force (N/mm)', 'Energy Return (%)',
    # 'Accel (g)', 'Max Pen (%)', 'Max Pen (mm)', 'Dwell T (ms)']
    # RF_COLS = ['Test Date' , 'Mass (g)','Thickness RearFoot (mm)', 'Force (N)', 'Force (N/mm)', 'Energy Return (%)',
    # 'Accel (g)', 'Max Pen (%)', 'Max Pen (mm)', 'Dwell T (ms)']
    
    ## NO THICKNESS VERSION
    FF_COLS = ['Test Date' , 'Mass (g)',  'Force (N)', 'Force (N/mm)', 'Energy Return (%)',
    'Accel (g)', 'Max Pen (%)', 'Max Pen (mm)', 'Dwell T (ms)']
    RF_COLS = ['Test Date' , 'Mass (g)', 'Force (N)', 'Force (N/mm)', 'Energy Return (%)',
    'Accel (g)', 'Max Pen (%)', 'Max Pen (mm)', 'Dwell T (ms)']   

    if selected_shoe_type != 'SPECIMEN':     
        chart_name = list(label[label['category'].isin(selected_shoe_type)]['chart_name'])
        FILT_RF = RF[RF['chart_name'].isin(chart_name)]
        FILT_FF = FF[FF['chart_name'].isin(chart_name)]
        
        filt_ff_df = FILT_FF[FILT_FF['chart_name'].isin(selected_chart_names)][FF_COLS]
        filt_rf_df = FILT_RF[FILT_RF['chart_name'].isin(selected_chart_names)][RF_COLS]

        ff_columns = [('ForeFoot', col) if col not in ['Test Date', 'Mass (g)'] else ('', col) for col in filt_ff_df.columns]
        rf_columns =  [('RearFoot', col) if col not in ['Test Date', 'Mass (g)'] else ('', col) for col in filt_rf_df.columns]

        columns = [*ff_columns, rf_columns]

        filt_ff_df.columns = pd.MultiIndex.from_tuples(ff_columns)
        filt_rf_df.columns = pd.MultiIndex.from_tuples(rf_columns)

        filt_df = pd.concat([filt_ff_df, filt_rf_df],axis=1)

        force_energy_fig = graph_impact.scatter_plotly("Force (N) - Energy Return (%) | ForeFoot", "Force (N) - Energy Return (%) | RearFoot", 'Force (N)', 'Energy Return (%)', selected_chart_names, FILT_FF, FILT_RF, 'Force (N)', 'Energy Return (%)')
        force_thick_fig = graph_impact.scatter_plotly("Force (N) - Displacement (mm) | ForeFoot", "Force (N) - Displacement (mm) | RearFoot", 'Force (N)', 'Max Pen (mm)', selected_chart_names, FILT_FF, FILT_RF, 'Force (N)', 'Displacement (mm)')
        
        # thick_fig    = graph_impact.bar_graph(FF, FF_SD, RF, RF_SD, 'Thickness'        , "Material Thickness (mm) | ForeFoot", "Material Thickness (mm) | RearFoot" , 'Thickness (mm)'   , selected_chart_names)
        er_fig       = graph_impact.bar_graph(FF, FF_SD, RF, RF_SD, 'Energy Return (%)', "Energy Return (%) | ForeFoot"      , "Energy Return (%) | RearFoot"       , 'Energy Return (%)', selected_chart_names)
        force_fig    = graph_impact.bar_graph(FF, FF_SD, RF, RF_SD, 'Force (N)'        , "Force (N) | ForeFoot"              , "Force (N) | RearFoot"               , 'Force (N)'        , selected_chart_names)
        force_n_fig  = graph_impact.bar_graph(FF, FF_SD, RF, RF_SD, 'Force (N/mm)'     , "Force (N/mm) | ForeFoot"           , "Force (N/mm) | RearFoot"            , 'Force (N/mm)'     , selected_chart_names)
        deform_fig   = graph_impact.bar_graph(FF, FF_SD, RF, RF_SD, 'Max Pen (mm)'     , 'Deformation (mm) | ForeFoot'       , 'Deformation (mm) | RearFoot'        , 'Deformation (mm)' , selected_chart_names)
        deform_p_fig = graph_impact.bar_graph(FF, FF_SD, RF, RF_SD, 'Max Pen (%)'      , 'Deformation (%) | ForeFoot'        , 'Deformation (mm) | RearFoot'        , 'Deformation (%)'  , selected_chart_names)
        force_dis_fig = graph_impact.for_dis_rel_graph(all_filtered_data, selected_chart_names, "Force (N) - Displacement (mm) | ForeFoot", 'Force (N) - Displacement (mm) | Rearfoot', "Force (N)", "Displacement (mm)", selected_shoe_type)
        ds1_upward_fig = graph_impact.slope_graph(all_filtered_data, selected_chart_names, 'ds1_upward_slope', 'DS1 Upward Slope | ForeFoot', 'DS1 Upward Slope | RearFoot', 'Slope Coefficient')
        ds2_upward_fig = graph_impact.slope_graph(all_filtered_data, selected_chart_names, 'ds2_upward_slope', 'DS2 Upward Slope | ForeFoot', 'DS2 Upward Slope | RearFoot', 'Slope Coefficient')

        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DATA OVERALL</h3>
                    </div>
                    """, unsafe_allow_html=True)

            st.dataframe(filt_ff_df, use_container_width=True)
            st.dataframe(filt_rf_df, use_container_width=True)

        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">IMPACT TEST RESULT</h3>
                    </div>
                    """, unsafe_allow_html=True)

            st.plotly_chart(force_energy_fig, use_container_width=True, key="force_energy_fig")
            st.plotly_chart(force_thick_fig, use_container_width=True, key="force_thick_fig")

        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">FORCE - DISPLACEMENT</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.plotly_chart(force_dis_fig, use_container_width=True, key="force_dis_fig")
            st.plotly_chart(ds1_upward_fig, use_container_width=True, key="ds1_upward_fig")
            st.plotly_chart(ds2_upward_fig, use_container_width=True, key="ds2_upward_fig")

        # with st.container(border=True):
        #     st.markdown("""
        #             <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
        #                 <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">MATERIAL THICKNESS</h3>
        #             </div>
        #             """, unsafe_allow_html=True)
        #     st.plotly_chart(thick_fig, use_container_width=True, key="thick_fig")
            
        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">ENERGY RETURN</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.plotly_chart(er_fig, use_container_width=True, key="er_fig")
            
        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">FORCE (N)</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.plotly_chart(force_fig, use_container_width=True, key="force_fig")

        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">Force (N/mm)</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.plotly_chart(force_n_fig, use_container_width=True, key="force_n_fig")
            
        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DEFORMATION (mm)</h3>
                    </div>
                    """, unsafe_allow_html=True)    
            st.plotly_chart(deform_fig, use_container_width=True, key="deform_fig")
            
        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DEFORMATION (%)</h3>
                    </div>
                    """, unsafe_allow_html=True)    
            st.plotly_chart(deform_p_fig, use_container_width=True, key="deform_p_fig")
            
    elif selected_shoe_type == 'SPECIMEN':
        chart_name = list(label[label['category'].isin(selected_shoe_type)]['chart_name'])
        FILT_FF = FF[FF['chart_name'].isin(chart_name)]
        filt_ff_df = FILT_FF[FILT_FF['chart_name'].isin(selected_chart_names)][FF_COLS]
        # ff_columns = [('ForeFoot', col) if col not in ['Test Date', 'Mass (g)'] else ('', col) for col in filt_ff_df.columns]

        # filt_ff_df.columns = pd.MultiIndex.from_tuples(ff_columns)
        
        force_energy_thick_fig = graph_impact.scatter__specimen_plotly("Force (N) - Energy Return (%)", "Force (N) - Displacement (mm)", 'Force (N)',  'Energy Return (%)', 'Max Pen (mm)', selected_chart_names, FILT_FF, 'Force (N)',  'Energy Return (%)', 'Displacement (mm)')
        thick_er_fig    = graph_impact.bar_speciment_graph(FF, FF_SD, 'Thickness ForeFoot (mm)' , 'Energy Return (%)', "Material Thickness (mm)", "Energy Return (%)" , 'Thickness (mm)'   , 'Energy Return (%)', selected_chart_names)
        force_fig    = graph_impact.bar_speciment_graph(FF, FF_SD, 'Force (N)' , 'Force (N/mm)', "Force (N)", "Force (N/mm)" , "Force (N)"   , "Force (N/mm)", selected_chart_names)
        deform_fig   = graph_impact.bar_speciment_graph(FF, FF_SD,'Max Pen (mm)','Max Pen (%)', 'Deformation (mm)'  , 'Deformation (%)' , 'Deformation (mm)', 'Deformation (%)' , selected_chart_names)
        force_dis_speciment_fig = graph_impact.for_dis_rel_specimen_graph(all_filtered_data, selected_chart_names, "Force (N) - Displacement (mm)", "Force (N)", "Displacement (mm)", selected_shoe_type)
        ds_upward_fig = graph_impact.slope_speciment_graph(all_filtered_data, selected_chart_names, 'ds1_upward_slope', 'ds2_upward_slope',  'DS1 Upward Slope', 'DS2 Upward Slope' , 'Slope Coefficient')

        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DATA OVERALL</h3>
                    </div>
                    """, unsafe_allow_html=True)

            st.dataframe(filt_ff_df, use_container_width=True)
            
        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">IMPACT TEST RESULT</h3>
                    </div>
                    """, unsafe_allow_html=True)

            st.plotly_chart(force_energy_thick_fig, use_container_width=True, key="force_energy_thick_fig")

        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">FORCE - DISPLACEMENT</h3>
                    </div>
                    """, unsafe_allow_html=True)
            col1, col2,col3 = st.columns([0.4,2,0.4])
            with col2:
                st.plotly_chart(force_dis_speciment_fig, use_container_width=True, key="force_dis_speciment_fig")
            st.plotly_chart(ds_upward_fig, use_container_width=True, key="ds_upward_fig")

        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">MATERIAL THICKNESS (mm) & ENERGY RETURN (%)</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.plotly_chart(thick_er_fig, use_container_width=True, key="thick_er_fig")
            
        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">FORCE (N) & FORCE (N/mm)</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.plotly_chart(force_fig, use_container_width=True, key="force_fig")
            
        with st.container(border=True):
            st.markdown("""
                    <div style="background-color: #f5f5f5; padding: 0px; border-radius: 0px;">
                        <h3 style="color: #4F60AF;  font-size: 17px;margin: 10px;">DEFORMATION (mm) & DEFORMATION (%)</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.plotly_chart(deform_fig, use_container_width=True, key="deform_fig")

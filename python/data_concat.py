import os
from glob import glob
import pandas as pd
import processing
import numpy as np


def impact_data_concat():
    parent_dir = os.getcwd()
    data_dir = [i.replace('\\','/') for i in glob(os.path.join(parent_dir, '1. IMPACT TEST', '**','**','*Series.xls')) + glob(os.path.join(parent_dir, '1. IMPACT TEST', '**', '**', '*Table.xls'))]
    label = pd.read_excel(os.path.join(parent_dir, '1. IMPACT TEST', 'Mechanical Test DB - Impact.xlsx'))
    label['chart_name'] = label['brand'] + " | " + label['model_name'] + " | " + label['test_date'].dt.strftime('%Y-%m-%d')
    label = label[label['test'] == 'IMPACT']

    image_dir = [i.replace('\\','/') for i in glob(os.path.join(parent_dir,  '1. IMPACT TEST','**','**','image','*.jpg'))+ glob(os.path.join(parent_dir,  '1. IMPACT TEST','**','**','image','*.JPG')) + glob(os.path.join(parent_dir,  '1. IMPACT TEST','**','**','image','*.PNG')) + glob(os.path.join(parent_dir,  '1. IMPACT TEST','**', '**', 'image', '*.png'))]
    DATA = processing.data_load(data_dir)
    
    # 파일 경로에서 폴더명 추출
    folder_names = [os.path.basename(os.path.dirname(path)) for path in data_dir]
    unique_folder_names = list(set(folder_names))

    folder_to_chart, chart_to_files = processing.chart_folder_to_files(label, unique_folder_names, data_dir)
    folder_to_chart_image, chart_to_images = processing.chart_folder_to_files(label, unique_folder_names, image_dir)

    all_chart_names = list(chart_to_files.keys())
    all_files  = {name : chart_to_files[name][0] for name in all_chart_names}
    all_labels = {name : chart_to_files[name][1] for name in all_chart_names}

    all_filtered_files = processing.process_files(all_files)
    all_filtered_data = processing.filtering_data(all_filtered_files, all_labels, DATA)
    all_filtered_data = processing.add_slope_data(all_filtered_data, all_labels)

    FF, RF, FF_SD, RF_SD = processing.overall_data_table(all_chart_names, all_filtered_data, all_labels)
    FF[('Test Date')] = FF[('Test Date')].dt.strftime('%Y-%m-%d')
    RF[('Test Date')] = RF[('Test Date')].dt.strftime('%Y-%m-%d')
    return chart_to_images, all_filtered_data, FF, RF, FF_SD, RF_SD, label

def flex_data_concat():
    Parent_dir = os.getcwd()
    flex_labels = pd.read_excel(os.path.join(Parent_dir, '2. FLEX TEST', "Mechanical Test DB - Flex.xlsx"))
    flex_labels['chart_name'] = flex_labels['brand'] + " | " + flex_labels['model_name'] + " | " + flex_labels['test_date'].dt.strftime('%Y-%m-%d')

    flx_dir = [i.replace('\\','/') for i in glob(os.path.join(Parent_dir, '2. FLEX TEST', '2024','**',"*flx")) if '45_N' in i]
    folder_names = [os.path.basename(os.path.dirname(path)) for path in flx_dir]
    unique_folder_names = list(set(folder_names))
    
    folder_to_chart = {}
    chart_to_files = {}

    # 폴더명에서 숫자 부분 추출 및 chart_name과 매칭
    for folder in unique_folder_names:
        folder_num = folder.split('.')[0]
        
        # 매칭되는 chart_name을 찾기
        matching_row = flex_labels[flex_labels['shoe_num'].astype(str) == folder_num]
        
        if not matching_row.empty:
            chart_name = matching_row['chart_name'].values[0]
            folder_to_chart[chart_name] = folder
            
            # 해당 폴더 내의 모든 파일 경로를 리스트로 가져오기
            # 여기서는 폴더 경로를 사용해 모든 파일을 검색
            files_in_folder = [i for i in flx_dir if folder in i][0]
            chart_to_files[chart_name] = files_in_folder, matching_row
    
    flex_data = {}
    FLEX_DF = pd.DataFrame()
    for file in chart_to_files:
        f = chart_to_files[file][0]
        df = pd.read_table(f, header = None)
        max_idx = np.argmax(df[1])
        df[3] = df[1] - df[1].min()

        # 파란색 구간 (0부터 최대값까지)
        x_blue = df[0][:max_idx]
        y_blue = df[3][:max_idx]

        # 주황색 구간 (최대값부터 끝까지)
        x_orange = df[0][max_idx:]
        y_orange = df[3][max_idx:]

        # 두 곡선의 차이 계산 (파란색에서 주황색 뺌)
        area_blue = np.trapz(y_blue, x_blue)
        area_orange = np.trapz(y_orange, x_orange)

        d = df[df[1] > 0]
        upward_0_deg, downward_0_deg = d[0].values[0], d[0].values[-1]

        info_use_cols = ['brand','model_name','test_date','category','shoe_mass','size_mm','chart_name']
        info = chart_to_files[file][1][info_use_cols]
        info['torque'] = round(df[1].max(),3)
        info['energy return'] = round(abs(100*area_orange/area_blue),3)
        info['energy (upward)'] = round(area_blue,3)
        info['energy (downward)'] = round(-area_orange,3)

        FLEX_DF = pd.concat([FLEX_DF, info],axis=0)
        flex_data[file] = df
        
    rename_cols = ['Brand', 'Model','Test Date', 'Category',' Shoe Mass (g)', 'Shoe Size (mm)','chart_name','Torque (N/mm)', 'Energy Return', 'Energy (Upward)', 'Energy (Downward)']
    FLEX_DF.columns = rename_cols
    FLEX_DF = FLEX_DF.set_index(['Brand','Model'])
    
    FLEX_DF = FLEX_DF[['Test Date', 'Category',' Shoe Mass (g)', 'Shoe Size (mm)','Torque (N/mm)', 'Energy Return', 'Energy (Upward)', 'Energy (Downward)','chart_name']]
    FLEX_DF['Test Date'] = FLEX_DF['Test Date'].dt.strftime('%Y-%m-%d')
    return flex_data, FLEX_DF, flex_labels
    # 각 행별로 가장 큰 값과 가장 작은 값을 제외한 나머지 값만 남김

def exclude_min_max(row, columns, col_name):
    rotation_values = row[columns].values
    filtered_values = sorted(rotation_values)[1:-1]  # 가장 작은 값과 가장 큰 값을 제외
    return pd.Series(filtered_values, index=[f"{col_name}_{i+1}" for i in range(len(filtered_values))])

def tracion_data_concat():
    Parent_dir = os.getcwd()
    traction_dir = glob(os.path.join(Parent_dir,'3. TRACTION TEST', '2024', '*xlsx'))

    TRACTION = pd.DataFrame()
    for dir in traction_dir:
        df = pd.read_excel(dir)
        TRACTION = pd.concat([TRACTION,df],axis=0)
        
    info = [
     'shoe_num', 'test_date', 'brand', 'model_name', 'category',
    'temperature', 'humidity', 'surface_no', 'loading_mass_kg',
        ]

    # rotation 열 선택
    rotation_columns  = [col for col in TRACTION.columns if 'rotation_p_t' in col]
    lateral_k_columns = [col for col in TRACTION.columns if 'lateral_k_t' in col]
    lateral_p_columns = [col for col in TRACTION.columns if 'lateral_p_t' in col]
    front_k_columns   = [col for col in TRACTION.columns if 'front_k_t' in col]
    front_p_columns   = [col for col in TRACTION.columns if 'front_p_t' in col]

    # 각 행별로 가장 큰 값과 가장 작은 값을 제외한 나머지 값만 남김
    def exclude_min_max(row, columns, col_name):
        rotation_values = row[columns].values
        filtered_values = sorted(rotation_values)[1:-1]  # 가장 작은 값과 가장 큰 값을 제외
        return pd.Series(filtered_values, index=[f"{col_name}_{i+1}" for i in range(len(filtered_values))])

    # 새로운 데이터프레임 생성
    rotation_p_df = TRACTION.apply(exclude_min_max, 
                                axis=1, 
                                args=(rotation_columns, "rotation_p")  # 추가 인자 전달
                                )
    lateral_p_df = TRACTION.apply(exclude_min_max, 
                                axis=1, 
                                args=(lateral_p_columns, "lateral_p")  # 추가 인자 전달
                                )
    front_p_df = TRACTION.apply(exclude_min_max, 
                                axis=1, 
                                args=(front_p_columns, "front_p")  # 추가 인자 전달
                                )
    lateral_k_df = TRACTION.apply(exclude_min_max, 
                                axis=1, 
                                args=(lateral_k_columns, "lateral_k")  # 추가 인자 전달
                                )
    front_k_df = TRACTION.apply(exclude_min_max, 
                                axis=1, 
                                args=(front_k_columns, "front_k")  # 추가 인자 전달
                                )

    front_k = pd.DataFrame({'front_k_mean' : front_k_df.mean(axis= 1)})
    front_k['front_k_std'] = front_k_df.std(axis= 1)
    front_p = pd.DataFrame({'front_p_mean' : front_p_df.mean(axis= 1)})
    front_p['front_p_std'] = front_p_df.std(axis= 1)
    lateral_k = pd.DataFrame({'lateral_k_mean' : lateral_k_df.mean(axis= 1)})
    lateral_k['lateral_k_std'] = lateral_k_df.std(axis= 1)
    lateral_p = pd.DataFrame({'lateral_p_mean' : lateral_p_df.mean(axis= 1)})
    lateral_p['lateral_p_std'] = lateral_p_df.std(axis= 1)
    rotation_p = pd.DataFrame({'rotation_p_mean' : rotation_p_df.mean(axis= 1)})
    rotation_p['rotation_p_std'] = rotation_p_df.std(axis= 1)

    TRACTION_DF = pd.concat([
        TRACTION[info], front_p, lateral_p, rotation_p, front_k, lateral_k
    ], axis=1)

    TRACTION_DF['chart_name'] = TRACTION_DF['brand'] + " | " + TRACTION_DF['model_name']+  " | " + TRACTION_DF['test_date'].dt.strftime('%Y-%m-%d')

    TRACTION_DF = TRACTION_DF.round(3)
    
    df =  TRACTION_DF.copy()

    df['Static Friction (front)'] = df['front_p_mean'].astype(str) + " (" + df['front_p_std'].astype(str) + ")"
    df['Static Friction (lateral)'] = df['lateral_p_mean'].astype(str) + " (" + df['lateral_p_std'].astype(str) + ")"
    df['Static Friction (rotation)'] = df['rotation_p_mean'].astype(str) + " (" + df['rotation_p_std'].astype(str) + ")"
    df['Kinetic Friction (front)'] = df['front_k_mean'].astype(str) + " (" + df['front_k_std'].astype(str) + ")"
    df['Kinetic Friction (lateral)'] = df['lateral_k_mean'].astype(str) + " (" + df['lateral_k_std'].astype(str) + ")"

    # 필요 없는 열 제거
    df = df[['brand', 'model_name', 'test_date', 'category', 'surface_no', 'loading_mass_kg',
            'Static Friction (front)', 'Static Friction (lateral)', 'Static Friction (rotation)',
            'Kinetic Friction (front)', 'Kinetic Friction (lateral)','chart_name']]


    # brand와 model_name을 인덱스로 설정
    df.set_index(['brand', 'model_name'], inplace=True)

    # 멀티 컬럼 설정
    df.columns = pd.MultiIndex.from_tuples([
        ('', 'Test Date'),
        ('', 'Category'),
        ('','Surface'),
        ('','Loading Mass (kg)'),
        ('Static Friction', 'Front (N)'),
        ('Static Friction', 'Lateral (N)'),
        ('Static Friction', 'Rotation (N/m)'),
        ('Kinetic Friction', 'Front (N)'),
        ('Kinetic Friction', 'Lateral (N)'),
        ('','chart_name')
    ])
    df[('','Test Date')] = df[('','Test Date')].dt.strftime('%Y-%m-%d')
    df[('','Loading Mass (kg)')] = df[('','Loading Mass (kg)')].astype(int)
    return TRACTION_DF, df
    
def drag_data_concat():
    Parent_dir = os.getcwd()
    drag_dir = glob(os.path.join(Parent_dir,'4. DRAG TEST', '2024', '*xlsx'))

    DRAG = pd.DataFrame()
    for dir in drag_dir:
        df = pd.read_excel(dir)
        DRAG = pd.concat([DRAG,df],axis=0)

    DRAG['chart_name'] = DRAG['brand'] + " | " + DRAG['model_name']+  " | " + DRAG['test_date'].dt.strftime('%Y-%m-%d')
    
    return DRAG
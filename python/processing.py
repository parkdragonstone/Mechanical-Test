from io import StringIO
import pandas as pd
from collections import defaultdict
import numpy as np


def convert_to_float(val):
    try:
        return float(val)
    except ValueError:
        return val

def custom_mean(column):
    if column.dtype == 'object':
        # Object 타입이면 첫 번째 값만 반환
        return column.iloc[0]
    else:
        # 숫자형 타입이면 평균을 반환
        return column.mean()

def custom_std(column):
    if column.dtype == 'object':
        # Object 타입이면 첫 번째 값만 반환
        return column.iloc[0]
    else:
        # 숫자형 타입이면 평균을 반환
        return column.std()
    
def data_load(data_dir):
    DATA = {}
    for f in data_dir:
        if 'Series' in f:
            # 파일을 텍스트 형식으로 읽기
            with open(f, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 'Time' 행 이후의 데이터만 추출
            start_index = next(i for i, line in enumerate(lines) if 'Time' in line)
            data_subset = lines[start_index:]
        
            # 리스트를 문자열로 결합
            data_str = ''.join(data_subset)

            # StringIO를 사용해 데이터프레임으로 변환
            df = pd.read_csv(StringIO(data_str), delimiter='\t')

            DATA[f] = df
        
        elif 'Table' in f:
            # 파일을 텍스트 형식으로 읽기
            with open(f, 'r', encoding='cp949') as file:
                lines = file.readlines()

            # 'Time' 행 이후의 데이터만 추출
            start_index = next(i for i, line in enumerate(lines) if 'Trial' in line)
            end_index = next(i for i, line in enumerate(lines) if 'Ave' in line)
            data_subset = lines[start_index:end_index]
            # 리스트를 문자열로 결합
            data_str = ''.join(data_subset)

            # StringIO를 사용해 데이터프레임으로 변환
            df = pd.read_csv(StringIO(data_str), delimiter='\t')
            df = df[~df['Trial'].str.contains('=+', na=False)]
            
            DATA[f] = df
            
    return DATA

def filter_files(file_list, keyword1, keyword2, limit=2):
    """ 주어진 파일 목록에서 keyword1, keyword2를 포함하는 파일들을 필터링하고 최대 limit만 남긴다. """
    filtered = [file for file in file_list if keyword1 in file and keyword2 in file]
    return filtered[:limit]  # 최대 limit 개수만큼 반환

def chart_folder_to_files(label, unique_folder_names, data_dir):
    # chart_name과 폴더 이름 및 원본 폴더 경로를 매핑하는 사전 생성
    folder_to_chart = {}
    chart_to_files = {}

    # 폴더명에서 숫자 부분 추출 및 chart_name과 매칭
    for folder in unique_folder_names:
        folder_num = folder.split('.')[0]
        
        # 매칭되는 chart_name을 찾기
        matching_row = label[label['shoe_num'].astype(str) == folder_num]
        
        if not matching_row.empty:
            chart_name = matching_row['chart_name'].values[0]
            folder_to_chart[chart_name] = folder
            
            # 해당 폴더 내의 모든 파일 경로를 리스트로 가져오기
            # 여기서는 폴더 경로를 사용해 모든 파일을 검색
            files_in_folder = [i for i in data_dir if folder in i]
            chart_to_files[chart_name] = files_in_folder, matching_row
    
    return folder_to_chart, chart_to_files

def process_files(selected_file):
    final_files = defaultdict(list)
    
    for key, files in selected_file.items():
        # RF Series
        rf_series = filter_files(files, 'RF', 'Series')
        # FF Series
        ff_series = filter_files(files, 'FF', 'Series')
        # RF Table
        rf_table = filter_files(files, 'RF', 'Table')
        # FF Table
        ff_table = filter_files(files, 'FF', 'Table')
        
        
        # 최종 리스트에 추가
        final_files[key].extend(rf_series)
        final_files[key].extend(ff_series)
        final_files[key].extend(rf_table)
        final_files[key].extend(ff_table)
    
    return dict(final_files)

# 빈 데이터프레임을 생성하는 함수
def empty_dataframe():
    return pd.DataFrame()

# 평균을 계산하는 함수 (없으면 빈 데이터프레임 반환)
def load_and_average(files, series_type, table_type, DATA, label):
        
    rf_series = [DATA.get(f, empty_dataframe()) for f in files if 'RF' in f and series_type in f]
    ff_series = [DATA.get(f, empty_dataframe()) for f in files if 'FF' in f and series_type in f]
    
    rf_table = [DATA.get(f, empty_dataframe()) for f in files if 'RF' in f and table_type in f]
    ff_table = [DATA.get(f, empty_dataframe()) for f in files if 'FF' in f and table_type in f]
    
    # 평균 계산 (Series 데이터는 두 개 이상인 경우 평균)
    rf_series_mean = sum(rf_series) / len(rf_series) if rf_series else empty_dataframe()
    ff_series_mean = sum(ff_series) / len(ff_series) if ff_series else empty_dataframe()
    
    # Table 데이터는 평균이 아닌 원본 데이터프레임을 가져옴
    rf_table_data = pd.concat(rf_table) if rf_table else empty_dataframe()
    ff_table_data = pd.concat(ff_table) if ff_table else empty_dataframe()

    # 테이블 데이터에서 Force (N/mm) 계산
    if not rf_table_data.empty:
        rf_table_data['Force (N/mm)'] = rf_table_data['Force (N)'] / rf_table_data['Max Pen (mm)']
    if not ff_table_data.empty:
        ff_table_data['Force (N/mm)'] = ff_table_data['Force (N)'] / ff_table_data['Max Pen (mm)']

    return rf_series_mean, ff_series_mean, rf_table_data.round(2), ff_table_data.round(2)



def filtering_data(filtered_selected_file, all_labels, DATA):
    filterd_data = {}
    # filtered_selected_file에 저장된 key 값과 파일 경로를 처리
    for key, files in filtered_selected_file.items():
        label = all_labels[key]
        # Series 데이터와 Table 데이터를 처리
        rf_series_avg, ff_series_avg, rf_table_data, ff_table_data = load_and_average(files, 'Series', 'Table',DATA, label)
        
        # 결과를 filterd_data에 저장
        filterd_data[key] = {
            'rf_series': rf_series_avg,
            'ff_series': ff_series_avg,
            'rf_table': rf_table_data,
            'ff_table': ff_table_data
        }
    
    return filterd_data


table_use_cols = [
    'Force (N)','Force (N/mm)','E Ret (%)',
    'Accel (g)','Max Pen (%)', 'Max Pen (mm)','Dwell T (ms)'
    ]

label_cols = [
    'test_date','test','brand','model_name','chart_name',
    'requested_by', 'requester_name','season',
    'size_mm','humidity','temperature',
    'shoe_mass','thickness_ff','thickness_rf',
    'missile_head_type','missile_head_diam_mm',
    'missile_mass_kg','drop_height_mm',
    'motor_speed']

rename_cols = ['Test Date','TEST', 'BRAND',
       'MODEL', 'chart_name', 'Requested By', 'Requester Name', 'Season',
       'Size', 'Humidity (%)', 'Temperature (℃)', 'Mass (g)', 'Thickness ForeFoot (mm)',
       'Thickness RearFoot (mm)', 'Missile Head Type', 'Missile Head Diameter (mm)',
       'Missile Mass (kg)', 'Drop Height (mm)', 'Motor Speed (%)', 
       'Force (N)', 'Force (N/mm)', 'Energy Return (%)', 'Accel (g)', 
       'Max Pen (%)','Max Pen (mm)', 'Dwell T (ms)']

def add_slope_data(all_filtered_data, all_labels):
    for trial in all_filtered_data:
        for place in ['rf_series','ff_series']:
            data = all_filtered_data[trial][place]
            category = all_labels[trial]['category'].values

            if data.empty:
                continue
            
            else:
                if category == 'TENNIS':
                    if place == 'rf_series':
                        # 빨간색 구간 정의
                        ds1_start = 250  # 첫 번째 빨간색 구간 시작 Force 값
                        ds1_end = 400    # 첫 번째 빨간색 구간 끝 Force 값
                        ds2_start = 600  # 두 번째 빨간색 구간 시작 Force 값
                        ds2_end = 750   # 두 번째 빨간색 구간 끝 Force 값
                    else:
                        ds1_start = 400  # 첫 번째 빨간색 구간 시작 Force 값
                        ds1_end = 600    # 첫 번째 빨간색 구간 끝 Force 값
                        ds2_start = 900  # 두 번째 빨간색 구간 시작 Force 값
                        ds2_end = 1100   # 두 번째 빨간색 구간 끝 Force 값
                        
                elif category == 'SPECIMEN':
                    if place == 'rf_series':
                        # 빨간색 구간 정의
                        ds1_start = 250  # 첫 번째 빨간색 구간 시작 Force 값
                        ds1_end = 400    # 첫 번째 빨간색 구간 끝 Force 값
                        ds2_start = 600  # 두 번째 빨간색 구간 시작 Force 값
                        ds2_end = 750   # 두 번째 빨간색 구간 끝 Force 값
                    else:
                        ds1_start = 250  # 첫 번째 빨간색 구간 시작 Force 값
                        ds1_end = 400    # 첫 번째 빨간색 구간 끝 Force 값
                        ds2_start = 600  # 두 번째 빨간색 구간 시작 Force 값
                        ds2_end = 750   # 두 번째 빨간색 구간 끝 Force 값

                else:
                    if place == 'rf_series':
                        # 빨간색 구간 정의
                        ds1_start = 350  # 첫 번째 빨간색 구간 시작 Force 값
                        ds1_end = 500    # 첫 번째 빨간색 구간 끝 Force 값
                        ds2_start = 750  # 두 번째 빨간색 구간 시작 Force 값
                        ds2_end = 900   # 두 번째 빨간색 구간 끝 Force 값
                    else:
                        ds1_start = 400  # 첫 번째 빨간색 구간 시작 Force 값
                        ds1_end = 600    # 첫 번째 빨간색 구간 끝 Force 값
                        ds2_start = 900  # 두 번째 빨간색 구간 시작 Force 값
                        ds2_end = 1100   # 두 번째 빨간색 구간 끝 Force 값
                        
                force_max_idx = np.where(data['Force (N)'] == data['Force (N)'].max())[0][0]

                # 각 구간별 데이터 선택
                ds1_data = data[(data['Force (N)'] >= ds1_start-30) & (data['Force (N)'] <= ds1_end+30)]
                ds2_data = data[(data['Force (N)'] >= ds2_start-30) & (data['Force (N)'] <= ds2_end+30)]

                # 첫 번째 구간 (ds1)
                ds1_upward = ds1_data.loc[:force_max_idx]  # 올라가는 부분
                ds1_downward = ds1_data.loc[force_max_idx:] # 내려가는 부분

                ds2_upward = ds2_data.loc[:force_max_idx]  # 올라가는 부분
                ds2_downward = ds2_data.loc[force_max_idx:]  # 내려가는 부분

                # 가장 가까운 시작과 끝 지점의 인덱스를 찾는 함수
                def get_closest_index(data, target):
                    return (data['Force (N)'] - target).abs().idxmin()

                # ds1
                ds1_start_idx = get_closest_index(ds1_upward, ds1_start)
                ds1_end_idx = get_closest_index(ds1_upward, ds1_end)
                ds1_upward = ds1_upward.loc[min(ds1_start_idx, ds1_end_idx):max(ds1_start_idx, ds1_end_idx)]

                ds1_start_idx = get_closest_index(ds1_downward, ds1_start)
                ds1_end_idx = get_closest_index(ds1_downward, ds1_end)
                ds1_downward = ds1_downward.loc[min(ds1_start_idx, ds1_end_idx):max(ds1_start_idx, ds1_end_idx)]

                if ds2_data.empty:
                    ds2_upward_slope, ds2_upward_f1, ds2_upward_f2, ds2_upward_d1, ds2_upward_d2 = 0,0,0,0,0
                    ds2_downward_slope, ds2_downward_f1, ds2_downward_f2, ds2_downward_d1, ds2_downward_d2 = 0,0,0,0,0
                
                else:
                    # ds2
                    ds2_start_idx = get_closest_index(ds2_upward, ds2_start)
                    ds2_end_idx = get_closest_index(ds2_upward, ds2_end)
                    ds2_upward = ds2_upward.loc[min(ds2_start_idx, ds2_end_idx):max(ds2_start_idx, ds2_end_idx)]


                    ds2_start_idx = get_closest_index(ds2_downward, ds2_start)
                    ds2_end_idx = get_closest_index(ds2_downward, ds2_end)
                    ds2_downward = ds2_downward.loc[min(ds2_start_idx, ds2_end_idx):max(ds2_start_idx, ds2_end_idx)]


                # 각 부분별 기울기 계산 함수
                def calculate_slope(data):
                    if len(data) > 1:
                        return (data['Force (N)'].iloc[-1] - data['Force (N)'].iloc[0]) / (data['Disp (mm)'].iloc[-1] - data['Disp (mm)'].iloc[0]), data['Force (N)'].iloc[0], data['Force (N)'].iloc[-1], data['Disp (mm)'].iloc[0], data['Disp (mm)'].iloc[-1]
                    return 0, 0, 0, 0, 0 # 데이터가 부족한 경우 0으로 설정

                # 기울기 계산
                ds1_upward_slope, ds1_upward_f1, ds1_upward_f2, ds1_upward_d1, ds1_upward_d2 = calculate_slope(ds1_upward)
                ds1_downward_slope, ds1_downward_f1, ds1_downward_f2, ds1_downward_d1, ds1_downward_d2 = calculate_slope(ds1_downward)
                ds2_upward_slope, ds2_upward_f1, ds2_upward_f2, ds2_upward_d1, ds2_upward_d2 = calculate_slope(ds2_upward)
                ds2_downward_slope, ds2_downward_f1, ds2_downward_f2, ds2_downward_d1, ds2_downward_d2 = calculate_slope(ds2_downward)

                if data['Force (N)'].max() < ds2_end:
                    ds2_upward_slope, ds2_upward_f1, ds2_upward_f2, ds2_upward_d1, ds2_upward_d2 = 0,0,0,0,0
                    ds2_downward_slope, ds2_downward_f1, ds2_downward_f2, ds2_downward_d1, ds2_downward_d2 = 0,0,0,0,0


                made_cols = [
                    'ds1_upward_slope'  , 'ds1_upward_f1'  , 'ds1_upward_f2'  , 'ds1_upward_d1'  ,'ds1_upward_d2',
                    'ds1_downward_slope', 'ds1_downward_f2', 'ds1_downward_f1', 'ds1_downward_d2','ds1_downward_d1',
                    'ds2_upward_slope'  , 'ds2_upward_f1'  , 'ds2_upward_f2'  , 'ds2_upward_d1'  , 'ds2_upward_d2',
                    'ds2_downward_slope', 'ds2_downward_f2', 'ds2_downward_f1', 'ds2_downward_d2', 'ds2_downward_d1'
                    ]

                made_data = [
                    ds1_upward_slope  , ds1_upward_f1  , ds1_upward_f2  , ds1_upward_d1  , ds1_upward_d2,
                    ds1_downward_slope, ds1_downward_f2, ds1_downward_f1, ds1_downward_d2, ds1_downward_d1,
                    ds2_upward_slope  , ds2_upward_f1  , ds2_upward_f2  , ds2_upward_d1  , ds2_upward_d2,
                    ds2_downward_slope, ds2_downward_f2, ds2_downward_f1, ds2_downward_d2, ds2_downward_d1
                ]

                for col, da in zip(made_cols, made_data):
                    data[col] = da

                all_filtered_data[trial][place] = data
    
    return all_filtered_data
        
def overall_data_table(all_chart_names, all_filtered_data, all_labels):
    TABLE_RF = pd.DataFrame(columns = rename_cols)
    TABLE_FF = pd.DataFrame(columns = rename_cols)

    TABLE_RF_SD = pd.DataFrame(columns = rename_cols)
    TABLE_FF_SD = pd.DataFrame(columns = rename_cols)
    # 비어있는 경우 사용할 빈 데이터프레임
    empty_table = pd.DataFrame(columns=table_use_cols)
    empty_label = pd.DataFrame(columns=label_cols)
    # empty_label2 = pd.DataFrame(columns=label2_cols)

    for f in all_chart_names:
        
        label_table = all_labels[f][label_cols].reset_index(drop=True) if not all_labels[f].empty else empty_label
        
        ff_table = pd.DataFrame(all_filtered_data[f]['ff_table'][table_use_cols].apply(custom_mean)).T if not all_filtered_data[f]['ff_table'].empty else empty_table
        rf_table = pd.DataFrame(all_filtered_data[f]['rf_table'][table_use_cols].apply(custom_mean)).T if not all_filtered_data[f]['rf_table'].empty else empty_table
        for col in ff_table.columns:
            ff_table[col] = ff_table[col].apply(convert_to_float)
        for col in rf_table.columns:
            rf_table[col] = rf_table[col].apply(convert_to_float)
                
        table_ff = pd.concat([label_table, ff_table],axis=1)
        table_rf = pd.concat([label_table, rf_table],axis=1)
        table_ff.columns = rename_cols
        table_rf.columns = rename_cols
        TABLE_FF = pd.concat([TABLE_FF, table_ff], axis=0) 
        TABLE_RF = pd.concat([TABLE_RF, table_rf], axis=0)
        
        ff_table_sd = pd.DataFrame(all_filtered_data[f]['ff_table'][table_use_cols].apply(custom_std)).T if not all_filtered_data[f]['ff_table'].empty else empty_table
        rf_table_sd = pd.DataFrame(all_filtered_data[f]['rf_table'][table_use_cols].apply(custom_std)).T if not all_filtered_data[f]['rf_table'].empty else empty_table
        for col in ff_table_sd.columns:
            ff_table_sd[col] = ff_table_sd[col].apply(convert_to_float)
        for col in rf_table_sd.columns:
            rf_table_sd[col] = rf_table_sd[col].apply(convert_to_float)
            
        table_ff_sd = pd.concat([label_table,ff_table_sd],axis=1)
        table_rf_sd = pd.concat([label_table,rf_table_sd],axis=1)
        table_ff_sd.columns = rename_cols
        table_rf_sd.columns = rename_cols
        TABLE_FF_SD = pd.concat([TABLE_FF_SD, table_ff_sd], axis=0) 
        TABLE_RF_SD = pd.concat([TABLE_RF_SD, table_rf_sd], axis=0)

    FF = TABLE_FF.set_index(['BRAND','MODEL'])
    RF = TABLE_RF.set_index(['BRAND','MODEL'])

    FF_SD = TABLE_FF_SD.set_index(['BRAND','MODEL'])
    RF_SD = TABLE_RF_SD.set_index(['BRAND','MODEL'])
    
    return FF, RF, FF_SD.round(2), RF_SD.round(2)
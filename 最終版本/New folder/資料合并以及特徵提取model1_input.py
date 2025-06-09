import pandas as pd
import openpyxl
from joblib import Parallel, delayed
import gc 

# 读取数据
file_path1 = '105年A1-A4所有當事人.xlsx'
file_path2 = '106年A1-A4所有當事人.xlsx'
file_path3 = '107年A1-A4所有當事人(新增戶籍地).xlsx'
file_path4 = '108年A1-A4所有當事人(新增戶籍地).xlsx'
file_path5 = '109年A1-A4所有當事人(新增戶籍地).xlsx'
file_paths = [file_path1, file_path2, file_path3, file_path4, file_path5]#依照需求刪除增加

# 定义读取和处理单个文件的函数
def process_file(file_path):
    try:
        wb = openpyxl.load_workbook(file_path)
        sheet = wb.worksheets[1] if len(wb.worksheets) > 1 else wb.worksheets[0]
        data = sheet.values
        columns = next(data)
        df = pd.DataFrame(data, columns=columns)
        
        # 定义函数来分类昼夜
        def categorize_day_night(day_night):
            return 0 if day_night == '夜' else 1

        # 特征基本处理
        df['date'] = pd.to_datetime(df['發生時間'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['quarter'] = df['date'].dt.quarter
        df['day_night_category'] = df['晝夜'].apply(categorize_day_night)
        #删除没有答案的数据
        #df = df.dropna(subset=['肇因碼-個別'])
        # 将案號列转换为相应的数字
        #df['區'], _ = pd.factorize(df['區'])
        df['district_code'] = df['區序'].str[:2]
        df['案號'], _ = pd.factorize(df['案號'])
        df['車種'], _ = pd.factorize(df['車種'])
        
        # 所有可能的特征
        columns_to_select = [
            'year', 'month', 'day', 'hour', 'quarter', 'day_night_category', 'X', 'Y','district_code','4天候', 
            '5光線', '6道路類別', '7速限', '8道路型態', '9事故位置',  '15事故類型及型態', 
            '外國人', '性別', '處理別', '案號', '死亡人數', '2-30日死亡人數', '受傷人數',
            '當事人序', '車種', '10路面狀況1', '10路面狀況2', '10路面狀況3', 
            '11道路障礙1', '11道路障礙2', '12號誌1', '12號誌2', '13車道劃分-分向', 
            '14車道劃分-分道1', '14車道劃分-分道2', '14車道劃分-分道3', '重大車損', 
            '年齡', '22受傷程度', '23主要傷處', '25行動電話', '28車輛用途', '29當事者行動狀態', 
            '30駕駛資格情形', '31駕駛執照種類', '32飲酒情形', '33_1主要車損', 
            '35個人肇逃否', '36職業', '37旅次目的','肇因碼-個別', '肇因碼-主要'
        ]

        # 删除不需要的列
        df = df[columns_to_select]

        # 重命名列
        new_columns = [
            'year', 'month', 'day', 'hour', 'quarter', 'day_night_category', 'X', 'Y', 'district_code','weather', 
            'light', 'road_type', 'speed_limit', 'road_form', 'accident_location', 'accident_type', 
            'foreigner', 'gender', 'processing_type', 'case_number', 
            'death_count', 'death_within_2_30_days', 'injury_count', 'party_sequence', 'vehicle_type', 
            'road_condition_1', 'road_condition_2', 'road_condition_3', 
            'road_obstacle_1', 'road_obstacle_2', 'signal_1', 'signal_2', 'lane_division_direction', 
            'lane_division_type_1', 'lane_division_type_2', 'lane_division_type_3', 'major_vehicle_damage', 
            'age', 'injury_severity', 'main_injury_part', 'mobile_phone', 'vehicle_purpose', 
            'party_action_status', 'driving_license_status', 'driving_license_type', 'alcohol_status', 
            'major_vehicle_damage_1', 'hit_and_run', 'occupation', 
            'trip_purpose','cause_code_individual','cause_code_main'
        ]
        
        df.columns = new_columns
        
        df = df.apply(pd.to_numeric, errors='coerce')
        
        df.dropna(axis=0, how='any', inplace=True)
        # 删除引用并显式调用垃圾回收
        gc.collect()

        return df
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return pd.DataFrame()  # 返回空 DataFrame 以避免合并时报错

# 并行读取和处理所有文件
dfs = Parallel(n_jobs=-1)(delayed(process_file)(file_path) for file_path in file_paths)

# 合并所有 DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

#new_df.dropna(axis=0, how='any', inplace=True)

# 删除包含 NaN 值的行
combined_df.dropna(axis=0, how='any', inplace=True)

# 保存到 CSV 文件
output_file_path = 'STEP1.csv'
combined_df.to_csv(output_file_path, index=False)

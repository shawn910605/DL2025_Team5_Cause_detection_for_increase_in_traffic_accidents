import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# 读取CSV文件
file_path = 'STEP1.csv'
new_df = pd.read_csv(file_path)
print('Data load -> OK')

# 保留指定的30个特征列（由step1.5得出）
columns_to_select = [
    'year', 'month', 'day', 'hour', 'quarter', 'day_night_category', 'X', 'Y', 
    'weather', 
    'light', 'speed_limit', 'road_form', 'accident_location', 'accident_type', 'gender', 
    'party_sequence', 'cause_code_main', 'lane_division_direction', 'lane_division_type_1', 
    'age', 'main_injury_part', 'alcohol_status', 'injury_count', 'driving_license_type', 
    'major_vehicle_damage_1', 'signal_1', 'signal_2', 'district', 
    'party_action_status'
]
new_df = new_df[columns_to_select + ['cause_code_individual']]

# 删除包含 NaN 值的行
new_df.dropna(axis=0, how='any', inplace=True)

#删除原因不明的结果（对准确率影响不大）
#new_df = new_df.apply(pd.to_numeric, errors='coerce')
#values_to_remove = [43, 44, 67]
#new_df = new_df[~new_df['cause_code_individual'].isin(values_to_remove)]

# 对非数值型特征进行 one-hot 编码
#categorical_columns = [ 'quarter','cause_code_main','day_night_category', 'weather', 'light', 'road_form', 'accident_location', 
#                       'accident_type', 'gender', 'lane_division_direction', 'lane_division_type_1', 
#                       'main_injury_part', 'alcohol_status', 'driving_license_type', 'signal_1', 
#                       'signal_2', 'district', 'party_action_status','major_vehicle_damage_1']
#new_df = pd.get_dummies(new_df, columns=categorical_columns)

new_df = new_df.apply(pd.to_numeric, errors='coerce')
new_df.dropna(axis=0, how='any', inplace=True)
print(new_df.head())

# 保存到 CSV 文件
output_file_path = 'STEP2V2.csv'
new_df.to_csv(output_file_path, index=False)

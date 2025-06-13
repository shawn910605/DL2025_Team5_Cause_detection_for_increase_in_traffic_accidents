import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 读取原始 STEP1.csv
file_path = '../STEP1/STEP1.csv'
new_df = pd.read_csv(file_path, low_memory=False)
print('Data load -> OK')

# 2. 只保留 STEP1.5 选出的特征 + 目标
columns_to_select = [
    'death_count', 'injury_severity', 'Y', 'signal_2', 'foreigner',
    'road_type', 'road_condition_2', 'month', 'road_obstacle_1',
    'accident_location', 'district', 'driving_license_status',
    'road_form', 'other_vehicle_damage', 'party_sequence', 'signal_1',
    'injury_count', 'X', 'processing_type', 'day', 'weather',
    'accident_type', 'lane_division_type_2', 'main_injury_part',
    'driving_license_type', 'major_vehicle_damage', 'light',
    'party_action_status', 'death_within_2_30_days',
    'day_night_category', 'lane_division_type_1', 'major_vehicle_damage_1',
    'vehicle_type', 'age', 'vehicle_purpose', 'alcohol_status',
    'year', 'speed_limit', 'lane_division_direction', 'gender',
    'road_obstacle_2', 'road_condition_1', 'lane_division_type_3',
    'hour', 'hit_and_run', 'cause_code_main', 'quarter',
    'road_condition_3'
]
# 过滤不存在的列
available = [c for c in columns_to_select if c in new_df.columns]
missing   = [c for c in columns_to_select if c not in new_df.columns]
if missing:
    print("⚠️ 以下字段不存在，已自动跳过：", missing)
new_df = new_df[available + ['cause_code_individual']]

# 3. 删除目标缺失，其余特征以 0 填补
new_df = new_df.dropna(subset=['cause_code_individual'])
new_df = new_df.fillna(0)

# 4. 对分类特征做 one-hot（仅针对 STEP1.5 中存在的）
categorical_columns = [
    'quarter', 'cause_code_main', 'day_night_category', 'weather',
    'light', 'road_type', 'accident_location', 'accident_type',
    'gender', 'lane_division_direction', 'lane_division_type_1',
    'main_injury_part', 'alcohol_status', 'driving_license_type',
    'signal_1', 'signal_2', 'district', 'party_action_status',
    'major_vehicle_damage_1'
]
to_dummy = [c for c in categorical_columns if c in new_df.columns]
new_df = pd.get_dummies(new_df, columns=to_dummy, drop_first=False)

# 5. 强制转数值并再次填 0
new_df = new_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# 6. 打印处理后形状
print("Prepared features shape:", new_df.shape)

# 7. 切分特征与标签
X = new_df.drop(columns=['cause_code_individual']).values
y = new_df['cause_code_individual'].values

# 8. 切分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 10. 转为 Tensor 并搬到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_t = torch.tensor(LabelEncoder().fit_transform(y_train), dtype=torch.long).to(device)
y_test_t  = torch.tensor(LabelEncoder().fit_transform(y_test), dtype=torch.long).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=100, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                          batch_size=100, shuffle=False)

print(f"✅ 前處理完成，train/test 樣本數：{X_train_t.shape[0]}/{X_test_t.shape[0]}")

# 11. 输出最终 CSV，与原脚本保持一致
output_file_path = 'STEP2_V2.csv'
new_df.to_csv(output_file_path, index=False)
print("✅ 已輸出 STEP2_V2.csv")

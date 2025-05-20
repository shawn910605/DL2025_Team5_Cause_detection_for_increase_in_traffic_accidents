import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'STEP1.csv'
new_df = pd.read_csv(file_path)
print('data load -> ok')

# 删除包含 NaN 值的行
new_df.dropna(axis=0, how='any', inplace=True)

# 对 case_number 和 vehicle_type 进行 one-hot 编码
new_df = pd.get_dummies(new_df, columns=['case_number', 'vehicle_type'])

new_df = new_df.apply(pd.to_numeric, errors='coerce')
new_df.dropna(axis=0, how='any', inplace=True)
print(new_df.head())

# 数据预处理
# 将 DataFrame 转换为 numpy 数组
X = new_df.drop(columns=['cause_code_individual']).values
y = new_df['cause_code_individual'].values

# 处理标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用随机森林进行特征重要性分析
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 评估模型
train_accuracy = rf.score(X_train, y_train)
test_accuracy = rf.score(X_test, y_test)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')


# 获取特征重要性
feature_importances = rf.feature_importances_

# 获取特征名称
feature_names = new_df.drop(columns=['cause_code_individual']).columns

# 创建 DataFrame 用于可视化
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 打印最重要的特征
print(importance_df.head(30))

# 绘制特征重要性图表
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.barh(importance_df['Feature'].head(30), importance_df['Importance'].head(30))
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

# 保留前20个重要特征
top_features = importance_df['Feature'].head(5).values
X_train_top = X_train[:, importance_df['Feature'].head(5).index]
X_test_top = X_test[:, importance_df['Feature'].head(5).index]

# 重新训练模型
rf_top = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top.fit(X_train_top, y_train)

# 评估模型
train_accuracy = rf_top.score(X_train_top, y_train)
test_accuracy = rf_top.score(X_test_top, y_test)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

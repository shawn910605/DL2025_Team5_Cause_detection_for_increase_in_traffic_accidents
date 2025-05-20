import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 读取Excel文件的第二个工作表
file_path = 'step2.csv'
new_df = pd.read_csv(file_path)

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

# 创建并训练SVM模型
svm_model = SVC(kernel='rbf', gamma='auto') 
svm_model.fit(X_train, y_train)

# 预测并评估模型
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# 打印分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))


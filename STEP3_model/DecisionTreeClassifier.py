import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import shap

# 超参数配置区块
class Config:
    num_epochs = 15
    batch_size = 100

# 读取CSV文件
file_path = 'STEP2V2.csv'
new_df = pd.read_csv(file_path)
print('Data load -> OK')

# 数据预处理
X = new_df.drop(columns=['cause_code_individual']).values
y = new_df['cause_code_individual'].values
unique_labels = np.unique(y)
num_classes = len(unique_labels)
print("Number of unique labels:", num_classes)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义决策树模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 预测与评估
train_accuracy = model.score(X_train, y_train)
val_accuracy = model.score(X_test, y_test)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

# 决策树可视化
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=new_df.drop(columns=['cause_code_individual']).columns, class_names=list(map(str, label_encoder.classes_)), filled=True)
plt.title('Decision Tree Visualization')
plt.show()

# SHAP解释部分
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 可视化SHAP summary plot
shap.summary_plot(shap_values, X_test)

# 可视化第一个测试样本的SHAP force plot
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt

# 读取step3.csv文件
df = pd.read_csv('model2.csv')

# 設置日期範圍
test_start_date = '2020-01-01'
train_end_date = '2019-12-31'

# 提取特征和目标变量
X_train = df[df['date'] <= train_end_date].drop(columns=['date', 'total'])
X_test = df[df['date'] >= test_start_date].drop(columns=['date', 'total'])
y_train = df[df['date'] <= train_end_date]['total'].diff().fillna(0)  # 車禍增長量，空值填充為0
y_test = df[df['date'] >= test_start_date]['total'].diff().fillna(0)  # 車禍增長量，空值填充為0
y_train_binary = (y_train > 0).astype(int)  # 二元分類問題：車禍是否增長
y_test_binary = (y_test > 0).astype(int)  # 二元分類問題：車禍是否增長

print("Data preparation complete.")

# 定义解释模型
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 使用解释模型训练
model = create_model()
history = model.fit(X_train, y_train_binary, epochs=100, batch_size=50, validation_data=(X_test, y_test_binary), verbose=0)

# 绘制训练准确率和验证准确率的图表
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# 绘制训练损失和验证损失的图表
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 训练准确率
_, train_accuracy = model.evaluate(X_train, y_train_binary)
print("训练准确率:", train_accuracy)

# 测试准确率
_, test_accuracy = model.evaluate(X_test, y_test_binary)
print("测试准确率:", test_accuracy)

# 使用 SHAP 解释模型
explainer = shap.Explainer(model, X_train)
shap_values_train = explainer.shap_values(X_train)

print("SHAP values computation complete.")

# 绘制每个特征的 SHAP 值总结图
shap.summary_plot(shap_values_train, X_train, feature_names=X_train.columns, plot_type="bar")

print("SHAP summary plot (bar) complete.")

# 绘制 SHAP 特征重要性图
shap.summary_plot(shap_values_train, X_train)

print("SHAP summary plot complete.")

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
# 超参数配置区块
class Config:
    input_size = 314 #279  # 输入层大小
    num_classes = 67 #64  # 输出类别数
    hidden_sizes = [512, 1024, 256, 128]  # 隐藏层大小
    learning_rate = 0.001  # 学习率
    momentum = 0.9  # 动量
    num_epochs = 15  # 迭代次数
    batch_size = 100  # 批量大小
    dropout_prob = 0.5  # Dropout 概率

# 读取CSV文件
#file_path = 'STEP2V2.csv'
file_path = 'STEP2V2.csv'
new_df = pd.read_csv(file_path)
print('Data load -> OK')

# 数据预处理
# 将 DataFrame 转换为 numpy 数组
X = new_df.drop(columns=['cause_code_individual']).values
y = new_df['cause_code_individual'].values
unique_labels = np.unique(y)
num_classes = len(unique_labels)
print("Number of unique labels:", num_classes)


# 处理标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 将数据移动到GPU
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

# 定义全连接神经网络模型
class FCNNModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes):
        super(FCNNModel, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip([input_size] + hidden_sizes, hidden_sizes):
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.dropout = nn.Dropout(Config.dropout_prob)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = FCNNModel(Config.input_size, Config.num_classes, Config.hidden_sizes).to(device)

# 定义优化器和损失函数
#optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate, momentum=Config.momentum)

optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
criterion = nn.CrossEntropyLoss()


# 训练模型
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(Config.num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # 创建一个进度条
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}, Training")
    
    for inputs, labels in train_progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_progress_bar.set_postfix(train_loss=running_loss / total, train_accuracy=correct / total)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    
    # 创建一个进度条
    val_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}, Validation")
    
    with torch.no_grad():
        for inputs, labels in val_progress_bar:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_progress_bar.set_postfix(val_loss=val_loss / total, val_accuracy=correct / total)
    
    val_loss /= len(test_loader.dataset)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    scheduler.step()
# 绘制训练和验证的损失以及准确性图表
plt.figure(figsize=(12, 4))

# 绘制损失图表
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确性图表
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


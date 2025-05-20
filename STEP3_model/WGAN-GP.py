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
    input_size = 314  # 输入层大小
    latent_dim = 100  # 隐变量的大小
    num_classes = 67  # 输出类别数
    learning_rate = 0.0001  # 学习率
    num_epochs = 3  # 迭代次数
    batch_size = 100  # 批量大小
    lambda_gp = 7  # 梯度惩罚项权重

# 读取CSV文件
file_path = 'STEP1.csv'
new_df = pd.read_csv(file_path)
print('Data load -> OK')

# 保留指定的30个特征列
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

# 对非数值型特征进行 one-hot 编码
categorical_columns = [ 'quarter','cause_code_main','day_night_category', 'weather', 'light', 'road_form', 'accident_location', 
                       'accident_type', 'gender', 'lane_division_direction', 'lane_division_type_1', 
                       'main_injury_part', 'alcohol_status', 'driving_license_type', 'signal_1', 
                       'signal_2', 'district', 'party_action_status','major_vehicle_damage_1']
new_df = pd.get_dummies(new_df, columns=categorical_columns)

new_df = new_df.apply(pd.to_numeric, errors='coerce')
new_df.dropna(axis=0, how='any', inplace=True)
#print(new_df.head())

# 数据预处理
# 将 DataFrame 转换为 numpy 数组
X = new_df.drop(columns=['cause_code_individual']).values
y = new_df['cause_code_individual'].values
unique_labels = np.unique(y)
num_classes = len(unique_labels)
#print("Number of unique labels:", num_classes)

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

class Generator(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_size),  # 修改输出层为输出67维向量
            nn.Softmax(dim=1)  # 添加softmax激活函数，确保输出是归一化的概率分布
        )

    def forward(self, z):
        output = self.model(z)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # 修改输出层为输出单一节点
            nn.Sigmoid()  # 添加sigmoid激活函数，将输出值压缩到0到1之间
        )

    def forward(self, x):
        output = self.model(x)
        return output


# 实例化生成器和判别器
generator = Generator(Config.latent_dim, Config.input_size).to(device)
discriminator = Discriminator(Config.input_size).to(device)

# 定义优化器
optimizer_G = optim.Adam(generator.parameters(), lr=Config.learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=Config.learning_rate, betas=(0.5, 0.999))

# 梯度惩罚函数
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练WGAN-GP
G_losses, D_losses = [], []

for epoch in range(Config.num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):  # 修改迭代器返回的元素为输入数据和标签
        batch_size = inputs.size(0)
        real_data = inputs.to(device)
        real_labels = labels.to(device)  # 将标签移动到相同的设备
        
        # 训练判别器
        optimizer_D.zero_grad()
        
        # 使用生成的数据
        z = torch.randn(batch_size, Config.latent_dim).to(device)
        fake_data = generator(z)
        real_validity = discriminator(real_data)
        fake_validity = discriminator(fake_data)
        
        gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + Config.lambda_gp * gradient_penalty
        
        d_loss.backward()
        optimizer_D.step()

        # 每 5 次更新生成器一次
        if i % 10 == 0:
            optimizer_G.zero_grad()
            
            # 生成器希望判别器认为生成的数据是真实的
            fake_data = generator(z)
            fake_validity = discriminator(fake_data)
            g_loss = -torch.mean(fake_validity)
            
            g_loss.backward()
            optimizer_G.step()

        # 记录损失
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{Config.num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 保存生成器和判别器的状态字典
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

# 加载保存的生成器状态字典
generator = Generator(Config.latent_dim, Config.input_size).to(device)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()  # 设置为评估模式，不进行梯度计算

# 将生成器设置为不需要梯度计算
with torch.no_grad():
    # 生成新数据
    num_generated_samples = 100
    z = torch.randn(num_generated_samples, Config.latent_dim).to(device)
    generated_data = generator(z)

# 将生成的数据转换为numpy数组并移动到CPU上
generated_data_np = generated_data.detach().cpu().numpy()

# 创建标准化器对象
scaler = StandardScaler()
scaler.fit(X_train.cpu())  # 使用训练集数据拟合标准化器

# 将数据还原为原始数据
generated_data_original = scaler.inverse_transform(generated_data_np)

# 逆操作：将 One-Hot 编码的 DataFrame 转换回原始的非数值型特征列
original_df = new_df[categorical_columns].idxmax(axis=1).to_frame()


# 将列标题命名为原始的特征列名
original_df.columns = ['original_feature']

# 将原始的非数值型特征列添加到原始数据中
original_df = pd.concat([original_df, new_df.drop(columns=categorical_columns)], axis=1)

# 使用原始的特征列来创建DataFrame
generated_df = pd.DataFrame(generated_data_original, columns=columns_to_select[:-1])

# 将生成的假数据保存到CSV文件
generated_df.to_csv('generated_fake_data.csv', index=False)
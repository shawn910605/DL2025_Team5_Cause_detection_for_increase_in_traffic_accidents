import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# =====================================================
# 固定隨機種子，確保可重現性
# =====================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# =====================================================
# 配置參數區塊
# =====================================================
class Config:
    DATA_PATH = 'step1.csv'
    TEST_YEAR = 2020
    HIDDEN_SIZES = [512, 256]
    DROPOUT_PROB = 0.3
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LR_STEP_SIZE = 5
    LR_GAMMA = 0.5
    EARLY_STOPPING_PATIENCE = 8

# =====================================================
# 資料處理：回傳 DataLoader、測試集與標籤名稱
# =====================================================
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    train_df = df[df['year'] != Config.TEST_YEAR].copy()
    test_df  = df[df['year'] == Config.TEST_YEAR].copy()

    y_train = train_df['cause_code_individual']
    y_test  = test_df['cause_code_individual']
    X_train = train_df.drop(columns=['cause_code_individual','year'])
    X_test  = test_df.drop(columns=['cause_code_individual','year'])

    # One-Hot 編碼
    cat_cols = X_train.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=cat_cols)
    X_test  = pd.get_dummies(X_test,  columns=cat_cols)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 標籤編碼並保留名稱
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test  = encoder.transform(y_test)
    class_names = encoder.classes_

    # 轉為 Tensor 並移到設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test  = torch.tensor(y_test, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test ,y_test ), batch_size=Config.BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, X_test, y_test, class_names

# =====================================================
# 模型定義
# =====================================================
class FCNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in Config.HIDDEN_SIZES:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(Config.DROPOUT_PROB))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =====================================================
# 訓練與驗證函式
# =====================================================
def train_and_validate(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    no_improve = 0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    for epoch in range(1, Config.NUM_EPOCHS+1):
        model.train()
        running_loss, preds, labels = 0.0, [], []
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
            preds.append(out.argmax(dim=1).cpu())
            labels.append(yb.cpu())
        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = accuracy_score(torch.cat(labels), torch.cat(preds))

        model.eval()
        val_loss, vpreds, vlabels = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                out = model(Xb)
                l = criterion(out, yb)
                val_loss += l.item() * Xb.size(0)
                vpreds.append(out.argmax(dim=1).cpu())
                vlabels.append(yb.cpu())
        val_loss = val_loss / len(val_loader.dataset)
        val_acc  = accuracy_score(torch.cat(vlabels), torch.cat(vpreds))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:02d}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
        scheduler.step()

    model.load_state_dict(torch.load('best_model.pth'))
    return history

# =====================================================
# 測試與繪製混淆矩陣及歷史曲線
# =====================================================
def test_and_visualize(model, X_test, y_test, history, class_names):
    model.eval()
    with torch.no_grad():
        out = model(X_test)
    preds = out.argmax(dim=1).cpu().numpy()
    trues = y_test.cpu().numpy()

    # 輸出 Test Loss & Accuracy
    loss_val = nn.CrossEntropyLoss()(out, y_test).item()
    acc = accuracy_score(trues, preds)
    print(f"Test Loss:     {loss_val:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # 混淆矩陣
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.tight_layout()
    plt.show()

    # 訓練/驗證 Loss & Accuracy曲線
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'],   label='Val Acc')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# =====================================================
# 主程式
# =====================================================
def main():
    train_loader, test_loader, X_test, y_test, classes = load_data(Config.DATA_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCNNModel(input_dim=X_test.shape[1], num_classes=len(classes)).to(device)
    history = train_and_validate(model, train_loader, test_loader)
    test_and_visualize(model, X_test, y_test, history, classes)

if __name__ == '__main__':
    main()

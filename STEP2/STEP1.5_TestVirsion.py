import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1. 讀取並清理 STEP1.csv
path_in = '../STEP1/STEP1.csv'
df = pd.read_csv(path_in, low_memory=False)
df.dropna(axis=0, how='any', inplace=True)

# 2. 移除無意義高基數欄位
for col in ['case_number', 'mobile_phone', 'trip_purpose', 'occupation']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# 3. 分離特徵與標籤
target = 'cause_code_individual'
X_raw = df.drop(columns=[target])
y = df[target]
y_enc = LabelEncoder().fit_transform(y)

# 4. Label Encode 類別欄位
for c in X_raw.select_dtypes(include=['object']).columns:
    X_raw[c] = LabelEncoder().fit_transform(X_raw[c].astype(str))

# 5. 切分訓練/測試（僅為計算重要性用，不會輸出）
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_enc, test_size=0.2, random_state=42)

# 6. 標準化
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# 7. 隨機森林重要性
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)
imp_rf = pd.Series(rf.feature_importances_, index=X_raw.columns).sort_values(ascending=False)

# 8. L1 邏輯回歸係數
lr = LogisticRegression(
    penalty='l1', solver='liblinear', random_state=42, max_iter=1000
)
lr.fit(X_tr, y_tr)
imp_l1 = pd.Series(
    np.abs(lr.coef_).mean(axis=0),
    index=X_raw.columns
).sort_values(ascending=False)

# 9. 綜合篩選：前50取交集（若不足則聯集）
top_n = 50
set_rf = set(imp_rf.index[:top_n])
set_l1 = set(imp_l1.index[:top_n])
common = set_rf & set_l1
if len(common) < 20:
    common = set_rf | set_l1
selected = list(common)

# 10. 輸出新檔（不影響原 STEP1.csv）
path_out = '../STEP1/STEP1_5_selected.csv'
df_selected = df[selected + [target]]
df_selected.to_csv(path_out, index=False, encoding='utf-8-sig')
print("✅ 已輸出篩選後檔案：", path_out)
print("   選出的特徵如下：", selected)

# 11. 可視化前 30 特徵重要性
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
imp_rf.head(30).plot.barh(ax=axes[0], title='RF Importance (Top 30)')
imp_l1.head(30).plot.barh(ax=axes[1], title='L1 Coefs (Top 30)')
for ax in axes:
    ax.invert_yaxis()
plt.tight_layout()
plt.show()

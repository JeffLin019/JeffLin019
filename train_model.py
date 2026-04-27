import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("正在讀取資料...")
# 1. 讀取資料 (請確保 hotel_bookings.csv 與此程式碼在同一個資料夾)
df = pd.read_csv('hotel_bookings.csv')

# 2. 資料清洗與預處理
print("進行資料清洗...")
# 處理缺失值
df['children'] = df['children'].fillna(0).astype(int)
df['country'] = df['country'].fillna('Unknown')
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# 刪除不合理的值 (沒有任何客人的訂單)
zero_guests = (df['adults'] + df['children'] + df['babies'] == 0)
df = df.drop(df[zero_guests].index)

# 3. 特徵選擇 (Feature Selection)
# 為了單純貝氏模型，我們挑選最具影響力的數值型特徵
features = [
    'lead_time',                    # 提前預訂天數
    'total_of_special_requests',    # 特殊需求總數
    'required_car_parking_spaces',  # 需要的停車位數量
    'booking_changes',              # 訂單變更次數
    'previous_cancellations',       # 過往取消次數
    'is_repeated_guest'             # 是否為回頭客 (1=是, 0=否)
]

X = df[features]
y = df['is_canceled'] # 目標變數：1=取消, 0=入住

# 4. 拆分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 建立與訓練模型 (單純貝氏分類器)
print("正在訓練單純貝氏模型...")
model = GaussianNB()
model.fit(X_train, y_train)

# 6. 預測與評估
y_pred = model.predict(X_test)
print("\n--- 模型評估結果 ---")
print(f"準確度 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\n分類報告 (Classification Report):")
print(classification_report(y_test, y_pred))

# 7. 儲存模型
joblib.dump(model, 'nb_model.pkl')
print("\n模型已成功儲存為 'nb_model.pkl'！")
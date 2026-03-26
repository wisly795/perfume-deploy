


import pandas as pd
import xgboost as xgb
import joblib
import os
from sqlalchemy import create_engine  # ⭐新增：改為使用 SQLAlchemy 連線 PostgreSQL
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder  # 💡 【新增】處理類別編碼的工具

def train_perfume_model():
    print("🚀 [1/4] 讀取資料中...")
    
    # 💡 修改點：使用絕對路徑鎖定資料庫與產出路徑
    # ⭐ 修改：改為使用 Docker Volume 路徑，避免部署後模型消失
    MODEL_DIR = "/app/models_storage"
    # ⭐ 修改：確保資料夾存在（避免部署時崩潰）
    os.makedirs(MODEL_DIR, exist_ok=True)
    # ⭐ 修改：模型與 encoder 都存到持久化資料夾
    model_output_path = os.path.join(MODEL_DIR, "perfume_model.joblib")
    encoder_output_path = os.path.join(MODEL_DIR, "name_encoders.joblib")
    # ============================================================
    # ⭐ 修改原因：資料庫底層切換
    # 
    # 本來為什麼：原本使用 sqlite3 庫與本地檔案 .db 連線。
    # 
    # 修改後為什麼：
    # 1. 改用 SQLAlchemy Engine，這是 PostgreSQL 的標準連線方式。
    # 2. 密碼設為 asdf7410，與你的 pgAdmin 和 database.py 保持同步。
    # 3. 移除檔案檢查 (os.path.exists)，因為資料庫現在是遠端/伺服器端的服務。
    # ============================================================

    from .database import DATABASE_URL  # ⭐新增：統一從 database.py 取得資料庫連線設定
    from sqlalchemy import create_engine

    # 建立 PostgreSQL 引擎
    engine = create_engine(DATABASE_URL)  # ⭐使用 database.py 中的 DATABASE_URL

    # ⭐ 修改：B 表只存活著的資料，直接讀全部，不需要過濾刪除標記
    df = pd.read_sql("SELECT * FROM cleaned_experiments", engine)

    print(f"📊 [2/4] 資料讀取成功 (已過濾刪除數據)，共 {len(df)} 筆有效數據。")

# ============================================================
# ⭐ 新增：0 筆資料保護機制（商業級邏輯）
# ============================================================
# 為什麼要改？
# 1. 如果資料庫沒有任何資料
# 2. 不應該進行 5-Fold
# 3. 不應該訓練模型
# 4. 不應該產生空模型檔案
# ============================================================

    if len(df) == 0:
        print("⚠️ 無訓練資料，停止模型訓練")
        print("📌 系統狀態應標記為 no_data")
        return "no_data"  
    # ============================================================
    # 💡 【核心修改】：處理 10 個類別型欄位的編碼 (Label Encoding)
    # ============================================================
    # 這些欄位ใน crud.py 已經被補成 "None" 了，這裡要把它們變數字
    categorical_cols = [
        "fragrance_name", "solvent_1_name", "solvent_2_name", 
        "solvent_3_name", "solvent_4_name", "solvent_5_name",
        "powder_name", "membrane", "capsule_gen", "wind_condition"
    ]
    
    encoders = {}
    for col in categorical_cols:

        le = LabelEncoder()
       

    # 1️⃣ 轉字串
        df[col] = df[col].astype(str)

    # 2️⃣ 統一空值
        df[col] = df[col].replace(["", "nan", "None"], "None")

    # 3️⃣ ⭐ 確保 encoder 一定學到 None
        temp_series = df[col]

        if "None" not in temp_series.values:
            temp_series = pd.concat([temp_series, pd.Series(["None"])])

    # 4️⃣ fit encoder
        le.fit(temp_series)

    # 5️⃣ transform 原資料
        df[f"{col}_encoded"] = le.transform(df[col])

        encoders[col] = le
    # 💾 儲存翻譯字典 (密碼本)
    joblib.dump(encoders, encoder_output_path)
    print("💾 已儲存類別編碼器 (name_encoders.joblib)")

    # ============================================================
    # ⭐ 修改原因：嚴格對齊 models.py 的 name 屬性標籤
    # 
    # 本來為什麼：原本使用英文 ID（如 temperature），會導致前端圖表也顯示英文。
    # 
    # 修改後為什麼：
    # 1. 建立 db_features 用於內部計算（對應資料庫英文欄位）。
    # 2. 建立 features 作為最終標籤（對應前端中文 UI）。
    # 3. 確保天數、溫度、重量、配方名稱與 models.py 的 name 屬性 100% 一致。
    # ============================================================

    # 給 AI 抓資料用的英文清單
    db_features = [
        'temperature', 'test_days', 'initial_weight',
        'fragrance_pct', 'solvent_1_pct', 'solvent_2_pct', 
        'solvent_3_pct', 'solvent_4_pct', 'solvent_5_pct', 'powder_pct',
        'solvent_1_exists', 'solvent_2_exists', 'solvent_3_exists', 
        'solvent_4_exists', 'solvent_5_exists', 'powder_exists',
        'fragrance_name_encoded', 'solvent_1_name_encoded', 'solvent_2_name_encoded',
        'solvent_3_name_encoded', 'solvent_4_name_encoded', 'solvent_5_name_encoded',
        'powder_name_encoded', 'membrane_encoded', 'capsule_gen_encoded', 'wind_condition_encoded'
    ]

    # 給前端圖表看的中文標籤（必須與 db_features 數量順序一致）
    features = [
        '測試溫度(℃)', '天數(Days)', '初始重量(g)', 
        '香精(%)', '溶劑1(%)', '溶劑2(%)', '溶劑3(%)', '溶劑4(%)', '溶劑5(%)', '稠粉(%)',
        '溶劑1(存在狀態)', '溶劑2(存在狀態)', '溶劑3(存在狀態)', '溶劑4(存在狀態)', '溶劑5(存在狀態)', '稠粉(存在狀態)',
        '香精名稱', '溶劑1名稱', '溶劑2名稱', '溶劑3名稱', '溶劑4名稱', '溶劑5名稱',
        '稠粉名稱', '膜料', '膠囊代數', '有/無吹風'
    ]

    # 🛡️ 【邏輯強化】：核心物理欄位若轉數字失敗，強制刪除該行以防誤導 AI
    core_phys_cols = ['temperature', 'initial_weight', 'test_days']
    df[core_phys_cols] = df[core_phys_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=core_phys_cols)

    # ⭐ 新增：過濾失重大於初始重量的物理不合理資料
    # 原因：失重不可能超過初始重量，這類異常資料會讓 AI 學到錯誤的物理規律
    df['weight_loss'] = pd.to_numeric(df['weight_loss'], errors='coerce')
    before_count = len(df)
    df = df[df['weight_loss'] <= df['initial_weight']]
    filtered_count = before_count - len(df)
    if filtered_count > 0:
        print(f"⚠️ 已過濾 {filtered_count} 筆失重大於初始重量的異常資料")

    # 處理空值並轉換格式
    # 修改點：使用 db_features (英文) 來進行 X 數據的抓取
    X = df[db_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['weight_loss'].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 🚀 【新增：單調性限制規則】
    # 任務：強制 AI 認定「天數」與「失重」正相關，防止出現天數增加失重反而減少的錯誤。
    # 修改點：因為 features 已改為中文，所以搜尋索引必須改為「天數(Days)」
    constraints = [0] * len(features)
    constraints[features.index('天數(Days)')] = 1  # 1 代表正相關限制

    print("🧠 [3/4] AI 正在學習配方揮發規律 (已加入物理單調性約束)...")
    
    # ============================================================
    # ⭐ 修改原因：實作 5-Fold 交叉驗證 (Cross-Validation)
    # 
    # 原本：使用 train_test_split，僅進行一次隨機抽樣考試，誤差值(MAE)容易產生偏差。
    # 
    # 修改後：
    # 1. 導入 KFold，將資料切成 5 份輪流考試。
    # 2. 確保每一筆實驗數據都曾作為「考卷」被驗證過，計算出更具公信力的平均誤差。
    # 3. 最終使用「全量數據」進行模型強化訓練，提升對未來配方的預測力。
    # ============================================================
    
    from sklearn.model_selection import KFold

    # 定義 5 折交叉驗證
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_maes = []
    
    print("🔄 啟動 5-Fold 交叉驗證模式 (評估模型穩定度)...")

    # 開始輪流考試
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        # 建立臨時模型用於該次考試
        cv_model = xgb.XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42,
            monotone_constraints=tuple(constraints)
        )
        
        cv_model.fit(X_train_cv, y_train_cv)
        
        # 預測並計算該次誤差
        fold_mae = mean_absolute_error(y_test_cv, cv_model.predict(X_test_cv))
        cv_maes.append(fold_mae)
        print(f"   第 {fold} 次驗證 MAE: {fold_mae:.4f} g")

    # 計算平均誤差
    mae = sum(cv_maes) / len(cv_maes)

    print("🧠 正在使用完整數據進行最終大腦訓練...")
    # 建立最終發布的模型 (學習所有已知的配方)
    model = xgb.XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42,
        monotone_constraints=tuple(constraints)
    )
    model.fit(X, y)

    # 儲存模型
    joblib.dump(model, model_output_path)
    print(f"✅ 模型訓練完成！已儲存至: {model_output_path}")

    print("-" * 40)
    print(f"✅ [4/4] 訓練成功！")
    print(f"📊 交叉驗證平均誤差 (MAE): {mae:.4f} g") # 此數值代表模型對未知配方的預期精準度
    print("💾 已生成大腦檔案: perfume_model.joblib")
    print("-" * 40)

    # 顯示排名
    print("\n🏆 配方影響力排名 (什麼最容易影響揮發)：")
    importance = model.feature_importances_
    feat_imp = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    for f, v in feat_imp[:12]: # 顯示前 12 名，看看名稱是否有影響力
        print(f" - {f}: {v:.4f}")

if __name__ == "__main__":
    train_perfume_model()
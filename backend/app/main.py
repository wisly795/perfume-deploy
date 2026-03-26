


from sqlalchemy import or_, cast, Numeric  # ⭐ 新增 cast 和 Numeric
import json
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import pandas as pd
import io
from . import crud, models, database
import numpy as np
import joblib
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
from .train_model import train_perfume_model
import threading

# 初始化 FastAPI 應用程式
app = FastAPI(title="香精揮發率預測系統 - 資料中心")

# 設定跨域資源共享 (CORS)，讓前端網頁可以順利存取後端 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源連線
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有傳輸方法 (GET, POST 等)
    allow_headers=["*"],  # 允許所有標頭
)

# 設定模型與編碼器的絕對路徑，確保搬移資料夾也不會出錯
# ⭐ 修改：改為使用 Docker Volume 路徑，與 train_model.py 完全一致
MODEL_DIR = "/app/models_storage"

MODEL_PATH = os.path.join(MODEL_DIR, "perfume_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "name_encoders.joblib")

# ============================================================
# ⭐ 新增：全域模型變數
# 原本程式碼：
#   每次 API 預測都會 joblib.load()
#
# 問題：
#   每次 request 都重新讀取模型檔案
#   如果未來有大量預測會造成 I/O 負擔
#
# 修改後：
#   Server 啟動時只讀取一次
#   所有 API request 共用同一個模型
# ============================================================

model = None
encoders = None
train_lock = threading.Lock()

# ⭐ 新增：模型狀態
model_status = "ready"

# ============================================================
# ⭐ 新增：FastAPI 啟動時載入模型
#
# 原本：
#   model = joblib.load() 在 API 內部
#
# 修改後：
#   Server 啟動時載入
#
# 優點：
#   1. 預測速度更快
#   2. 減少硬碟讀取
#   3. production 常見架構
# ============================================================


@app.on_event("startup")
def load_model():
    global model, encoders

    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        print("✅ 模型與編碼器載入成功")
    else:
        print("⚠️ 找不到模型檔案，預測功能暫時不可用")

# ============================================================
# ⭐ 新增：背景模型訓練
# ============================================================


def retrain_model():
    global model, encoders, train_lock, model_status

    # 如果目前已經有訓練在進行
    if train_lock.locked():
        print("⚠️ 已有模型訓練進行中，本次跳過")
        return

    # 上鎖（確保只有一個訓練）
    with train_lock:
        try:
            model_status = "training"
            print("🚀 背景開始訓練模型")

            result = train_perfume_model()

            if result == "no_data":
                model_status = "no_data"
                print("⚠️ 沒有資料，模型狀態設定為 no_data")
                return

            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODER_PATH)
            model_status = "ready"
            print("✅ 模型重新訓練完成")

        except Exception as e:
            model_status = "outdated"
            print("🔥 訓練失敗:", e)

        finally:
            # ⭐ 保證鎖一定釋放（關鍵）
            pass

        # ============================================================
        # ⭐ 新增：判斷是否為 no_data
        # ============================================================
        # 為什麼要改？
        # 如果 train_model 回傳 no_data
        # 不應該嘗試讀取模型檔
        # 不應該標記 ready
        # ============================================================

        if result == "no_data":
            model_status = "no_data"
            print("⚠️ 沒有資料，模型狀態設定為 no_data")
            return

        # 正常情況才載入模型
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        model_status = "ready"
        print("✅ 模型重新訓練完成")

# 定義前端傳進來的數據格式 (驗證資料類型)
class PredictRequest(BaseModel):
    temperature: float      # 測試溫度
    initial_weight: float   # 初始重量
    fragrance_pct: float    # 香精比例
    solvent_1_pct: float = 0.0
    solvent_2_pct: float = 0.0
    solvent_3_pct: float = 0.0
    solvent_4_pct: float = 0.0
    solvent_5_pct: float = 0.0
    powder_pct: float = 0.0
    fragrance_name: str = "None"
    solvent_1_name: str = "None"
    solvent_2_name: str = "None"
    solvent_3_name: str = "None"
    solvent_4_name: str = "None"
    solvent_5_name: str = "None"
    powder_name: str = "None"
    membrane: str = "None"
    capsule_gen: str = "None"
    wind_condition: str = "None"


# 初始化資料庫表格 (如果還沒建立的話)
models.Base.metadata.create_all(bind=database.engine)

# 取得資料庫連線的工具函數


# 這部分不需要修改，但前提是你的 database.py 必須像上面那樣定義了 SessionLocal。
def get_db():
    db = database.SessionLocal() # 這裡會去 database.py 找 SessionLocal
    try:
        yield db
    finally:
        db.close()

# 💡 核心工具：依照要求，將數值無條件捨去至小數點後 2 位 (不進位)


def floor_dec(x):
    if x is None: return 0.0
    try:
        # 放大100倍取整後再除回100，達成截斷效果
        return int(float(x) * 100) / 100.0
    except:
        return 0.0

# 判定資料被清洗掉的原因 (供前端除錯使用)


def get_deletion_reason(item):
    reasons = []
    if item.temperature is None: reasons.append("缺失溫度")
    if item.wind_condition is None: reasons.append("缺失吹風標籤")
    if item.initial_weight is None or item.initial_weight <= 0: reasons.append(
        "重量異常")
    if item.test_days is None: reasons.append("缺失天數")
    if item.weight_loss is None: reasons.append("缺失失重數據")
    return "、".join(reasons) if reasons else "系統去重或格式微調"

# ============================================================
# 🚀 核心接口：預測與查詢 0-45 天揮發曲線
# ============================================================


@app.post("/predict-timeline/")
async def predict_timeline(req: PredictRequest, db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="模型尚未準備完成"
        )

    try:

        # ============================================================
        # ⭐ 商業級輸入驗證邏輯（物理合理性 + 結構完整性）
        # ============================================================

        # 溶劑整理
        s_list = [
            {"n": req.solvent_1_name or "None", "p": floor_dec(req.solvent_1_pct)},
            {"n": req.solvent_2_name or "None", "p": floor_dec(req.solvent_2_pct)},
            {"n": req.solvent_3_name or "None", "p": floor_dec(req.solvent_3_pct)},
            {"n": req.solvent_4_name or "None", "p": floor_dec(req.solvent_4_pct)},
            {"n": req.solvent_5_name or "None", "p": floor_dec(req.solvent_5_pct)},
        ]

        # ------------------------------------------------------------
        # 1️⃣ 名稱未選但比例 > 0 → 不合理
        # ------------------------------------------------------------

        if req.fragrance_name in ["", None, "None"] and floor_dec(req.fragrance_pct) > 0:
            raise HTTPException(
                status_code=400,
                detail="香精名稱未選擇，不能填寫香精(%)"
            )

        if req.powder_name in ["", None, "None"] and floor_dec(req.powder_pct) > 0:
            raise HTTPException(
                status_code=400,
                detail="稠粉名稱未選擇，不能填寫稠粉(%)"
            )

        for i, s in enumerate(s_list, start=1):
            if s["n"] in ["", None, "None"] and s["p"] > 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"溶劑{i}名稱未選擇，不能填寫溶劑{i}(%)"
                )

        # ------------------------------------------------------------
        # 2️⃣ 比例為 0 但名稱存在 → 自動歸 None（修正輸入）
        # ------------------------------------------------------------

        if floor_dec(req.fragrance_pct) == 0:
            req.fragrance_name = "None"

        if floor_dec(req.powder_pct) == 0:
            req.powder_name = "None"

        for s in s_list:
            if s["p"] == 0:
                s["n"] = "None"

        # ------------------------------------------------------------
        # 3️⃣ 至少必須存在：香精 或 任一溶劑
        # ------------------------------------------------------------

        has_fragrance = (
            req.fragrance_name not in ["", None, "None"]
            and floor_dec(req.fragrance_pct) > 0
        )

        has_solvent = any(
            s["n"] not in ["", None, "None"] and s["p"] > 0
            for s in s_list
        )

        if not (has_fragrance or has_solvent):
            raise HTTPException(
                status_code=400,
                detail="配方至少需要 香精(%) 或 任一溶劑(%)"
            )

        # ============================================================
        # ⭐ 驗證結束
        # ============================================================

        # 【第一步】：標準化輸入數值。前端傳什麼進來，我們先統一「截斷到兩位」再比對
        req.temperature = floor_dec(req.temperature)
        req.initial_weight = floor_dec(req.initial_weight)
        req.fragrance_pct = floor_dec(req.fragrance_pct)

        # 🛡️ 【第二步：完全命中判定 - 高精確搜尋】
        # ⭐ 修改：方法三，改用 cast(Numeric(10,2)) 讓 PostgreSQL 在 SQL 層截斷後比對
        # 原因：Python floor_dec 和 PostgreSQL double precision 之間有精度差異，
        # 改為在 PostgreSQL 內部用 NUMERIC(10,2) 做截斷，比對永遠正確
        from sqlalchemy import func, Numeric
        def nc(col):
            return func.cast(col, Numeric(10, 2))

        potential_hits = db.query(models.CleanedExperiment).filter(
            nc(models.CleanedExperiment.temperature) == nc(floor_dec(req.temperature)),
            models.CleanedExperiment.fragrance_name == (req.fragrance_name if req.fragrance_name else "None"),
            nc(models.CleanedExperiment.fragrance_pct) == nc(floor_dec(req.fragrance_pct)),

            # 溶劑 1
            models.CleanedExperiment.solvent_1_name == (s_list[0]["n"] if s_list[0]["n"] else "None"),
            nc(models.CleanedExperiment.solvent_1_pct) == nc(floor_dec(s_list[0]["p"])),
            # 溶劑 2
            models.CleanedExperiment.solvent_2_name == (s_list[1]["n"] if s_list[1]["n"] else "None"),
            nc(models.CleanedExperiment.solvent_2_pct) == nc(floor_dec(s_list[1]["p"])),
            # 溶劑 3
            models.CleanedExperiment.solvent_3_name == (s_list[2]["n"] if s_list[2]["n"] else "None"),
            nc(models.CleanedExperiment.solvent_3_pct) == nc(floor_dec(s_list[2]["p"])),
            # 溶劑 4
            models.CleanedExperiment.solvent_4_name == (s_list[3]["n"] if s_list[3]["n"] else "None"),
            nc(models.CleanedExperiment.solvent_4_pct) == nc(floor_dec(s_list[3]["p"])),
            # 溶劑 5
            models.CleanedExperiment.solvent_5_name == (s_list[4]["n"] if s_list[4]["n"] else "None"),
            nc(models.CleanedExperiment.solvent_5_pct) == nc(floor_dec(s_list[4]["p"])),

            # 其他條件
            models.CleanedExperiment.membrane == (req.membrane if req.membrane else "None"),
            models.CleanedExperiment.powder_name == (req.powder_name if req.powder_name else "None"),
            nc(models.CleanedExperiment.powder_pct) == nc(floor_dec(req.powder_pct)),
            models.CleanedExperiment.wind_condition == (req.wind_condition if req.wind_condition else "None"),
            models.CleanedExperiment.capsule_gen == (req.capsule_gen if req.capsule_gen else "None"),
            nc(models.CleanedExperiment.initial_weight) == nc(floor_dec(req.initial_weight))
        ).order_by(models.CleanedExperiment.test_days).all()
        
        # ============================================================
# ⭐ 相同配方查詢
# ============================================================
    # ============================================================
        # ⭐ 修改：移除 if 判定，讓「完全命中」的資料也能出現在「相同配方」
        # ============================================================
        similar_formula_hits = []

        input_solvents = {
            s_list[0]["n"],
            s_list[1]["n"],
            s_list[2]["n"],
            s_list[3]["n"],
            s_list[4]["n"]
        }

        input_solvents.discard("None")

        candidates = db.query(models.CleanedExperiment).filter(
            models.CleanedExperiment.fragrance_name == req.fragrance_name
        ).all()

        for c in sorted(candidates, key=lambda x: x.test_days):
            db_solvents = {
                c.solvent_1_name,
                c.solvent_2_name,
                c.solvent_3_name,
                c.solvent_4_name,
                c.solvent_5_name
            }
            db_solvents.discard("None")

            # 溶劑集合完全相同才算相同配方
            if input_solvents == db_solvents:
                # ⭐ 修改：改用 round(x, 2) 做 Python 層比對，避免浮點數精度問題
                # 原因：c.temperature 等欄位從 PostgreSQL 讀出來是 double precision，
                # 直接用 == floor_dec() 比對會因精度差異失敗
                is_exact_match = (
                    round(float(c.temperature), 2) == round(float(req.temperature), 2) and
                    round(float(c.initial_weight), 2) == round(float(req.initial_weight), 2) and
                    round(float(c.fragrance_pct), 2) == round(float(req.fragrance_pct), 2) and
                    c.powder_name == (req.powder_name if req.powder_name else "None") and
                    round(float(c.powder_pct), 2) == round(float(req.powder_pct), 2) and
                    c.membrane == (req.membrane if req.membrane else "None") and
                    c.capsule_gen == (req.capsule_gen if req.capsule_gen else "None") and
                    c.wind_condition == (req.wind_condition if req.wind_condition else "None")
                )

                similar_formula_hits.append({
                    "fragrance_name": c.fragrance_name,
                    "fragrance_pct": c.fragrance_pct,
                    "solvent_1_name": c.solvent_1_name,
                    "solvent_1_pct": c.solvent_1_pct,
                    "solvent_2_name": c.solvent_2_name,
                    "solvent_2_pct": c.solvent_2_pct,
                    "solvent_3_name": c.solvent_3_name,
                    "solvent_3_pct": c.solvent_3_pct,
                    "solvent_4_name": c.solvent_4_name,
                    "solvent_4_pct": c.solvent_4_pct,
                    "solvent_5_name": c.solvent_5_name,
                    "solvent_5_pct": c.solvent_5_pct,
                    "powder_name": c.powder_name,
                    "powder_pct": c.powder_pct,
                    "membrane": c.membrane,
                    "capsule_gen": c.capsule_gen,
                    "wind_condition": c.wind_condition,
                    "initial_weight": c.initial_weight,
                    "temperature": c.temperature,
                    "day": c.test_days,
                    "weight_loss": floor_dec(c.weight_loss),
                    # 將嚴格比對的結果傳給前端
                    "is_exact": is_exact_match,
                    # ⭐ 新增：來源 Excel 檔名與行號，供前端顯示
                    "upload_filename": c.upload_filename,
                    "excel_row_no": c.excel_row_no
                })

        measured_points = [
            {
                "day": h.test_days,
                "weight_loss": floor_dec(h.weight_loss),

                "fragrance_name": h.fragrance_name,
                "fragrance_pct": h.fragrance_pct,

                "solvent_1_name": h.solvent_1_name,
                "solvent_1_pct": h.solvent_1_pct,

                "solvent_2_name": h.solvent_2_name,
                "solvent_2_pct": h.solvent_2_pct,

                "solvent_3_name": h.solvent_3_name,
                "solvent_3_pct": h.solvent_3_pct,

                "solvent_4_name": h.solvent_4_name,
                "solvent_4_pct": h.solvent_4_pct,

                "solvent_5_name": h.solvent_5_name,
                "solvent_5_pct": h.solvent_5_pct,

                "powder_name": h.powder_name,
                "powder_pct": h.powder_pct,

                "membrane": h.membrane,
                "capsule_gen": h.capsule_gen,
                "wind_condition": h.wind_condition,

                "temperature": h.temperature,
                "initial_weight": h.initial_weight,
                # ⭐ 新增：來源 Excel 檔名與行號，供前端顯示
                "upload_filename": h.upload_filename,
                "excel_row_no": h.excel_row_no
            }
            for h in potential_hits
        ]
       

        

        # ============================================================
        # ⭐ 修改：不再每次 joblib.load()
        #
        # 原本：
        #   model = joblib.load(MODEL_PATH)
        #   encoders = joblib.load(ENCODER_PATH)
        #
        # 修改後：
        #   使用 server 啟動時載入的全域模型
        # ============================================================

        if model is None or encoders is None:
            raise FileNotFoundError("模型尚未載入")

        def safe_encode(col_name, value):

            le = encoders[col_name]

            # 先轉字串
            if value is None:
                val_str = "None"
            else:
                val_str = str(value).strip()

            # 如果是空字串或不在 encoder 裡
            if val_str == "" or val_str not in le.classes_:
                val_str = "None"

            return int(le.transform([val_str])[0])
        feature_names = [
            'temperature', 'test_days', 'initial_weight',
            'fragrance_pct', 'solvent_1_pct', 'solvent_2_pct', 
            'solvent_3_pct', 'solvent_4_pct', 'solvent_5_pct', 'powder_pct',
            'solvent_1_exists', 'solvent_2_exists', 'solvent_3_exists', 
            'solvent_4_exists', 'solvent_5_exists', 'powder_exists',
            'fragrance_name_encoded', 'solvent_1_name_encoded', 'solvent_2_name_encoded',
            'solvent_3_name_encoded', 'solvent_4_name_encoded', 'solvent_5_name_encoded',
            'powder_name_encoded', 'membrane_encoded', 'capsule_gen_encoded', 'wind_condition_encoded'
        ]
        encoded_values = {
            "fragrance_name_encoded": safe_encode("fragrance_name", req.fragrance_name),
            "solvent_1_name_encoded": safe_encode("solvent_1_name", s_list[0]["n"]),
            "solvent_2_name_encoded": safe_encode("solvent_2_name", s_list[1]["n"]),
            "solvent_3_name_encoded": safe_encode("solvent_3_name", s_list[2]["n"]),
            "solvent_4_name_encoded": safe_encode("solvent_4_name", s_list[3]["n"]),
            "solvent_5_name_encoded": safe_encode("solvent_5_name", s_list[4]["n"]),
            "powder_name_encoded": safe_encode("powder_name", req.powder_name),
            "membrane_encoded": safe_encode("membrane", req.membrane),
            "capsule_gen_encoded": safe_encode("capsule_gen", req.capsule_gen),
            "wind_condition_encoded": safe_encode("wind_condition", req.wind_condition),
        }

        ai_timeline = []
        for d in range(46):

            input_row = {
                'temperature': req.temperature,
                'test_days': d,
                'initial_weight': req.initial_weight,
                'fragrance_pct': req.fragrance_pct,

                'solvent_1_pct': s_list[0]["p"],
                'solvent_2_pct': s_list[1]["p"],
                'solvent_3_pct': s_list[2]["p"],
                'solvent_4_pct': s_list[3]["p"],
                'solvent_5_pct': s_list[4]["p"],

                'powder_pct': floor_dec(req.powder_pct),

                'solvent_1_exists': 1 if s_list[0]["n"] != "None" else 0,
                'solvent_2_exists': 1 if s_list[1]["n"] != "None" else 0,
                'solvent_3_exists': 1 if s_list[2]["n"] != "None" else 0,
                'solvent_4_exists': 1 if s_list[3]["n"] != "None" else 0,
                'solvent_5_exists': 1 if s_list[4]["n"] != "None" else 0,

                'powder_exists': 1 if req.powder_name != "None" else 0,

                **encoded_values
            }

            X_input = pd.DataFrame([input_row])[feature_names]

            pred_loss = model.predict(X_input)[0]

            # ⭐ 修改：同時限制下限(≥0) 和上限(≤初始重量)，符合物理定律
            # 原因：失重不可能為負數，也不可能超過初始重量
            pred_loss_clamped = max(0, min(pred_loss, req.initial_weight))

            ai_timeline.append({
                "day": d,
                "weight_loss": floor_dec(pred_loss_clamped)
            })
        source = "AI預測"

        if potential_hits:
            source = "完全命中"
        
        # ⭐ 修改後：明確清理 input_formula 並移除 RD 編號
        # 我們直接從 req (PredictRequest) 抓取資料，確保 1-5 溶劑都存在
        clean_input = {
            "fragrance_name": req.fragrance_name,
            "fragrance_pct": floor_dec(req.fragrance_pct),
            "solvent_1_name": req.solvent_1_name,
            "solvent_1_pct": floor_dec(req.solvent_1_pct),
            "solvent_2_name": req.solvent_2_name,
            "solvent_2_pct": floor_dec(req.solvent_2_pct),
            "solvent_3_name": req.solvent_3_name,
            "solvent_3_pct": floor_dec(req.solvent_3_pct),
            "solvent_4_name": req.solvent_4_name,
            "solvent_4_pct": floor_dec(req.solvent_4_pct),
            "solvent_5_name": req.solvent_5_name,
            "solvent_5_pct": floor_dec(req.solvent_5_pct),
            "powder_name": req.powder_name,
            "powder_pct": floor_dec(req.powder_pct),
            "membrane": req.membrane,
            "capsule_gen": req.capsule_gen,
            "temperature": floor_dec(req.temperature),
            "wind_condition": req.wind_condition,
            "initial_weight": floor_dec(req.initial_weight)
        }

        return {
            "status": "success",
            "source": source,
            "exact_match": measured_points,
            "input_formula": clean_input, # 使用清理後的配方
            "similar_formula": similar_formula_hits, # 這裡原本就已經移除 RD
            "ai_results": ai_timeline
        }
        # ============================================================
    # ⭐ 修改：保留原本的 HTTPException 狀態碼
    # ============================================================

    except HTTPException as e:
        # ⭐ 如果本來就是 400 / 503
        # 不要重新包裝成 500
        print("🔥 Predict API HTTPException:", e.detail)
        raise e

    except Exception as e:
        # ⭐ 這裡才是真正未知錯誤
        print("🔥 Predict API Unexpected Error:", e)

        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="系統暫時無法完成預測，請稍後再試"
        )
    
@app.get("/get-options/")
def get_options(db: Session = Depends(get_db)):
    def get_dist(col):
        # ⭐ 修改：B 表只存活著的資料，直接撈不需要過濾刪除標記
        res = db.query(col).distinct().all()
        return sorted([r[0] for r in res if r[0] not in ["None", None, "nan", ""]])

    return {
        "香精名稱": get_dist(models.CleanedExperiment.fragrance_name),
        "溶劑1名稱": get_dist(models.CleanedExperiment.solvent_1_name),
        "溶劑2名稱": get_dist(models.CleanedExperiment.solvent_2_name),
        "溶劑3名稱": get_dist(models.CleanedExperiment.solvent_3_name),
        "溶劑4名稱": get_dist(models.CleanedExperiment.solvent_4_name),
        "溶劑5名稱": get_dist(models.CleanedExperiment.solvent_5_name),
        "稠粉名稱": get_dist(models.CleanedExperiment.powder_name),
        "膜料": get_dist(models.CleanedExperiment.membrane),
        "膠囊代數": get_dist(models.CleanedExperiment.capsule_gen)
    }

@app.post("/upload-excel/")
async def upload_excel(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    global model_status

    # ⭐ 修改：統一在最開頭處理 filename，小寫 + 特殊字元清洗一次到位
    # 原因：原本只做 lower()，但 crud.py 內部還會做特殊字元清洗
    # 導致 A 表存的 filename 和後續傳入的 filename 不一致，清洗靜默失敗
    import re as _re
    filename = _re.sub(r'[^\w\.\-]', '_', file.filename.lower())

    if not filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(status_code=400, detail="請上傳 Excel 或 CSV 檔案")

    try:
        contents = await file.read()

        # ⭐ 修改：強制以字串模式讀取，避免浮點數誤差
        if filename.endswith('.csv'):
            df = pd.read_csv(
                io.BytesIO(contents),
                encoding='utf-8-sig',
                dtype=str,               # 所有欄位讀為字串
                keep_default_na=False     # 不自動轉 NaN
            )
        else:
            df = pd.read_excel(
                io.BytesIO(contents),
                sheet_name=0,
                dtype=str,               # 所有欄位讀為字串
                keep_default_na=False
            )

        # 去除欄位名稱前後空白
        df.columns = [str(c).strip() for c in df.columns]

        # 將空字串轉為 None (Pandas 讀取空cell會變成空字串)
        df = df.replace(r'^\s*$', None, regex=True)

        # ⭐ 修改：全程使用統一處理過的 filename，不再用 file.filename
        inserted_new, duplicate_skipped = crud.batch_insert_raw(db, df, filename)

        # ⭐ 修改：接收 sync_raw_to_cleaned 回傳的清洗報告
        clean_report = crud.sync_raw_to_cleaned(db, filename=filename)

        # 標記模型為 outdated
        model_status = "outdated"
        cleaned_count = db.query(models.CleanedExperiment).count()

        # ============================================================
        # ⭐ 新增：寫入操作紀錄
        # ============================================================
        log = models.OperationLog(
            action_type="upload_excel",
            target=filename,
            affected_rows=inserted_new,
            group_snapshot=json.dumps({
                "received_rows": len(df),
                "inserted_new": inserted_new,
                "duplicate_skipped": clean_report["duplicate_skipped"] if clean_report else 0,
                "cleaned_inserted": clean_report["cleaned_inserted"] if clean_report else 0,
                "abnormal_corrected": clean_report["abnormal_corrected"] if clean_report else 0,
                "rejected_groups": clean_report["rejected_groups"] if clean_report else 0,
                "total_cleaned_rows": cleaned_count
            }, ensure_ascii=False)
        )
        db.add(log)
        db.commit()

        return {
            "status": "success",
            "summary": {
                "received_rows": len(df),                          # Excel 收到的總列數
                "inserted_new": inserted_new,                      # 成功寫入原始資料庫的筆數
                # ⭐ 修改：改為顯示 B 表的去重跳過數，A 表已改為無腦全部寫入
                "duplicate_skipped": clean_report["duplicate_skipped"] if clean_report else 0,
                "cleaned_inserted": clean_report["cleaned_inserted"] if clean_report else 0,      # 成功進入訓練資料庫的筆數
                "abnormal_corrected": clean_report["abnormal_corrected"] if clean_report else 0,  # 異常天數被自動修正的筆數
                "rejected_groups": clean_report["rejected_groups"] if clean_report else 0,        # 整組被排除的組數
                "total_cleaned_rows": cleaned_count                # 訓練資料庫目前累計總筆數
            }
        }

    except Exception as e:
        print("🔥 API ERROR:", e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

# ============================================================
# ⭐ Health Check API
# 任務：讓 Docker / Fly.io 檢查服務是否正常
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "perfume_volatility_system"
    }


# ============================================================
# ⭐ Model Status API
# ============================================================

# ============================================================
# ✅ 模型狀態查詢接口
# ============================================================
@app.get("/model-status")
def get_model_status():
    return {
        "status": model_status
    }


# ============================================================
# ⭐ 手動訓練模型 API
# ============================================================

@app.post("/manual-retrain")
def manual_retrain():

    global model_status

    if train_lock.locked():
        raise HTTPException(status_code=400, detail="模型正在訓練中")

    try:
        

        retrain_model()

        

        return {
            "status": "success",
            "message": "模型重新訓練完成"
        }

    except Exception as e:
        model_status = "outdated"
        print("🔥 手動訓練失敗:", e)
        raise HTTPException(status_code=500, detail="模型訓練失敗")
# ============================================================
# ✅ 修改後：特徵重要性分析接口 (標籤與 models.py 之 name 屬性完全一致)
# ============================================================
@app.get("/feature-importance")
def get_feature_importance():
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未準備完成")
    
    try:
        # 標籤順序嚴格對應模型特徵列，字樣對齊 models.py
        feature_labels = [
            '測試溫度(℃)', '天數(Days)', '初始重量(g)', 
            '香精(%)', '溶劑1(%)', '溶劑2(%)', '溶劑3(%)', '溶劑4(%)', '溶劑5(%)', '稠粉(%)',
            '溶劑1(存在狀態)', '溶劑2(存在狀態)', '溶劑3(存在狀態)', '溶劑4(存在狀態)', '溶劑5(存在狀態)', '稠粉(存在狀態)',
            '香精名稱', '溶劑1名稱', '溶劑2名稱', '溶劑3名稱', '溶劑4名稱', '溶劑5名稱',
            '稠粉名稱', '膜料', '膠囊代數', '有/無吹風'
        ]
        
        # 提取模型權重並格式化
        importances = model.feature_importances_.tolist()
        
        data = [
            {"name": label, "value": round(float(val), 4)} 
            for label, val in zip(feature_labels, importances)
        ]
        # 依影響力由高至低排序
        data.sort(key=lambda x: x["value"], reverse=True)
        
        return {"status": "success", "data": data}
        
    except Exception as e:
        print(f"🔥 Feature Importance API Error: {e}")
        raise HTTPException(status_code=500, detail="無法提取模型特徵權重數據")
    
    

# ============================================================
# 📁 資料庫管理系統：第一層 - 獲取檔案清單
# ============================================================
@app.get("/manage/files")
def list_uploaded_files(db: Session = Depends(get_db)):
    # 從原始表抓取不重複的檔名與最後上傳時間
    from sqlalchemy import func
    files = db.query(
        models.RawExperiment.upload_filename,
        func.max(models.RawExperiment.created_at).label("upload_time")
    ).group_by(models.RawExperiment.upload_filename).all()
    
    result = []
    for f in files:
        # ⭐ 修改：B 表只存活著的資料，直接 count 不需要過濾刪除標記
        active_count = db.query(models.CleanedExperiment).filter(
            models.CleanedExperiment.upload_filename == f.upload_filename
        ).count()
        result.append({
            "filename": f.upload_filename,
            "upload_time": f.upload_time,
            "row_count": active_count,
            "has_active": active_count > 0
        })
    
    return result

# ============================================================
# 🧪 資料庫管理系統：第二層 - 獲取特定檔案內的實驗組（含異常判定）
# ============================================================
@app.get("/manage/experiments/{filename}")
def list_experiments_in_file(filename: str, db: Session = Depends(get_db)):
    # ⭐ 修改：B 表只存活著的資料，直接撈全部
    items = db.query(models.CleanedExperiment).filter(
        models.CleanedExperiment.upload_filename == filename
    ).all()
    
    groups = {}
    for item in items:
        key = (
            item.fragrance_name, item.fragrance_pct,
            item.solvent_1_name, item.solvent_1_pct,
            item.solvent_2_name, item.solvent_2_pct,
            item.solvent_3_name, item.solvent_3_pct,
            item.solvent_4_name, item.solvent_4_pct,
            item.solvent_5_name, item.solvent_5_pct,
            item.powder_name, item.powder_pct,
            item.temperature, item.initial_weight,
            item.membrane, item.capsule_gen, item.wind_condition
        )
        
        if key not in groups:
            groups[key] = {
                "details": [],
                "is_invalid": False,
                "reasons": [],
            }
        
        groups[key]["details"].append({
            "day": item.test_days,
            "loss": item.weight_loss
        })
        # ⭐ 新增：收集這組的所有 B 表 id，供前端刪除時使用（方法一：ID 比對）
        if "group_ids" not in groups[key]:
            groups[key]["group_ids"] = []
        groups[key]["group_ids"].append(item.id)
        # ⭐ 新增：收集這組所有天的 excel_row_no，供前端顯示行號範圍
        if "excel_rows" not in groups[key]:
            groups[key]["excel_rows"] = []
        groups[key]["excel_rows"].append(item.excel_row_no)

    result = []
    for key, data in groups.items():
        data["details"].sort(key=lambda x: x["day"])

        for i in range(len(data["details"])):
            curr = data["details"][i]

            if curr["loss"] is None:
                data["is_invalid"] = True
                data["reasons"].append(f"第 {curr['day']} 天數據缺失 (空白)")
                continue

            if curr["loss"] < 0:
                data["is_invalid"] = True
                data["reasons"].append(f"第 {curr['day']} 天失重為負數")
            
            if i > 0:
                prev = data["details"][i-1]
                if prev["loss"] is not None and curr["loss"] < prev["loss"]:
                    data["is_invalid"] = True
                    data["reasons"].append(f"第 {curr['day']} 天失重 ({curr['loss']}g) 小於第 {prev['day']} 天 ({prev['loss']}g)")

        result.append({
            "formula": {
                "fragrance_name": key[0],
                "fragrance_pct": floor_dec(key[1]),
                "solvent_1_name": key[2],
                "solvent_1_pct": floor_dec(key[3]),
                "solvent_2_name": key[4],
                "solvent_2_pct": floor_dec(key[5]),
                "solvent_3_name": key[6],
                "solvent_3_pct": floor_dec(key[7]),
                "solvent_4_name": key[8],
                "solvent_4_pct": floor_dec(key[9]),
                "solvent_5_name": key[10],
                "solvent_5_pct": floor_dec(key[11]),
                "powder_name": key[12],
                "powder_pct": floor_dec(key[13]),
                "temperature": floor_dec(key[14]),
                "initial_weight": floor_dec(key[15]),
                "membrane": key[16],
                "capsule_gen": key[17],
                "wind_condition": key[18]
            },
            "group_ids": data["group_ids"],  # ⭐ 新增：這組所有 B 表 id，供刪除用
            "is_active": True,
            "is_invalid": data["is_invalid"],
            "reasons": list(set(data["reasons"])),
            "timeline": data["details"],
            # ⭐ 新增：這組在 Excel 的行號範圍
            "excel_row_start": min((r for r in data["excel_rows"] if r is not None), default=None),
            "excel_row_end": max((r for r in data["excel_rows"] if r is not None), default=None)
        })

    return sorted(result, key=lambda x: x["is_invalid"], reverse=True)



# ============================================================
# 🗑️ 資料庫管理系統：執行整組刪除
# ============================================================
class DeleteExperimentRequest(BaseModel):

    upload_filename: str
    # ⭐ 新增：方法一，前端傳這組所有 B 表 id，後端直接用 id 刪除，不比對浮點數
    group_ids: list

    # 以下配方欄位保留，用於寫入 OperationLog 的 group_snapshot
    fragrance_name: str
    fragrance_pct: float

    solvent_1_name: str
    solvent_1_pct: float

    solvent_2_name: str
    solvent_2_pct: float

    solvent_3_name: str
    solvent_3_pct: float

    solvent_4_name: str
    solvent_4_pct: float

    solvent_5_name: str
    solvent_5_pct: float

    powder_name: str
    powder_pct: float

    temperature: float
    initial_weight: float

    membrane: str
    capsule_gen: str
    wind_condition: str

class DeleteDayRequest(DeleteExperimentRequest):
    test_days: int

from sqlalchemy import or_

# ============================================================
# 🗑️ 三、資料庫管理系統：執行整組刪除 (全系統通用寬容版)
# ============================================================
@app.post("/manage/delete-group")
async def delete_experiment_group(req: DeleteExperimentRequest, db: Session = Depends(get_db)):
    try:
        import json
        global model_status
        w_B = floor_dec(req.initial_weight)
        
        # ⭐ 修改：方法一，直接用 group_ids 查詢，完全不需要浮點數比對
        # 原因：浮點數 == 比對在 PostgreSQL 中不可靠，改用整數 id 永遠正確
        cleaned_records = db.query(models.CleanedExperiment).filter(
            models.CleanedExperiment.id.in_(req.group_ids)
        ).all()

        if not cleaned_records:
            return {"status": "success", "deleted_rows": 0}

        # ⭐ 新增：寫入 OperationLog，儲存配方快照供歷史紀錄區顯示
        # ⭐ 修改：從 cleaned_records 中取得 excel_row_no 的最小值與最大值，存入 group_snapshot
        # 原因：歷史紀錄頁面需要顯示這組資料在 Excel 的行號範圍
        excel_rows = [rec.excel_row_no for rec in cleaned_records if rec.excel_row_no is not None]
        snapshot_excel_row_start = min(excel_rows) if excel_rows else None
        snapshot_excel_row_end = max(excel_rows) if excel_rows else None

        group_snapshot = json.dumps({
            "fragrance_name": req.fragrance_name,
            "fragrance_pct": floor_dec(req.fragrance_pct),
            "solvent_1_name": req.solvent_1_name,
            "solvent_1_pct": floor_dec(req.solvent_1_pct),
            "solvent_2_name": req.solvent_2_name,
            "solvent_2_pct": floor_dec(req.solvent_2_pct),
            "solvent_3_name": req.solvent_3_name,
            "solvent_3_pct": floor_dec(req.solvent_3_pct),
            "solvent_4_name": req.solvent_4_name,
            "solvent_4_pct": floor_dec(req.solvent_4_pct),
            "solvent_5_name": req.solvent_5_name,
            "solvent_5_pct": floor_dec(req.solvent_5_pct),
            "powder_name": req.powder_name,
            "powder_pct": floor_dec(req.powder_pct),
            "temperature": floor_dec(req.temperature),
            "initial_weight": w_B,
            "membrane": req.membrane,
            "capsule_gen": req.capsule_gen,
            "wind_condition": req.wind_condition,
            "upload_filename": req.upload_filename,
            "excel_row_start": snapshot_excel_row_start,  # ⭐ 新增：Excel 起始行號
            "excel_row_end": snapshot_excel_row_end        # ⭐ 新增：Excel 結束行號
        }, ensure_ascii=False)

        log = models.OperationLog(
            action_type="delete_group",
            target=req.upload_filename,
            affected_rows=len(cleaned_records),
            group_snapshot=group_snapshot
        )
        db.add(log)
        db.flush()  # 取得 log.id

        # ⭐ 新增：備份到 DeletedExperiment 表
        for rec in cleaned_records:
            backup = models.DeletedExperiment(
                upload_filename=rec.upload_filename,
                product_rd_no=rec.product_rd_no,
                fragrance_name=rec.fragrance_name,
                solvent_1_name=rec.solvent_1_name,
                solvent_2_name=rec.solvent_2_name,
                solvent_3_name=rec.solvent_3_name,
                solvent_4_name=rec.solvent_4_name,
                solvent_5_name=rec.solvent_5_name,
                powder_name=rec.powder_name,
                membrane=rec.membrane,
                capsule_gen=rec.capsule_gen,
                wind_condition=rec.wind_condition,
                fragrance_pct=rec.fragrance_pct,
                solvent_1_pct=rec.solvent_1_pct,
                solvent_2_pct=rec.solvent_2_pct,
                solvent_3_pct=rec.solvent_3_pct,
                solvent_4_pct=rec.solvent_4_pct,
                solvent_5_pct=rec.solvent_5_pct,
                powder_pct=rec.powder_pct,
                temperature=rec.temperature,
                initial_weight=rec.initial_weight,
                test_days=rec.test_days,
                solvent_1_exists=rec.solvent_1_exists,
                solvent_2_exists=rec.solvent_2_exists,
                solvent_3_exists=rec.solvent_3_exists,
                solvent_4_exists=rec.solvent_4_exists,
                solvent_5_exists=rec.solvent_5_exists,
                powder_exists=rec.powder_exists,
                weight_loss=rec.weight_loss,
                raw_id=rec.raw_id,
                operation_log_id=log.id,
                excel_row_no=rec.excel_row_no  # ⭐ 新增：備份時保留 Excel 行號
            )
            db.add(backup)

        # ⭐ 修改：物理刪除 B 表資料
        for rec in cleaned_records:
            db.delete(rec)

        db.commit()
        model_status = "outdated"
        print(f"✅ [結果] 物理刪除 {len(cleaned_records)} 筆，已備份至 DeletedExperiment")
        return {"status": "success", "deleted_rows": len(cleaned_records)}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================
# 🗑️ 刪除整個 Excel 批次 (修正 404 問題)
# ============================================================
@app.delete("/manage/delete-file")
def delete_entire_file(filename: str, db: Session = Depends(get_db)):
    global model_status
    try:
        # ⭐ 修改：查出所有要刪的 B 表資料
        cleaned_records = db.query(models.CleanedExperiment).filter(
            models.CleanedExperiment.upload_filename == filename
        ).all()

        # ⭐ 新增：寫入 OperationLog
        log = models.OperationLog(
            action_type="delete_file",
            target=filename,
            affected_rows=len(cleaned_records)
        )
        db.add(log)
        db.flush()

        # ⭐ 新增：備份到 DeletedExperiment 表
        for rec in cleaned_records:
            backup = models.DeletedExperiment(
                upload_filename=rec.upload_filename,
                product_rd_no=rec.product_rd_no,
                fragrance_name=rec.fragrance_name,
                solvent_1_name=rec.solvent_1_name,
                solvent_2_name=rec.solvent_2_name,
                solvent_3_name=rec.solvent_3_name,
                solvent_4_name=rec.solvent_4_name,
                solvent_5_name=rec.solvent_5_name,
                powder_name=rec.powder_name,
                membrane=rec.membrane,
                capsule_gen=rec.capsule_gen,
                wind_condition=rec.wind_condition,
                fragrance_pct=rec.fragrance_pct,
                solvent_1_pct=rec.solvent_1_pct,
                solvent_2_pct=rec.solvent_2_pct,
                solvent_3_pct=rec.solvent_3_pct,
                solvent_4_pct=rec.solvent_4_pct,
                solvent_5_pct=rec.solvent_5_pct,
                powder_pct=rec.powder_pct,
                temperature=rec.temperature,
                initial_weight=rec.initial_weight,
                test_days=rec.test_days,
                solvent_1_exists=rec.solvent_1_exists,
                solvent_2_exists=rec.solvent_2_exists,
                solvent_3_exists=rec.solvent_3_exists,
                solvent_4_exists=rec.solvent_4_exists,
                solvent_5_exists=rec.solvent_5_exists,
                powder_exists=rec.powder_exists,
                weight_loss=rec.weight_loss,
                raw_id=rec.raw_id,
                operation_log_id=log.id,
                excel_row_no=rec.excel_row_no  # ⭐ 新增：備份時保留 Excel 行號，讓復原後行號不消失
            )
            db.add(backup)

        # ⭐ 修改：物理刪除 B 表資料
        for rec in cleaned_records:
            db.delete(rec)

        db.commit()
        model_status = "outdated"
        return {"status": "success", "deleted_rows": len(cleaned_records)}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ⭐ 新增：待處理問題組別 API
# 任務：查詢清洗時被排除（異常天數 >= 10）的問題組別，供廠商處理
# ============================================================
@app.get("/manage/rejected")
def get_rejected_experiments(db: Session = Depends(get_db)):
    import json
    records = db.query(models.RejectedExperiment).order_by(
        models.RejectedExperiment.upload_filename,
        models.RejectedExperiment.created_at
    ).all()

    result = []
    for rec in records:
        try:
            detail = json.loads(rec.abnormal_detail) if rec.abnormal_detail else []
        except Exception:
            detail = []

        result.append({
            "id": rec.id,
            "upload_filename": rec.upload_filename,
            "product_rd_no": rec.product_rd_no,
            "fragrance_name": rec.fragrance_name,
            "fragrance_pct": rec.fragrance_pct,
            "solvent_1_name": rec.solvent_1_name,
            "solvent_1_pct": rec.solvent_1_pct,  # ⭐ 補上：原本缺少此欄位導致顯示 undefined%
            "solvent_2_name": rec.solvent_2_name,
            "solvent_2_pct": rec.solvent_2_pct,  # ⭐ 補上
            "solvent_3_name": rec.solvent_3_name,
            "solvent_3_pct": rec.solvent_3_pct,  # ⭐ 補上
            "solvent_4_name": rec.solvent_4_name,
            "solvent_4_pct": rec.solvent_4_pct,  # ⭐ 補上
            "solvent_5_name": rec.solvent_5_name,
            "solvent_5_pct": rec.solvent_5_pct,  # ⭐ 補上
            "powder_name": rec.powder_name,
            "powder_pct": rec.powder_pct,         # ⭐ 補上
            "membrane": rec.membrane,
            "capsule_gen": rec.capsule_gen,
            "wind_condition": rec.wind_condition,
            "temperature": rec.temperature,
            "initial_weight": rec.initial_weight,
            "abnormal_day_count": rec.abnormal_day_count,
            "abnormal_detail": detail,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
            "excel_row_start": rec.excel_row_start,
            "excel_row_end": rec.excel_row_end
        })
    return result

# ============================================================
# 📋 歷史紀錄 API：查詢所有刪除操作紀錄
# ============================================================
@app.get("/manage/history")
def get_history(db: Session = Depends(get_db)):
    # ⭐ 修改：加入 upload_excel，讓歷史中心也顯示上傳紀錄
    from sqlalchemy import desc
    logs = db.query(models.OperationLog).filter(
        models.OperationLog.action_type.in_(["delete_group", "delete_file", "upload_excel"])
    ).order_by(desc(models.OperationLog.created_at)).all()

    result = []
    for log in logs:
        # ⭐ 新增：撈出這筆操作對應的備份天數資料，供前端天數細節顯示
        deleted_records = db.query(models.DeletedExperiment).filter(
            models.DeletedExperiment.operation_log_id == log.id
        ).order_by(models.DeletedExperiment.test_days).all()

        timeline = [
            {"day": rec.test_days, "loss": floor_dec(rec.weight_loss)}
            for rec in deleted_records
        ]

        result.append({
            "id": log.id,
            "action_type": log.action_type,
            "target": log.target,
            "affected_rows": log.affected_rows,
            "created_at": log.created_at.isoformat() if log.created_at else None,
            "is_undone": log.is_undone,
            "group_snapshot": log.group_snapshot,
            "timeline": timeline  # ⭐ 新增：這組每天的備份數據
        })
    return result


# ============================================================
# 🔄 歷史紀錄 API：依操作紀錄 ID 復原資料
# ============================================================
@app.post("/manage/undo/{log_id}")
def undo_by_log_id(log_id: int, db: Session = Depends(get_db)):
    global model_status
    try:
        # 找到這筆操作紀錄
        log = db.query(models.OperationLog).filter(
            models.OperationLog.id == log_id
        ).first()

        if not log:
            raise HTTPException(status_code=404, detail="找不到該操作紀錄")

        if log.is_undone == 1:
            raise HTTPException(status_code=400, detail="此操作已經復原過了")

        # 找到這筆操作備份的所有資料
        deleted_records = db.query(models.DeletedExperiment).filter(
            models.DeletedExperiment.operation_log_id == log_id
        ).all()

        if not deleted_records:
            raise HTTPException(status_code=404, detail="找不到備份資料")

        # ⭐ 修改：復原前先一次撈出 B 表現有資料建立比對 set
        # 原因：防止重複復原同一批資料導致 B 表出現重複天數
        # ⭐ 修改：改用全部條件去重，避免溶劑不同但其他條件相同被誤判為重複
        existing_keys = set()
        existing_in_b = db.query(
            models.CleanedExperiment.upload_filename,
            models.CleanedExperiment.fragrance_name,
            models.CleanedExperiment.solvent_1_name,
            models.CleanedExperiment.solvent_2_name,
            models.CleanedExperiment.solvent_3_name,
            models.CleanedExperiment.solvent_4_name,
            models.CleanedExperiment.solvent_5_name,
            models.CleanedExperiment.powder_name,
            models.CleanedExperiment.membrane,
            models.CleanedExperiment.capsule_gen,
            models.CleanedExperiment.wind_condition,
            models.CleanedExperiment.fragrance_pct,
            models.CleanedExperiment.solvent_1_pct,
            models.CleanedExperiment.solvent_2_pct,
            models.CleanedExperiment.solvent_3_pct,
            models.CleanedExperiment.solvent_4_pct,
            models.CleanedExperiment.solvent_5_pct,
            models.CleanedExperiment.powder_pct,
            models.CleanedExperiment.temperature,
            models.CleanedExperiment.initial_weight,
            models.CleanedExperiment.test_days
        ).all()
        for row in existing_in_b:
            existing_keys.add((
                row.upload_filename,
                row.fragrance_name,
                row.solvent_1_name or "None",
                row.solvent_2_name or "None",
                row.solvent_3_name or "None",
                row.solvent_4_name or "None",
                row.solvent_5_name or "None",
                row.powder_name or "None",
                row.membrane or "None",
                row.capsule_gen or "None",
                row.wind_condition or "None",
                round(float(row.fragrance_pct), 2) if row.fragrance_pct else 0,
                round(float(row.solvent_1_pct), 2) if row.solvent_1_pct else 0,
                round(float(row.solvent_2_pct), 2) if row.solvent_2_pct else 0,
                round(float(row.solvent_3_pct), 2) if row.solvent_3_pct else 0,
                round(float(row.solvent_4_pct), 2) if row.solvent_4_pct else 0,
                round(float(row.solvent_5_pct), 2) if row.solvent_5_pct else 0,
                round(float(row.powder_pct), 2) if row.powder_pct else 0,
                round(float(row.temperature), 2) if row.temperature else 0,
                round(float(row.initial_weight), 2) if row.initial_weight else 0,
                row.test_days
            ))

        # ⭐ 新增：印出去重前的狀況
        print(f"🔄 [復原] log_id={log_id}，備份筆數={len(deleted_records)}，B 表現有筆數={len(existing_keys)}")

        # ⭐ 修改：把備份資料搬回 B 表，每筆搬回前先檢查 B 表是否已有重複
        restored_count = 0
        skipped_count = 0  # ⭐ 新增：記錄跳過筆數
        for rec in deleted_records:
            check_key = (
                rec.upload_filename,
                rec.fragrance_name,
                rec.solvent_1_name or "None",
                rec.solvent_2_name or "None",
                rec.solvent_3_name or "None",
                rec.solvent_4_name or "None",
                rec.solvent_5_name or "None",
                rec.powder_name or "None",
                rec.membrane or "None",
                rec.capsule_gen or "None",
                rec.wind_condition or "None",
                round(float(rec.fragrance_pct), 2) if rec.fragrance_pct else 0,
                round(float(rec.solvent_1_pct), 2) if rec.solvent_1_pct else 0,
                round(float(rec.solvent_2_pct), 2) if rec.solvent_2_pct else 0,
                round(float(rec.solvent_3_pct), 2) if rec.solvent_3_pct else 0,
                round(float(rec.solvent_4_pct), 2) if rec.solvent_4_pct else 0,
                round(float(rec.solvent_5_pct), 2) if rec.solvent_5_pct else 0,
                round(float(rec.powder_pct), 2) if rec.powder_pct else 0,
                round(float(rec.temperature), 2) if rec.temperature else 0,
                round(float(rec.initial_weight), 2) if rec.initial_weight else 0,
                rec.test_days
            )
            if check_key in existing_keys:
                # ⭐ 修改：B 表已有這筆，跳過不寫入，避免重複
                skipped_count += 1
                continue

            restored = models.CleanedExperiment(
                upload_filename=rec.upload_filename,
                product_rd_no=rec.product_rd_no,
                fragrance_name=rec.fragrance_name,
                solvent_1_name=rec.solvent_1_name,
                solvent_2_name=rec.solvent_2_name,
                solvent_3_name=rec.solvent_3_name,
                solvent_4_name=rec.solvent_4_name,
                solvent_5_name=rec.solvent_5_name,
                powder_name=rec.powder_name,
                membrane=rec.membrane,
                capsule_gen=rec.capsule_gen,
                wind_condition=rec.wind_condition,
                fragrance_pct=rec.fragrance_pct,
                solvent_1_pct=rec.solvent_1_pct,
                solvent_2_pct=rec.solvent_2_pct,
                solvent_3_pct=rec.solvent_3_pct,
                solvent_4_pct=rec.solvent_4_pct,
                solvent_5_pct=rec.solvent_5_pct,
                powder_pct=rec.powder_pct,
                temperature=rec.temperature,
                initial_weight=rec.initial_weight,
                test_days=rec.test_days,
                solvent_1_exists=rec.solvent_1_exists,
                solvent_2_exists=rec.solvent_2_exists,
                solvent_3_exists=rec.solvent_3_exists,
                solvent_4_exists=rec.solvent_4_exists,
                solvent_5_exists=rec.solvent_5_exists,
                powder_exists=rec.powder_exists,
                weight_loss=rec.weight_loss,
                raw_id=rec.raw_id,
                excel_row_no=rec.excel_row_no  # ⭐ 修改：補回 Excel 行號，避免復原後行號消失
            )
            db.add(restored)
            existing_keys.add(check_key)
            restored_count += 1

        # ⭐ 新增：標記這筆操作已復原
        log.is_undone = 1

        db.commit()
        model_status = "outdated"

        # ⭐ 新增：印出最終結果，讓你在後端 terminal 看到去重是否有效
        print(f"✅ [復原完成] log_id={log_id}，成功復原={restored_count} 筆，去重跳過={skipped_count} 筆")
        if skipped_count > 0:
            print(f"⚠️ [去重生效] 有 {skipped_count} 筆因 B 表已存在相同資料而跳過，避免重複寫入")
        else:
            print(f"✅ [去重正常] 沒有重複資料，全部 {restored_count} 筆都是新寫入")

        return {"status": "success", "restored_rows": restored_count}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


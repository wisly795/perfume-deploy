
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .database import Base

# ============================================================
# 資料庫 A：原始鏡像區 (Raw Table)
# 任務：100% 複製 Excel，作為原始存檔與格式檢查。
# ============================================================
class RawExperiment(Base):
    __tablename__ = "raw_experiments"

    # 自動生成的流水號
    id = Column(Integer, primary_key=True, index=True)

    # 💡 【新增欄位】記錄這筆資料是從哪一個 Excel 檔案上傳的，用於前端分層顯示
    upload_filename = Column(String, index=True)
    
    # --- 原始 Excel 所有欄位 (使用 name 屬性精準對照中文標題) ---
    product_rd_no = Column(String, name="成品品號/RD編號")
    fragrance_no_internal = Column(String, name="香精品號")
    fragrance_name = Column(String, name="香精名稱")
    fragrance_code = Column(String, name="香精編號")
    # ⭐ 修改：將 Float 改為 String，儲存原始文字，避免浮點數誤差
    fragrance_pct = Column(String, name="香精(%)")

    # 溶劑 1~5：位置全部留好，Excel 空白就存入 NULL
    solvent_1_no = Column(String, name="溶劑1品號")
    solvent_1_name = Column(String, name="溶劑1名稱")
    # ⭐ 修改：將 Float 改為 String
    solvent_1_pct = Column(String, name="溶劑1(%)")
    
    solvent_2_no = Column(String, name="溶劑2品號")
    solvent_2_name = Column(String, name="溶劑2名稱")
    # ⭐ 修改：將 Float 改為 String
    solvent_2_pct = Column(String, name="溶劑2(%)")
    
    solvent_3_no = Column(String, name="溶劑3品號")
    solvent_3_name = Column(String, name="溶劑3名稱")
    # ⭐ 修改：將 Float 改為 String
    solvent_3_pct = Column(String, name="溶劑3(%)")
    
    solvent_4_no = Column(String, name="溶劑4品號")
    solvent_4_name = Column(String, name="溶劑4名稱")
    # ⭐ 修改：將 Float 改為 String
    solvent_4_pct = Column(String, name="溶劑4(%)")
    
    solvent_5_no = Column(String, name="溶劑5品號")
    solvent_5_name = Column(String, name="溶劑5名稱")
    # ⭐ 修改：將 Float 改為 String
    solvent_5_pct = Column(String, name="溶劑5(%)")

    powder_no = Column(String, name="稠粉品號")
    powder_name = Column(String, name="稠粉名稱")
    # ⭐ 修改：將 Float 改為 String
    powder_pct = Column(String, name="稠粉(%)")

    # 影響揮發的關鍵 X 特徵 (文字型)
    membrane = Column(String, name="膜料")
    capsule_gen = Column(String, name="膠囊代數")
    wind_condition = Column(String, name="有/無吹風")

    # 影響揮發的關鍵 X 特徵 (數值型)
    # ⭐ 修改：將 Float 改為 String
    temperature = Column(String, name="測試溫度(℃)")
    # ⭐ 修改：將 Float 改為 String
    initial_weight = Column(String, name="初始重量(g)") # 你的動力學核心
    test_days = Column(Integer, name="天數(Days)")     # X 特徵之一（Integer 保持不變）
    
    # 過程數據與答案
    # ⭐ 修改：將 Float 改為 String
    daily_weight = Column(String, name="每日重量(g)")
    # ⭐ 修改：將 Float 改為 String
    weight_loss = Column(String, name="失重(g)")        # 這是 y (Label)
    # ⭐ 修改：將 Float 改為 String
    evap_rate_pct = Column(String, name="揮發速率(%)")

    # 系統欄位：上傳時間
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # 💡 【核心新增】：軟刪除標記 (0: 正常, 1: 已刪除)
    # 任務：實現誤刪復原，資料不物理抹除
    is_deleted = Column(Integer, default=0, index=True)
    is_cleaned = Column(Integer, default=0, index=True)
    # ⭐ 新增：記錄這筆資料在原始 Excel 的實際行號（標題列為第1行，資料從第2行開始）
    excel_row_no = Column(Integer, nullable=True)

# ============================================================
# 資料庫 B：精華預測區 (Cleaned Table)
# 任務：存放 crud.py 清洗後、拉直後、剔除沒失重的純淨數據。
# ============================================================
class CleanedExperiment(Base):
    __tablename__ = "cleaned_experiments"

    id = Column(Integer, primary_key=True, index=True)
    
    # 💡 【新增欄位】同步記錄檔案來源，確保清洗後的數據也能追溯並進行批次刪除
    upload_filename = Column(String, index=True)

    # 修正點 1：增加產品編號，方便你以後回頭找 Excel 哪一筆
    product_rd_no = Column(String)

    # 【名稱軌道】：存中文，送模型前才動態編碼
    fragrance_name = Column(String, index=True) 
    solvent_1_name = Column(String); solvent_2_name = Column(String)
    solvent_3_name = Column(String); solvent_4_name = Column(String)
    solvent_5_name = Column(String)
    powder_name = Column(String)
    membrane = Column(String); capsule_gen = Column(String); wind_condition = Column(String)

    # 【數值軌道】：所有 X 特徵比例與環境
    fragrance_pct = Column(Float)
    solvent_1_pct = Column(Float); solvent_2_pct = Column(Float)
    solvent_3_pct = Column(Float); solvent_4_pct = Column(Float)
    solvent_5_pct = Column(Float)
    powder_pct = Column(Float)
    
    temperature = Column(Float)
    initial_weight = Column(Float) # 關鍵 X 特徵保留
    test_days = Column(Integer)    # 橫轉縱後的關鍵天數
    
    # 【固化特徵】：溶劑存在標籤 (0或1)
    # crud.py 會幫你把「空白」轉化為「0」
    solvent_1_exists = Column(Integer)
    solvent_2_exists = Column(Integer)
    solvent_3_exists = Column(Integer)
    solvent_4_exists = Column(Integer)
    solvent_5_exists = Column(Integer)
    # --- 稠粉標籤 (直接用 powder_exists，絕對不會跟 s 搞混) ---
    powder_exists = Column(Integer, name="powder_exists")
    # 【最終答案】：只有「有失重」的資料才能進來這裡
    weight_loss = Column(Float)
    # ⭐ 新增：對應 RawExperiment 的 id，用於快速同步刪除/復原
    raw_id = Column(Integer, index=True)  # 對應 RawExperiment.id
    # ⭐ 新增：記錄這筆資料在原始 Excel 的實際行號
    excel_row_no = Column(Integer, nullable=True)
    # ⭐ 修改：移除三個獨立刪除標記，改為簡單設計
    # 原因：三個標記造成刪除復原邏輯複雜，按鈕狀態難以維護
    # 新設計：B 表只存活著的資料，刪除時物理移到 DeletedExperiment 備份表

# ============================================================
# ⭐ 新增：操作紀錄表 (Operation Log)
# 任務：記錄所有資料庫異動行為（上傳、刪除、重訓）
# ============================================================
class OperationLog(Base):
    __tablename__ = "operation_logs"

    id = Column(Integer, primary_key=True, index=True)

    # 操作類型：upload / delete_group / delete_file / retrain
    action_type = Column(String, index=True)

    # 影響對象（例如檔名或配方名稱）
    target = Column(String)

    # 影響筆數
    affected_rows = Column(Integer)

    # 操作時間
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # ⭐ 修改：is_deleted 改名為 is_undone，語意更清楚（0=未復原，1=已復原）
    is_undone = Column(Integer, default=0, index=True)
    # ⭐ 新增：儲存被刪組別的配方快照（JSON 字串），供歷史紀錄區顯示
    group_snapshot = Column(String, nullable=True)


# ============================================================
# ⭐ 新增：已刪除實驗備份表 (DeletedExperiment)
# 任務：儲存從 B 表物理刪除的資料，供歷史紀錄區復原使用
# ============================================================
class DeletedExperiment(Base):
    __tablename__ = "deleted_experiments"

    id = Column(Integer, primary_key=True, index=True)
    upload_filename = Column(String, index=True)
    product_rd_no = Column(String)
    fragrance_name = Column(String)
    solvent_1_name = Column(String); solvent_2_name = Column(String)
    solvent_3_name = Column(String); solvent_4_name = Column(String)
    solvent_5_name = Column(String)
    powder_name = Column(String)
    membrane = Column(String); capsule_gen = Column(String); wind_condition = Column(String)
    fragrance_pct = Column(Float)
    solvent_1_pct = Column(Float); solvent_2_pct = Column(Float)
    solvent_3_pct = Column(Float); solvent_4_pct = Column(Float)
    solvent_5_pct = Column(Float)
    powder_pct = Column(Float)
    temperature = Column(Float)
    initial_weight = Column(Float)
    test_days = Column(Integer)
    solvent_1_exists = Column(Integer); solvent_2_exists = Column(Integer)
    solvent_3_exists = Column(Integer); solvent_4_exists = Column(Integer)
    solvent_5_exists = Column(Integer)
    powder_exists = Column(Integer)
    weight_loss = Column(Float)
    raw_id = Column(Integer)
    # ⭐ 新增：記錄這筆資料是被哪一筆 OperationLog 刪除的
    operation_log_id = Column(Integer, index=True)
    # ⭐ 新增：保留 Excel 行號，讓復原後行號不消失
    excel_row_no = Column(Integer, nullable=True)
    deleted_at = Column(DateTime(timezone=True), server_default=func.now())

# ============================================================
# ⭐ 新增：被排除組別表 (RejectedExperiment)
# 任務：儲存清洗時異常天數 >= 10 天、不納入 B 表訓練的問題組別
# 供前端「待處理問題組別」按鈕查詢，讓廠商對應 Excel 行號處理
# ============================================================
class RejectedExperiment(Base):
    __tablename__ = "rejected_experiments"

    id = Column(Integer, primary_key=True, index=True)

    # 來源 Excel 檔名
    upload_filename = Column(String, index=True)

    # 配方資訊（對應 CleanedExperiment 欄位，供廠商識別是哪一組）
    product_rd_no = Column(String)
    fragrance_name = Column(String)
    solvent_1_name = Column(String)
    solvent_2_name = Column(String)
    solvent_3_name = Column(String)
    solvent_4_name = Column(String)
    solvent_5_name = Column(String)
    powder_name = Column(String)
    membrane = Column(String)
    capsule_gen = Column(String)
    wind_condition = Column(String)
    fragrance_pct = Column(Float)
    solvent_1_pct = Column(Float)
    solvent_2_pct = Column(Float)
    solvent_3_pct = Column(Float)
    solvent_4_pct = Column(Float)
    solvent_5_pct = Column(Float)
    powder_pct = Column(Float)
    temperature = Column(Float)
    initial_weight = Column(Float)

    # ⭐ 異常統計：異常天數總數
    abnormal_day_count = Column(Integer)

    # ⭐ 異常細節：JSON 字串，記錄每一個異常天的天數、失重值、異常原因
    # 格式範例：[{"day": 5, "loss": -0.02, "reason": "失重為負數"}, ...]
    abnormal_detail = Column(String)

    # ⭐ 新增：這組資料在原始 Excel 的行號範圍
    excel_row_start = Column(Integer, nullable=True)
    excel_row_end = Column(Integer, nullable=True)

    # 建立時間
    created_at = Column(DateTime(timezone=True), server_default=func.now())
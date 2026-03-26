



import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from . import models

def floor_dec(x):
    """
    將數值強制截斷至小數點後兩位。
    加入 0.0000001 的微小補償值，防止電腦因浮點數誤差將 25.00 誤判為 24.999999。
    """
    if x is None: return 0.0
    try:
        # ✅ 修改點：在乘以 100 之前先加上一個極小補償值 0.0000001
        # 這樣可以確保 int() 轉換時，24.999999999 會被推回 25.00，避免 0.01 的誤差
        return int(float(x) * 100 + 0.0000001) / 100.0
    except (ValueError, TypeError):
        return 0.0
# ============================================================
# 1. 【新增的搬運工】：負責把 Excel 原始數據塞進資料庫 A
# ============================================================
import re # 💡 記得在檔案最上方 import re，或者直接用這段

def batch_insert_raw(db: Session, df: pd.DataFrame, filename: str):
    """
    將 DataFrame 的資料批次寫入資料庫 A (RawExperiment)。
    ⭐ 修改：A 表不再做去重，無腦全部寫入。
    去重責任完全移交給 B 表的 sync_raw_to_cleaned()。
    原因：A 表去重加了 is_deleted==0 條件，導致刪除再重傳時失效，累積重複資料。
    """
    # 檔名安全性清洗
    # 使用正規表達式只保留字母、數字、點、底線、橫線，將其他特殊字元（如 & 空白 %）替換為底線
    # 這樣可以防止廠商上傳奇怪檔名導致後端刪除 API 找不到檔案的問題
    filename = re.sub(r'[^\w\.\-]', '_', filename)

    # 處理 Pandas 的空值，將 NaN 轉為 Python 的 None
    df = df.replace({np.nan: None})

    raw_records = []
    duplicate_count = 0  # ⭐ 修改：A 表不再去重，此值永遠為 0，保留是為了不影響 main.py 的接收格式

    # ⭐ 修改：改用 enumerate 同時取得 index，計算 Excel 實際行號
    # index 從 0 開始，第 1 行是標題，所以 excel_row_no = index + 2
    for index, row in df.iterrows():
        # ⭐ 修改：移除所有去重查詢邏輯，直接準備寫入
        # 不重複或失重不同（情況 2），則視為新數據準備存入
        record = models.RawExperiment(
            upload_filename=filename,
            excel_row_no=index + 2,  # ⭐ 新增：存入 Excel 實際行號
            product_rd_no=str(row.get("成品品號/RD編號")) if row.get("成品品號/RD編號") else None,
            fragrance_no_internal=str(row.get("香精品號")) if row.get("香精品號") else None,
            fragrance_name=str(row.get("香精名稱")) if row.get("香精名稱") else None,
            fragrance_code=str(row.get("香精編號")) if row.get("香精編號") else None,
            # ⭐ 修改：所有數值欄位存文字（保持原始格式）
            fragrance_pct=str(row.get("香精(%)")) if row.get("香精(%)") is not None else None,

            solvent_1_no=str(row.get("溶劑1品號")) if row.get("溶劑1品號") else None,
            solvent_1_name=str(row.get("溶劑1名稱")) if row.get("溶劑1名稱") else None,
            solvent_1_pct=str(row.get("溶劑1(%)")) if row.get("溶劑1(%)") is not None else None,

            solvent_2_no=str(row.get("溶劑2品號")) if row.get("溶劑2品號") else None,
            solvent_2_name=str(row.get("溶劑2名稱")) if row.get("溶劑2名稱") else None,
            solvent_2_pct=str(row.get("溶劑2(%)")) if row.get("溶劑2(%)") is not None else None,

            solvent_3_no=str(row.get("溶劑3品號")) if row.get("溶劑3品號") else None,
            solvent_3_name=str(row.get("溶劑3名稱")) if row.get("溶劑3名稱") else None,
            solvent_3_pct=str(row.get("溶劑3(%)")) if row.get("溶劑3(%)") is not None else None,

            solvent_4_no=str(row.get("溶劑4品號")) if row.get("溶劑4品號") else None,
            solvent_4_name=str(row.get("溶劑4名稱")) if row.get("溶劑4名稱") else None,
            solvent_4_pct=str(row.get("溶劑4(%)")) if row.get("溶劑4(%)") is not None else None,

            solvent_5_no=str(row.get("溶劑5品號")) if row.get("溶劑5品號") else None,
            solvent_5_name=str(row.get("溶劑5名稱")) if row.get("溶劑5名稱") else None,
            solvent_5_pct=str(row.get("溶劑5(%)")) if row.get("溶劑5(%)") is not None else None,

            powder_no=str(row.get("稠粉品號")) if row.get("稠粉品號") else None,
            powder_name=str(row.get("稠粉名稱")) if row.get("稠粉名稱") else None,
            powder_pct=str(row.get("稠粉(%)")) if row.get("稠粉(%)") is not None else None,

            membrane=str(row.get("膜料")) if row.get("膜料") else None,
            capsule_gen=str(row.get("膠囊代數")) if row.get("膠囊代數") else None,

            temperature=str(row.get("測試溫度(℃)")) if row.get("測試溫度(℃)") is not None else None,
            wind_condition=str(row.get("有/無吹風")) if row.get("有/無吹風") else None,
            initial_weight=str(row.get("初始重量(g)")) if row.get("初始重量(g)") is not None else None,
            test_days=row.get("天數(Days)"),  # test_days 是 Integer，保持原樣
            daily_weight=str(row.get("每日重量(g)")) if row.get("每日重量(g)") is not None else None,
            weight_loss=str(row.get("失重(g)")) if row.get("失重(g)") is not None else None,
            evap_rate_pct=str(row.get("揮發速率(%)")) if row.get("揮發速率(%)") is not None else None
        )
        raw_records.append(record)

    # 執行批次寫入
    if raw_records:
        db.bulk_save_objects(raw_records)
        db.commit()

    return len(raw_records), duplicate_count  # 回傳新存入筆數與重複跳過筆數



# ============================================================
# 2. 【資料清洗】：將 RawExperiment 整理後寫入 CleanedExperiment
# ============================================================

def sync_raw_to_cleaned(db: Session, filename: str):
    """
    將 RawExperiment 中尚未清洗的資料寫入 CleanedExperiment，
    並將成功清洗的原始資料標記為 is_cleaned = 1。
    ⭐ 修改：
    1. 移除 upload_filename 過濾，改為只過濾 is_cleaned == 0。
    2. 新增 B 表批次去重邏輯。
    3. ⭐ 新增：分組判斷異常天數邏輯：
       - 異常天數 < 10：用前一天的值取代後寫入 B 表
       - 異常天數 >= 10：整組不寫入 B 表，改寫入 RejectedExperiment 表
    """
    import json

    raw_items = db.query(models.RawExperiment).filter(
        models.RawExperiment.is_cleaned == 0
    ).all()

    all_cleaned = db.query(models.CleanedExperiment).all()
    cleaned_map = {}
    for rec in all_cleaned:
        # ⭐ 修改：移除 upload_filename，加入 weight_loss
        # 原因：跨 Excel 去重時不看來源檔名，只看實驗條件與結果是否完全一致
        # 失重值不同 → 視為不同實驗結果，兩筆都保留
        key = (
            rec.fragrance_name or "None",
            rec.solvent_1_name or "None",
            rec.solvent_2_name or "None",
            rec.solvent_3_name or "None",
            rec.solvent_4_name or "None",
            rec.solvent_5_name or "None",
            rec.powder_name or "None",
            rec.membrane or "None",
            rec.capsule_gen or "None",
            rec.wind_condition or "None",
            floor_dec(rec.fragrance_pct),
            floor_dec(rec.solvent_1_pct),
            floor_dec(rec.solvent_2_pct),
            floor_dec(rec.solvent_3_pct),
            floor_dec(rec.solvent_4_pct),
            floor_dec(rec.solvent_5_pct),
            floor_dec(rec.powder_pct),
            floor_dec(rec.temperature),
            floor_dec(rec.initial_weight),
            rec.test_days,
            floor_dec(rec.weight_loss)  # ⭐ 新增：失重值不同視為不同實驗
        )
        cleaned_map.setdefault(key, rec)

    # ⭐ 新增：先把所有 A 表資料按「配方組別」分組，再整組判斷異常天數
    # 原因：必須看整組所有天的數據才能判斷異常天數是否達到 10 天
    # group_key 不含 test_days，讓同一組的所有天聚集在同一個 key 下
    group_map = {}  # group_key -> list of (item, temperature_val, initial_weight_val, weight_loss_val, s_list)

    for item in raw_items:
        try:
            temperature_val = float(item.temperature) if item.temperature is not None else None
            initial_weight_val = float(item.initial_weight) if item.initial_weight is not None else None
            daily_weight_val = float(item.daily_weight) if item.daily_weight is not None else None
            weight_loss_val = float(item.weight_loss) if item.weight_loss is not None else None
        except (ValueError, TypeError):
            continue

        must_have_values = [
            temperature_val,
            item.wind_condition,
            initial_weight_val,
            item.test_days,
            daily_weight_val,
            weight_loss_val
        ]

        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in must_have_values):
            continue

        if initial_weight_val <= 0:
            continue

        # ⭐ 新增：過濾失重大於初始重量的物理不合理資料
        # 原因：失重不可能超過初始重量，這類資料會誤導 AI 學到錯誤的物理規律
        if weight_loss_val is not None and weight_loss_val > initial_weight_val:
            continue

        s_list = [
            {"n": item.solvent_1_name or "None",
             "p": float(item.solvent_1_pct) if item.solvent_1_pct is not None else 0.0},
            {"n": item.solvent_2_name or "None",
             "p": float(item.solvent_2_pct) if item.solvent_2_pct is not None else 0.0},
            {"n": item.solvent_3_name or "None",
             "p": float(item.solvent_3_pct) if item.solvent_3_pct is not None else 0.0},
            {"n": item.solvent_4_name or "None",
             "p": float(item.solvent_4_pct) if item.solvent_4_pct is not None else 0.0},
            {"n": item.solvent_5_name or "None",
             "p": float(item.solvent_5_pct) if item.solvent_5_pct is not None else 0.0}
        ]

        # ⭐ 新增：group_key 不含 test_days，讓同一組所有天聚在一起
        group_key = (
            item.upload_filename,
            item.fragrance_name or "None",
            s_list[0]["n"], s_list[1]["n"], s_list[2]["n"], s_list[3]["n"], s_list[4]["n"],
            item.powder_name or "None",
            item.membrane or "None",
            item.capsule_gen or "None",
            item.wind_condition or "None",
            floor_dec(float(item.fragrance_pct) if item.fragrance_pct is not None else 0),
            floor_dec(s_list[0]["p"]), floor_dec(s_list[1]["p"]),
            floor_dec(s_list[2]["p"]), floor_dec(s_list[3]["p"]), floor_dec(s_list[4]["p"]),
            floor_dec(float(item.powder_pct) if item.powder_pct is not None else 0),
            floor_dec(temperature_val),
            floor_dec(initial_weight_val)
        )

        if group_key not in group_map:
            group_map[group_key] = []
        group_map[group_key].append({
            "item": item,
            "temperature_val": temperature_val,
            "initial_weight_val": initial_weight_val,
            "weight_loss_val": weight_loss_val,
            "s_list": s_list
        })

    cleaned_entries = []
    cleaned_raw_ids = []
    b_table_duplicate_count = 0  # ⭐ 新增：統計 B 表去重跳過筆數的計數器

    for group_key, rows in group_map.items():
        # ⭐ 新增：依天數排序，確保前後天比較正確
        rows.sort(key=lambda x: x["item"].test_days)

        # ⭐ 新增：掃描這組所有天，統計異常天數與細節
        abnormal_days = []
        for i, row in enumerate(rows):
            loss = row["weight_loss_val"]
            day = row["item"].test_days
            reason = None

            if loss < 0:
                reason = "失重為負數"
            elif i > 0:
                prev_loss = rows[i - 1]["weight_loss_val"]
                if loss < prev_loss:
                    reason = f"失重({loss}g)小於前一天({prev_loss}g)"

            if reason:
                abnormal_days.append({"day": day, "loss": loss, "reason": reason})

        abnormal_count = len(abnormal_days)

        # ⭐ 新增：情況 B — 異常天數 >= 10，整組寫入 RejectedExperiment，不進 B 表
        if abnormal_count >= 10:
            first = rows[0]
            item = first["item"]
            s_list = first["s_list"]

            # ⭐ 新增：同一組若已存在 RejectedExperiment 則跳過（避免重複寫入）
            existing_rejected = db.query(models.RejectedExperiment).filter(
                models.RejectedExperiment.upload_filename == item.upload_filename,
                models.RejectedExperiment.fragrance_name == (item.fragrance_name or "None"),
                models.RejectedExperiment.temperature == floor_dec(first["temperature_val"]),
                models.RejectedExperiment.initial_weight == floor_dec(first["initial_weight_val"])
            ).first()

            if existing_rejected is None:
                rejected = models.RejectedExperiment(
                    upload_filename=item.upload_filename,
                    product_rd_no=item.product_rd_no,
                    fragrance_name=item.fragrance_name or "None",
                    solvent_1_name=s_list[0]["n"],
                    solvent_2_name=s_list[1]["n"],
                    solvent_3_name=s_list[2]["n"],
                    solvent_4_name=s_list[3]["n"],
                    solvent_5_name=s_list[4]["n"],
                    powder_name=item.powder_name or "None",
                    membrane=item.membrane or "None",
                    capsule_gen=item.capsule_gen or "None",
                    wind_condition=item.wind_condition or "None",
                    fragrance_pct=floor_dec(float(item.fragrance_pct) if item.fragrance_pct is not None else 0),
                    solvent_1_pct=floor_dec(s_list[0]["p"]),
                    solvent_2_pct=floor_dec(s_list[1]["p"]),
                    solvent_3_pct=floor_dec(s_list[2]["p"]),
                    solvent_4_pct=floor_dec(s_list[3]["p"]),
                    solvent_5_pct=floor_dec(s_list[4]["p"]),
                    powder_pct=floor_dec(float(item.powder_pct) if item.powder_pct is not None else 0),
                    temperature=floor_dec(first["temperature_val"]),
                    initial_weight=floor_dec(first["initial_weight_val"]),
                    abnormal_day_count=abnormal_count,
                    abnormal_detail=json.dumps(abnormal_days, ensure_ascii=False),
                    # ⭐ 新增：從這組所有天的 excel_row_no 取最小和最大值
                    excel_row_start=min(
                        (r["item"].excel_row_no for r in rows if r["item"].excel_row_no is not None),
                        default=None
                    ),
                    excel_row_end=max(
                        (r["item"].excel_row_no for r in rows if r["item"].excel_row_no is not None),
                        default=None
                    )
                )
                db.add(rejected)

            # ⭐ 整組 A 表全部標記已清洗（已處理，不再重複掃描）
            for row in rows:
                cleaned_raw_ids.append(row["item"].id)
            continue

        # ⭐ 新增：情況 A — 異常天數 < 10，用前一天的值取代後寫入 B 表
        prev_loss = None
        for i, row in enumerate(rows):
            item = row["item"]
            s_list = row["s_list"]
            temperature_val = row["temperature_val"]
            initial_weight_val = row["initial_weight_val"]
            loss = row["weight_loss_val"]

            # ⭐ 新增：異常時用前一天的值取代，第 0 天異常則用 0 取代
            is_abnormal = (loss < 0) or (prev_loss is not None and loss < prev_loss)
            if is_abnormal:
                loss = prev_loss if prev_loss is not None else 0.0

            lookup_key = (
                # ⭐ 修改：移除 upload_filename，加入 loss（取代後的值）
                # 原因：與 cleaned_map 的 key 結構保持一致，確保跨 Excel 去重正確運作
                item.fragrance_name or "None",
                s_list[0]["n"], s_list[1]["n"], s_list[2]["n"], s_list[3]["n"], s_list[4]["n"],
                item.powder_name or "None",
                item.membrane or "None",
                item.capsule_gen or "None",
                item.wind_condition or "None",
                floor_dec(float(item.fragrance_pct) if item.fragrance_pct is not None else 0),
                floor_dec(s_list[0]["p"]), floor_dec(s_list[1]["p"]),
                floor_dec(s_list[2]["p"]), floor_dec(s_list[3]["p"]), floor_dec(s_list[4]["p"]),
                floor_dec(float(item.powder_pct) if item.powder_pct is not None else 0),
                floor_dec(temperature_val),
                floor_dec(initial_weight_val),
                item.test_days,
                floor_dec(loss)  # ⭐ 新增：失重值（已經過異常修正的值）
            )

            existing_cleaned = cleaned_map.get(lookup_key)
            if existing_cleaned is not None:
                cleaned_raw_ids.append(item.id)
                prev_loss = loss
                # ⭐ 新增：統計 B 表去重跳過的筆數
                b_table_duplicate_count += 1
                continue

            cleaned = models.CleanedExperiment(
                upload_filename=item.upload_filename,
                product_rd_no=item.product_rd_no,
                fragrance_name=item.fragrance_name or "None",
                solvent_1_name=s_list[0]["n"],
                solvent_2_name=s_list[1]["n"],
                solvent_3_name=s_list[2]["n"],
                solvent_4_name=s_list[3]["n"],
                solvent_5_name=s_list[4]["n"],
                powder_name=item.powder_name or "None",
                membrane=item.membrane or "None",
                capsule_gen=item.capsule_gen or "None",
                wind_condition=item.wind_condition or "None",
                fragrance_pct=floor_dec(float(item.fragrance_pct) if item.fragrance_pct is not None else 0),
                solvent_1_pct=floor_dec(s_list[0]["p"]),
                solvent_2_pct=floor_dec(s_list[1]["p"]),
                solvent_3_pct=floor_dec(s_list[2]["p"]),
                solvent_4_pct=floor_dec(s_list[3]["p"]),
                solvent_5_pct=floor_dec(s_list[4]["p"]),
                powder_pct=floor_dec(float(item.powder_pct) if item.powder_pct is not None else 0),
                temperature=floor_dec(temperature_val),
                initial_weight=floor_dec(initial_weight_val),
                test_days=item.test_days,
                solvent_1_exists=1 if s_list[0]["n"] != "None" else 0,
                solvent_2_exists=1 if s_list[1]["n"] != "None" else 0,
                solvent_3_exists=1 if s_list[2]["n"] != "None" else 0,
                solvent_4_exists=1 if s_list[3]["n"] != "None" else 0,
                solvent_5_exists=1 if s_list[4]["n"] != "None" else 0,
                powder_exists=1 if item.powder_name else 0,
                # ⭐ 新增：寫入取代後的 loss（異常時已被修正）
                weight_loss=floor_dec(loss),
                raw_id=item.id,
                # ⭐ 新增：從 A 表同步帶入 Excel 行號
                excel_row_no=item.excel_row_no
            )

            cleaned_entries.append(cleaned)
            cleaned_raw_ids.append(item.id)
            prev_loss = loss

    if cleaned_entries:
        db.add_all(cleaned_entries)
        db.flush()
        db.query(models.RawExperiment).filter(
            models.RawExperiment.id.in_(cleaned_raw_ids)
        ).update({"is_cleaned": 1}, synchronize_session=False)
        db.commit()
    elif cleaned_raw_ids:
        db.query(models.RawExperiment).filter(
            models.RawExperiment.id.in_(cleaned_raw_ids)
        ).update({"is_cleaned": 1}, synchronize_session=False)
        db.commit()
    else:
        # ⭐ 新增：即使全部是 rejected，也要 commit RejectedExperiment 的寫入
        db.commit()

    # ⭐ 新增：統計清洗報告數據並回傳
    # 統計異常天數被修正的筆數（情況A：用前一天取代）
    abnormal_corrected = sum(
        1 for group_rows in group_map.values()
        for row in group_rows
        if len([r for r in group_rows if r["item"].id == row["item"].id]) > 0
        and (row["weight_loss_val"] < 0 or (
            group_rows.index(row) > 0 and
            row["weight_loss_val"] < group_rows[group_rows.index(row) - 1]["weight_loss_val"]
        ))
    )

    # 統計整組被排除的組數（情況B：異常天數>=10）
    rejected_groups = db.query(models.RejectedExperiment).filter(
        models.RejectedExperiment.upload_filename == filename
    ).count()

    # 統計 B 表中這次新增的筆數
    cleaned_inserted = len(cleaned_entries)

    return {
        "cleaned_inserted": cleaned_inserted,              # 成功寫入 B 表的筆數
        "abnormal_corrected": abnormal_corrected,          # 異常天數被自動修正的筆數
        "rejected_groups": rejected_groups,                # 整組被排除的組數
        "duplicate_skipped": b_table_duplicate_count,      # ⭐ 新增：B 表去重跳過的筆數
    }

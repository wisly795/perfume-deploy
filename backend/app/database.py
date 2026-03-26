
# backend/app/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os 

# ============================================================
# ⭐ 修改原因
#
# 原本：
# SQLite 本地資料庫
#
# 問題：
# SQLite 不適合部署
#
# 修改後：
# 本地與部署全部使用 PostgreSQL
# ============================================================


# ============================================================
# ⭐ PostgreSQL 連線設定
#
# 本地：
# postgresql+psycopg2://postgres:123456@localhost:5432/perfume_db
#
# 部署：
# Fly / Render 會提供 DATABASE_URL
# ============================================================

# 新增/修改點：確保密碼正確，並支援環境變數
# 如果你的 pgAdmin 密碼不是 123456，請將下方的 123456 改掉
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:asdf7410@localhost:5432/perfume_db"
)


# ============================================================
# ⭐ 修改 create_engine
#
# 原本 SQLite：
#
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL,
#     connect_args={"check_same_thread": False}
# )
#
# 修改後：PostgreSQL 不需要 connect_args (這是給 SQLite 用的線程設定)
# ============================================================

# 修改點：直接傳入 URL 即可，移除 connect_args
engine = create_engine(DATABASE_URL)

# ============================================================
# ⭐ 修改原因：修復 ImportError: cannot import name 'Base'
# 
# 本來為什麼：
# 只有定義 engine，卻沒有建立 Base 物件與 SessionLocal。
# 這會導致 models.py 在執行時找不到父類別，main.py 也無法建立連線。
# 
# 修改後為什麼：
# 1. 建立 SessionLocal：讓 FastAPI 的 get_db 函數可以正常運作。
# 2. 建立 Base：讓 models.py 的資料表模型可以正確繼承並建立表格。
# ============================================================
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

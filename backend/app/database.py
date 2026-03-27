from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os 

# ✅ 強制使用 Railway DATABASE_URL（沒有就直接報錯）
DATABASE_URL = os.environ["DATABASE_URL"]

# ✅ 建立 engine（避免斷線問題）
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

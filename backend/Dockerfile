FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# ⭐ 建立模型資料夾（對應你的程式碼）
RUN mkdir -p /app/models_storage

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
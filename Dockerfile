FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# HF cache helye (a RunPodon ide fogjuk mountolni a volume-ot)
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# teljes kód
COPY . .

# Queue: worker indul, nem HTTP
CMD ["python", "-u", "worker.py"]

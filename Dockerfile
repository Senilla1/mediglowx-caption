FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# másoljuk az egész appot
COPY . .

# FastAPI default port a RunPodon
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -f http://localhost:8080/healthz || exit 1

# nálad app.py-ben az app neve app:app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

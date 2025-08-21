# ---------- Base ----------
FROM python:3.10-slim

# Opcionális rendszercsomagok (kicsi image + kerek pip fordításokhoz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && rm -rf /var/lib/apt/lists/*

# ---------- Hugging Face cache -> Network Volume ----------
# A Serverless Queue workerbe a Network Volume automatikusan /runpod-volume alá mountolódik
ENV HF_HOME=/runpod-volume/.cache/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/.cache/huggingface/hub \
    TORCH_HOME=/runpod-volume/.cache/torch \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# ---------- App ----------
WORKDIR /app

# Ha requirements.txt a repo gyökerében van:
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Források
COPY . .

# ---------- Start (Queue / worker) ----------
# A worker.py-ben legyen RunPod handler és runpod.start()
CMD ["python", "-u", "worker.py"]

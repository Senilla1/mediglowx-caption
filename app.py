# app.py
import os
import time
from io import BytesIO

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

APP_START_TS = time.time()

# ---------- FastAPI ----------
app = FastAPI(title="MediGlowX BLIP-1 Caption API", version="1.0.0")

# ---------- Request/Response schemas ----------
class CaptionRequest(BaseModel):
    id: str
    image: HttpUrl

class CaptionResponse(BaseModel):
    id: str
    caption: str
    model_ready_at: float

# ---------- Model load at startup (no device_map!) ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("BLIP_MODEL", "Salesforce/blip-image-captioning-large")

processor: BlipProcessor | None = None
model: BlipForConditionalGeneration | None = None
MODEL_READY_AT = 0.0

def load_model():
    global processor, model, MODEL_READY_AT
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval().to(DEVICE)
    # Warm-up with a tiny black image so the first real call is faster
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)
    MODEL_READY_AT = time.time()

load_model()  # <-- betöltjük induláskor

# ---------- Helpers ----------
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

async def fetch_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {r.status_code}")
        try:
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid image file: {e}")

# ---------- Endpoints ----------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": model is not None and processor is not None,
        "model_ready_at": MODEL_READY_AT,
        "ts": time.time(),
    }

@app.post("/caption", response_model=CaptionResponse)
async def caption(body: CaptionRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    img = await fetch_image(str(body.image))

    # BLIP-1 inference
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=30,   # hosszabb leírás
            num_beams=3,         # minőség vs. sebesség balansz
            repetition_penalty=1.1
        )
    text = processor.decode(out_ids[0], skip_special_tokens=True).strip()

    return CaptionResponse(id=body.id, caption=text, model_ready_at=MODEL_READY_AT)

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

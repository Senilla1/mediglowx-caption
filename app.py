# app.py
import io
import os
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# ====== FastAPI app ======
app = FastAPI(title="MediGlowX BLIP-2 Caption API", version="1.0.0")

# ====== Globális modell változók ======
MODEL = None
PROCESSOR = None
MODEL_READY_AT: Optional[float] = None


def load_model():
    """BLIP-2 modell és processor betöltése induláskor."""
    global MODEL, PROCESSOR, MODEL_READY_AT
    if MODEL is not None and PROCESSOR is not None:
        return MODEL, PROCESSOR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading BLIP-2 model on {device}...")

    MODEL = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    PROCESSOR = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

    MODEL_READY_AT = time.time()
    print("[INFO] BLIP-2 model loaded successfully.")
    return MODEL, PROCESSOR


@app.on_event("startup")
def _startup():
    load_model()


class CaptionRequest(BaseModel):
    id: str
    image: HttpUrl


class CaptionResponse(BaseModel):
    id: str
    caption: str
    model_ready_at: Optional[float] = None


async def fetch_image_bytes(url: str, timeout_s: int = 30) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": MODEL is not None,
        "model_ready_at": MODEL_READY_AT,
        "ts": time.time(),
    }


@app.post("/caption", response_model=CaptionResponse)
async def caption(body: CaptionRequest):
    model, processor = load_model()

    img_bytes = await fetch_image_bytes(str(body.image))
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    caption_text = processor.decode(generated_ids[0], skip_special_tokens=True)

    return CaptionResponse(
        id=body.id,
        caption=caption_text,
        model_ready_at=MODEL_READY_AT,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1)

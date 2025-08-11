# app.py  — BLIP (helyi, gyors) változat
import os
import io
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image

from transformers import pipeline

APP_STARTED_AT = time.time()
MODEL_NAME = os.getenv("MODEL_NAME", "Salesforce/blip-image-captioning-large")  # << BLIP
MODEL_STATE = {"pipe": None, "ready_at": None}

app = FastAPI(title="MediGlowX BLIP Caption API", version="1.0.0")


# ----- Model betöltés induláskor ------------------------------------------------
def load_model():
    # image-to-text pipeline BLIP-pel (CPU-n is gyors)
    pipe = pipeline(
        task="image-to-text",
        model=MODEL_NAME,
        device_map="auto"  # CPU-n is OK
    )
    MODEL_STATE["pipe"] = pipe
    MODEL_STATE["ready_at"] = time.time()


@app.on_event("startup")
def _startup():
    load_model()


# ----- Sémák -------------------------------------------------------------------
class CaptionRequest(BaseModel):
    id: str
    image: HttpUrl  # Cloudinary secure_url erősen ajánlott


class CaptionResponse(BaseModel):
    id: str
    caption: str
    model_ready_at: float


# ----- Segédfüggvények ---------------------------------------------------------
async def fetch_image(url: str) -> Image.Image:
    # TIPP: Cloudinary-n kérd kicsiben: .../upload/w_768/...
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Image download failed: {r.status_code}")
        img_bytes = io.BytesIO(r.content)
    img = Image.open(img_bytes).convert("RGB")
    return img


def generate_caption(pil_image: Image.Image) -> str:
    pipe = MODEL_STATE["pipe"]
    out = pipe(pil_image, max_new_tokens=60)
    # BLIP output formátuma: [{'generated_text': '...'}]
    text = (out[0].get("generated_text") or "").strip()
    return text


# ----- Endpontok ---------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": MODEL_STATE["pipe"] is not None,
        "model_ready_at": MODEL_STATE["ready_at"],
        "ts": time.time(),
    }


@app.post("/caption", response_model=CaptionResponse)
async def caption(body: CaptionRequest):
    # kép letöltés
    img = await fetch_image(str(body.image))
    # felirat generálás
    text = generate_caption(img)
    if not text:
        raise HTTPException(status_code=500, detail="Empty caption")
    return CaptionResponse(
        id=body.id,
        caption=text,
        model_ready_at=MODEL_STATE["ready_at"] or APP_STARTED_AT,
    )

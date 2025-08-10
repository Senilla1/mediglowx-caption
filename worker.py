# worker.py
import os, io, logging
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("worker")

MODEL_ID = os.getenv("MODEL_ID", "Salesforce/blip-image-captioning-large")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log.info(f"Loading model {MODEL_ID} on {DEVICE}…")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()

class Req(BaseModel):
    id: str
    image: HttpUrl

app = FastAPI()

@app.post("/analyze")
async def analyze(req: Req):
    # 1) letöltés
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            r = await client.get(str(req.image))
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        log.error(f"download failed: {e}")
        raise HTTPException(502, "Image download failed")

    # 2) prompt + generálás (hosszabb, célzott leíráshoz)
    prompt = (
        "Describe the face skin in 3-5 sentences: tone, redness, pores, acne, "
        "wrinkles, pigmentation, dryness or oiliness. Be specific but neutral."
    )

    inputs = processor(images=img, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,            # hosszabb szöveg
            num_beams=4,                  # minőség
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    if not caption:
        caption = "No clear findings."

    return {"caption": caption}

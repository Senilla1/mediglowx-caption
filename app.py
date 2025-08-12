# app.py
import os
import time
from io import BytesIO
from typing import List, Optional, Dict, Any

import httpx
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    BlipForQuestionAnswering,  # alias of BlipForConditionalGeneration for VQA ckpt
)

APP_START_TS = time.time()

# ------------------------------
# FastAPI
# ------------------------------
app = FastAPI(title="MediGlowX BLIP API", version="1.0.0")

# ------------------------------
# Schemas
# ------------------------------
class CaptionRequest(BaseModel):
    id: str
    image: HttpUrl

class CaptionResponse(BaseModel):
    id: str
    caption: str
    model_ready_at: float

class AnalyzeRequest(BaseModel):
    id: str
    image: HttpUrl
    questions: Optional[List[str]] = None  # ha None, a default 19 kérdést használjuk

class AnalyzeResponse(BaseModel):
    id: str
    answers: Dict[str, str]           # {"Fine lines visible?": "yes/no/unsure"}
    leds: List[str]                   # ajánlott LED módok
    rationale: str                    # rövid összefoglaló
    model_ready_at: float

# ------------------------------
# Model config
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP-1 caption
CAPTION_MODEL_NAME = os.environ.get("BLIP_CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
caption_processor: Optional[BlipProcessor] = None
caption_model: Optional[BlipForConditionalGeneration] = None

# BLIP VQA
VQA_MODEL_NAME = os.environ.get("BLIP_VQA_MODEL", "Salesforce/blip-vqa-base")
vqa_processor: Optional[AutoProcessor] = None
vqa_model: Optional[BlipForQuestionAnswering] = None

MODEL_READY_AT: float = 0.0

# Default 19 kérdés (a te listád)
DEFAULT_QUESTIONS = [
    "Fine lines visible?",
    "Skin looks dull?",
    "Uneven skin tone?",
    "Deep wrinkles visible?",
    "Skin looks saggy?",
    "Skin looks tired?",
    "Acne or pimples visible?",
    "Clogged pores visible?",
    "Skin looks oily?",
    "Redness visible?",
    "Skin discoloration visible?",
    "Pigmentation spots visible?",
    "Signs of irritation?",
    "Facial puffiness visible?",
    "Aging signs with breakouts?",
    "Uneven skin texture?",
    "Skin looks stressed?",
    "Multiple mild skin issues visible?",
    "Dry or flaky skin visible?",
]

# ------------------------------
# Load models at startup (with warm‑up)
# ------------------------------
def load_models() -> None:
    global caption_processor, caption_model, vqa_processor, vqa_model, MODEL_READY_AT

    # BLIP caption
    caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
    caption_model.to(DEVICE)

    # BLIP VQA
    vqa_processor = AutoProcessor.from_pretrained(VQA_MODEL_NAME)
    vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME)
    vqa_model.to(DEVICE)

    # Warm-up (1 tiny 32x32 fekete kép, hogy a first-call ne legyen hideg)
    img = Image.new("RGB", (32, 32), (0, 0, 0))

    with torch.inference_mode():
        # caption warmup
        cap_inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
        _ = caption_model.generate(**cap_inputs, max_new_tokens=5)

        # vqa warmup (triviális kérdés)
        vqa_inputs = vqa_processor(images=img, text="Is anything visible?", return_tensors="pt").to(DEVICE)
        _ = vqa_model.generate(**vqa_inputs, max_new_tokens=3)

    MODEL_READY_AT = time.time()

load_models()

# ------------------------------
# Helpers
# ------------------------------
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

def yes_no_from_text(txt: str) -> str:
    """
    BLIP VQA gyakran rövid választ ad. Normalizáljuk yes/no/unsure értékre.
    """
    t = (txt or "").strip().lower()
    if any(k in t for k in ["yes", "yeah", "yep", "true"]):
        return "yes"
    if any(k in t for k in ["no", "nope", "false"]):
        return "no"
    # gyakori egy szavas alternatívák
    if t in {"oily", "acne", "redness"}:
        return "yes"
    return "unsure"

def leds_from_flags(flags: Dict[str, str]) -> List[str]:
    """
    Egyszerű backend‑szabályok az ajánláshoz (később bővíthető).
    """
    Y = lambda k: flags.get(k, "no") == "yes"

    leds: List[str] = []
    # Blue
    if Y("Acne or pimples visible?") or Y("Clogged pores visible?") or Y("Skin looks oily?"):
        leds.append("Blue 465 nm")

    # Red
    if Y("Fine lines visible?") or Y("Skin looks tired?") or Y("Uneven skin texture?"):
        leds.append("Red 630 nm")

    # Red+NIR
    if Y("Deep wrinkles visible?") or Y("Skin looks saggy?"):
        leds.append("Red + Near‑IR 630+850 nm")

    # Green
    if Y("Redness visible?") or Y("Skin discoloration visible?") or Y("Pigmentation spots visible?"):
        leds.append("Green 530 nm")

    # Yellow
    if Y("Skin looks tired?") or Y("Facial puffiness visible?") or Y("Signs of irritation?"):
        leds.append("Yellow 590 nm")

    # Purple (aging + breakouts/texture)
    if Y("Aging signs with breakouts?") or (Y("Fine lines visible?") and (Y("Acne or pimples visible?") or Y("Uneven skin texture?"))):
        leds.append("Purple 630+465 nm")

    # Indigo (vibrancy + calming)
    if Y("Skin looks stressed?") or Y("Multiple mild skin issues visible?"):
        leds.append("Indigo 465+530 nm")

    # White (mindenből kicsi)
    if not leds or Y("Multiple mild skin issues visible?"):
        leds.append("White 630+530+465 nm")

    # dedup, sorrend megtartásával
    seen = set()
    ordered = []
    for m in leds:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered

# ------------------------------
# Endpoints
# ------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": all([caption_model, caption_processor, vqa_model, vqa_processor]) is not None,
        "model_ready_at": MODEL_READY_AT,
        "ts": time.time(),
    }

@app.post("/caption", response_model=CaptionResponse)
async def caption(body: CaptionRequest):
    if caption_model is None or caption_processor is None:
        raise HTTPException(status_code=503, detail="Caption model not loaded")

    img = await fetch_image(str(body.image))
    inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        ids = caption_model.generate(
            **inputs,
            max_new_tokens=35,
            num_beams=3,
            repetition_penalty=1.2,
        )
        txt = caption_processor.decode(ids[0], skip_special_tokens=True).strip()

    return CaptionResponse(id=body.id, caption=txt, model_ready_at=MODEL_READY_AT)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest):
    if vqa_model is None or vqa_processor is None:
        raise HTTPException(status_code=503, detail="VQA model not loaded")

    questions = body.questions or DEFAULT_QUESTIONS
    img = await fetch_image(str(body.image))

    answers: Dict[str, str] = {}
    with torch.inference_mode():
        for q in questions:
            # BLIP VQA formátum: "Question: ... Answer:"
            prompt = f"Question: {q} Answer:"
            inputs = vqa_processor(images=img, text=prompt, return_tensors="pt").to(DEVICE)
            out_ids = vqa_model.generate(
                **inputs,
                max_new_tokens=5,
                num_beams=3,
                length_penalty=0.0,
            )
            raw = vqa_processor.decode(out_ids[0], skip_special_tokens=True)
            answers[q] = yes_no_from_text(raw)

    led_modes = leds_from_flags(answers)

    # Rövid indoklás
    positives = [k for k, v in answers.items() if v == "yes"]
    rationale = (
        "Detected concerns: " + (", ".join(positives) if positives else "none with high confidence") +
        ". Recommended LED modes: " + ", ".join(led_modes) + "."
    )

    return AnalyzeResponse(
        id=body.id,
        answers=answers,
        leds=led_modes,
        rationale=rationale,
        model_ready_at=MODEL_READY_AT,
    )

# ------------------------------
# Local run
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

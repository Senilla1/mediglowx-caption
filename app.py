# app.py
import os
import time
from io import BytesIO
from typing import List, Optional, Dict, Any

import logging
import httpx
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# Transformers
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    BlipForQuestionAnswering,  # alias of BLIP for VQA ckpt
)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_START_TS = time.time()

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------
app = FastAPI(title="MediGlowX BLIP API", version="1.0.0")

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
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
    questions: Optional[List[str]] = None  # ha None, a default kérdéseket használjuk


class AnalyzeResponse(BaseModel):
    id: str
    answers: Dict[str, str]        # "Fine lines visible?" -> "yes/no/unsure"
    led_modes: List[str]           # ajánlott LED módok
    rationale: str                 # rövid összefoglaló
    model_ready_at: float

# Opcionális “final result” fogadó endpoint (nálatok már megvolt)
from typing import Optional as _Optional, List as _List  # csak hogy pydantic ne kavarjon

class ReceiveRequest(BaseModel):
    analysis_id: _Optional[str] = None
    user_id: _Optional[str] = None
    final_text: _Optional[str] = None
    image_url: _Optional[HttpUrl] = None
    questions: _Optional[_List[str]] = None

class ReceiveResponse(BaseModel):
    ok: bool
    payload: Dict[str, Any]

# ------------------------------------------------------------------------------
# Model config
# ------------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAPTION_MODEL_NAME = os.environ.get("BLIP_CAPTION_MODEL", "salesforce/blip-image-captioning-large")
VQA_MODEL_NAME     = os.environ.get("BLIP_VQA_MODEL",     "salesforce/blip-vqa-base")

caption_processor: Optional[BlipProcessor] = None
caption_model:     Optional[BlipForConditionalGeneration] = None

vqa_processor: Optional[AutoProcessor] = None
vqa_model:     Optional[BlipForQuestionAnswering] = None

MODEL_READY_AT: float = 0.0

# ------------------------------------------------------------------------------
# Default  kérdések  (a ti listátok alapján)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Load models at startup (with warm-up)
# ------------------------------------------------------------------------------
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

    # ~1 tiny 32x32 fekete kép warmuphoz
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

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)

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
    Egyszerű backend-szabályok az ajánláshoz – sorrendtartással és deduplikálással.
    A flags kulcsai a kérdések, értékei "yes/no/unsure".
    """
    Y = lambda k: (flags.get(k, "no") == "yes")
    leds: List[str] = []

    # Blue
    if Y("Acne or pimples visible?") or Y("Clogged pores visible?") or Y("Skin looks oily?"):
        leds.append("Blue 465 nm")

    # Red
    if Y("Fine lines visible?") or Y("Skin looks tired?") or Y("Uneven skin texture?"):
        leds.append("Red 630 nm")

    # Red+NIR
    if Y("Deep wrinkles visible?") or Y("Skin looks saggy?"):
        leds.append("Red + Near-IR 630+850 nm")

    # Green
    if Y("Redness visible?") or Y("Skin discoloration visible?") or Y("Pigmentation spots visible?"):
        leds.append("Green 530 nm")

    # Yellow
    if Y("Skin looks tired?") or Y("Facial puffiness visible?") or Y("Signs of irritation?"):
        leds.append("Yellow 590 nm")

    # Purple (aging + breakouts/texture)
    if Y("Aging signs with breakouts?") and (Y("Fine lines visible?") or Y("Acne or pimples visible?") or Y("Uneven skin texture?")):
        leds.append("Purple 630+405 nm")

    # Indigo (vibrancy + calming)
    if Y("Skin looks stressed?") or Y("Multiple mild skin issues visible?"):
        leds.append("Indigo 465+530 nm")

    # White (mindenből kicsi)
    if Y("Multiple mild skin issues visible?") or Y("Dry or flaky skin visible?"):
        leds.append("White 630+530+465 nm")

    # dedup, sorrend megtartásával
    seen = set()
    ordered: List[str] = []
    for m in leds:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": all(x is not None for x in [caption_model, caption_processor, vqa_model, vqa_processor]),
        "model_ready_at": MODEL_READY_AT,
        "uptime_s": time.time() - APP_START_TS,
    }

@app.post("/caption", response_model=CaptionResponse)
async def caption(body: CaptionRequest):
    if caption_model is None or caption_processor is None:
        raise HTTPException(status_code=503, detail="Caption model not loaded")

    img = await fetch_image(str(body.image))
    with torch.inference_mode():
        inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
        out_ids = caption_model.generate(
            **inputs,
            max_new_tokens=35,
            num_beams=3,
            repetition_penalty=1.2,
        )
        txt = caption_processor.decode(out_ids[0], skip_special_tokens=True).strip()

    return CaptionResponse(id=body.id, caption=txt, model_ready_at=MODEL_READY_AT)

# ------------------------  MÉRŐPONTOS /analyze  -------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest):
    logger.info("analyze:start req_id=%s", body.id)

    if vqa_model is None or vqa_processor is None:
        raise HTTPException(status_code=503, detail="VQA model not loaded")

    try:
        # 1) kép letöltés
        t0 = time.time()
        img = await fetch_image(str(body.image))
        t1 = time.time()

        # 2) kérdés-válasz inferencia
        answers: Dict[str, str] = {}
        qs = body.questions or DEFAULT_QUESTIONS

        with torch.inference_mode():
            for q in qs:
                prompt = f"Question: {q} Answer:"
                inputs = vqa_processor(images=img, text=prompt, return_tensors="pt").to(DEVICE)
                out = vqa_model.generate(
                    **inputs,
                    max_new_tokens=5,
                    num_beams=3,
                    length_penalty=0.0,
                )
                raw = vqa_processor.decode(out[0], skip_special_tokens=True)
                answers[q] = yes_no_from_text(raw)

        t2 = time.time()

        # 3) posztprocessz
        positives = [k for k, v in answers.items() if v == "yes"]
        rationale = ", ".join(positives) if positives else "none with high confidence"
        led_modes = leds_from_flags(answers)

        t3 = time.time()

        logger.info(
            "analyze:timings req_id=%s download=%.3f infer=%.3f post=%.3f total=%.3f flags=%s",
            body.id, (t1 - t0), (t2 - t1), (t3 - t2), (t3 - t0), answers,
        )

        return AnalyzeResponse(
            id=body.id,
            answers=answers,
            led_modes=led_modes,
            rationale=rationale,
            model_ready_at=MODEL_READY_AT,
        )

    except Exception as e:
        logger.exception("analyze:error req_id=%s err=%s", body.id, e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------  “Final result” fogadó -------------------------------

@app.post("/receive", response_model=ReceiveResponse)
async def receive(req: ReceiveRequest):
    payload = {
        "analysis_id": req.analysis_id,
        "user_id": req.user_id,
        "final_text": req.final_text,
        "image_url": str(req.image_url) if req.image_url else None,
        "questions": req.questions or [],
    }
    # TODO: itt lehet DB-be menteni, e-mailt küldeni stb.
    return ReceiveResponse(ok=True, payload=payload)

# ------------------------  Local run  -----------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
# --- CORS engedély a Typedream és a saját domainek felé ---
try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://skincheck.mediglowx.com",   # Typedream aldomain
            "https://mediglowx.com",             # fődomain
            "https://www.mediglowx.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# --- Eredmények ideiglenes tárhelye (MVP) ---
RESULTS: Dict[str, Dict[str, Any]] = {}

# A Make IDE KÜLD: itt tároljuk el az eredményt
class ReceiveRequest(BaseModel):
    analysis_id: str       # = Tally responseId (rid)
    user_email: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    final_text: Optional[str] = None
    leds: Optional[List[str]] = None
    rationale: Optional[str] = None
    questions: Optional[List[str]] = None

class ReceiveResponse(BaseModel):
    ok: bool
    payload: Dict[str, Any] = {}

@app.post("/receive", response_model=ReceiveResponse)
async def receive(body: ReceiveRequest):
    RESULTS[body.analysis_id] = {
        "rid": body.analysis_id,
        "email": body.user_email,
        "image_url": str(body.image_url) if body.image_url else None,
        "final_text": body.final_text,
        "leds": body.leds or [],
        "rationale": body.rationale,
        "questions": body.questions or [],
        "ts": time.time(),
    }
    return ReceiveResponse(ok=True, payload=RESULTS[body.analysis_id])

# A TYPEDREAM LEKÉR: ezzel húzza be az oldal a tartalmat
@app.get("/result", response_model=Dict[str, Any])
async def get_result(rid: str):
    data = RESULTS.get(rid)
    return {"ok": bool(data), "rid": rid, "data": data}


# --- FONTOS: Lokális futtatás ---
if __name__ == "__main__":
    app.run(debug=True)
    @app.get("/healthz")
def healthz():
    return {"ok": True}

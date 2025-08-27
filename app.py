# app.py — MediGlowX BLIP API (Render proxy + RunPod serverless támogatás)
import os
import asyncio
import time
from io import BytesIO
from typing import List, Optional, Dict, Any

import logging
import httpx
import torch
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
# Transformers
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    BlipForQuestionAnswering,  # alias of BLIP for VQA ckpt
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
APP_START_TS = time.time()

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="MediGlowX BLIP API", version="1.0.0")

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
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
    answers: Dict[str, str]
    led_modes: List[Dict[str, Any]] = Field(default_factory=list)    rationale: str = ""
    model_ready_at: float = 0.0
    # ÚJ mezők:
    metrics: Dict[str, float] = Field(default_factory=dict)
    auto_flags: Dict[str, bool] = Field(default_factory=dict)


# Opcionális "final result" fogadó endpointhoz (nálatok már megvolt)
class ReceiveRequest(BaseModel):
    analysis_id: Optional[str] = None      # Tally responseID (rid)
    user_email: Optional[str] = None
    final_text: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    leds: Optional[List[str]] = None
    questions: Optional[List[str]] = None


class ReceiveResponse(BaseModel):
    ok: bool
    payload: Dict[str, Any]


# -----------------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-large"
VQA_MODEL_NAME     = "Salesforce/blip-vqa-base"

caption_processor: Optional[BlipProcessor] = None
caption_model: Optional[BlipForConditionalGeneration] = None

vqa_processor: Optional[AutoProcessor] = None
vqa_model: Optional[BlipForQuestionAnswering] = None

MODEL_READY_AT: float = 0.0

# -----------------------------------------------------------------------------
# Default kérdések (a ti listátok alapján)
# -----------------------------------------------------------------------------
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
    "Skin discoloration visible?",
    "Pigmentation spots visible?",
    "Aging signs with breakouts?",
    "Cheeks skin textured?",
    "Skin looks stressed?",
    "Multiple mild skin issues visible?",
    "Dry or flaky skin visible?",
]
# ===== Dark-circle (under-eye) mérő: MediaPipe Face Mesh + LAB összehasonlítás =====
_mp_face = mp.solutions.face_mesh

def _to_uint8(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _roi(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, int(x1))); x2 = max(0, min(w-1, int(x2)))
    y1 = max(0, min(h-1, int(y1))); y2 = max(0, min(h-1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def detect_dark_circles(image_rgb) -> dict:
    """
    Visszaad: {'dark_circles': bool, 'dark_circles_score': float, 'debug': {...}}
    Heuriszta: under-eye régió L* (LAB) sötétebb-e a referencia (cheek) régiónál,
    és mennyivel kékesebb (b* kisebb).
    """
    img = _to_uint8(image_rgb)
    h, w = img.shape[:2]

    with _mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not res.multi_face_landmarks:
            return {'dark_circles': False, 'dark_circles_score': 0.0}

        lm = res.multi_face_landmarks[0].landmark
        def P(i): return np.array([lm[i].x*w, lm[i].y*h])

        # Landmark indexek (MediaPipe 468-pontos mesh)
        L_OUT, L_IN, L_BOTTOM = 33, 133, 145
        R_OUT, R_IN, R_BOTTOM = 263, 362, 374

        pL_out, pL_in, pL_b = P(L_OUT), P(L_IN), P(L_BOTTOM)
        pR_out, pR_in, pR_b = P(R_OUT), P(R_IN), P(R_BOTTOM)

        # Inter-ocular distance mint skála
        iod = np.linalg.norm(pL_in - pR_in) + 1e-6

        # Under-eye és cheek téglalapok
        def under_eye_rect(p_out, p_in, p_bot):
            x1 = min(p_out[0], p_in[0]) - 0.05*iod
            x2 = max(p_out[0], p_in[0]) + 0.05*iod
            y1 = p_bot[1] + 0.12*iod
            y2 = p_bot[1] + 0.45*iod
            return int(x1), int(y1), int(x2), int(y2)

        def cheek_rect(p_out, p_in, p_bot):
            x1 = min(p_out[0], p_in[0]) - 0.05*iod
            x2 = max(p_out[0], p_in[0]) + 0.05*iod
            y1 = p_bot[1] + 0.55*iod
            y2 = p_bot[1] + 0.95*iod
            return int(x1), int(y1), int(x2), int(y2)

        rects = [
            under_eye_rect(pL_out, pL_in, pL_b),
            under_eye_rect(pR_out, pR_in, pR_b),
        ]
        cheeks = [
            cheek_rect(pL_out, pL_in, pL_b),
            cheek_rect(pR_out, pR_in, pR_b),
        ]

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        Lc, ac, bc = cv2.split(lab)

        def mean_lab(r):
            if r is None: return None
            x1,y1,x2,y2 = r
            roiL = _roi(Lc, x1,y1,x2,y2); roiB = _roi(bc, x1,y1,x2,y2)
            if roiL is None or roiB is None: return None
            return float(np.mean(roiL)), float(np.mean(roiB))

        vals_under = [mean_lab(r) for r in rects]
        vals_cheek = [mean_lab(r) for r in cheeks]
        vals_under = [v for v in vals_under if v is not None]
        vals_cheek = [v for v in vals_cheek if v is not None]
        if not vals_under or not vals_cheek:
            return {'dark_circles': False, 'dark_circles_score': 0.0}

        L_under = np.mean([v[0] for v in vals_under])
        B_under = np.mean([v[1] for v in vals_under])
        L_cheek = np.mean([v[0] for v in vals_cheek])
        B_cheek = np.mean([v[1] for v in vals_cheek])

        dL = L_cheek - L_under      # + ha under-eye sötétebb
        dB = B_cheek - B_under      # + ha under-eye "kékebb"

        # Normalizált pontszám [0..1]
        l_drop = max(0.0, dL / 40.0)
        b_drop = max(0.0, dB / 20.0)
        score  = float(np.clip(0.7*l_drop + 0.3*b_drop, 0.0, 1.0))

        detected = (dL > 8 and score > 0.12)  # küszöbök finomíthatók

        return {
            'dark_circles': bool(detected),
            'dark_circles_score': round(score, 3),
            'debug': {'dL': round(dL,2), 'dB': round(dB,2)}
        }
# -----------------------------------------------------------------------------
# Load models at startup (with warm-up)
# -----------------------------------------------------------------------------
def load_models() -> None:
    global caption_processor, caption_model, vqa_processor, vqa_model, MODEL_READY_AT

    logger.info("CAPTION_MODEL_NAME = %s", CAPTION_MODEL_NAME)
    logger.info("VQA_MODEL_NAME = %s", VQA_MODEL_NAME)
    logger.info("Loading BLIP models.. device=%s", DEVICE)

    # BLIP caption
    from transformers import AutoProcessor
    caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_NAME)    
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME).to(DEVICE)

    # BLIP VQA
    vqa_processor = AutoProcessor.from_pretrained(VQA_MODEL_NAME)
    vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME).to(DEVICE)

    # Mini warm-up (32x32 fekete kép)
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    with torch.inference_mode():
        cap_inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
        _ = caption_model.generate(**cap_inputs, max_new_tokens=5)

        vqa_inputs = vqa_processor(images=img, text="Is anything visible?", return_tensors="pt").to(DEVICE)
        _ = vqa_model.generate(**vqa_inputs, max_new_tokens=3)

    MODEL_READY_AT = time.time()
    logger.info("Models ready at %.3f", MODEL_READY_AT)

# csak akkor töltsön modellt, ha NEM RunPod proxy
if os.getenv("USE_RUNPOD", "0") != "1":
    load_models()
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)

async def fetch_image(url: str) -> Image.Image:
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {r.status_code}")
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image file: {e}")

def yes_no_from_text(txt: str) -> str:
    t = (txt or "").strip().lower()
    if t in ("y", "yes", "true", "yep", "yeah"): 
        return "yes"
    if t in ("n", "no", "nope", "false"):
        return "no"
    # gyakori egy szavas alternatívák
    if t in ("oily", "acne", "redness"):
        return "yes"
    return "unsure"

def leds_from_flags(flags: Dict[str, str]) -> List[str]:
    """Egyszerű szabályok LED ajánláshoz – sorrendtartással és deduplikálással."""
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
        leds.append("Near-IR 638-850+ nm")

    # Green
    if Y("Pigmentation spots visible?") or Y("Skin discoloration visible?"):
        leds.append("Green 538 nm")

    # Yellow
    if Y("Aging signs with breakouts?") or Y("Cheeks skin textured?") or Y("Skin looks oily?"):
        leds.append("Yellow 590 nm")

    # Purple (aging + breakouts/texture)
    if (Y("Aging signs with breakouts?") and Y("Cheeks skin textured?")) or Y("Acne or pimples visible?"):
        leds.append("Purple 638+405 nm")

    # Indigo (barrier + calming)
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

# -----------------------------------------------------------------------------
# RunPod proxy támogatás (kapcsolóval)
# -----------------------------------------------------------------------------
# --- RunPod proxy (Queue API) ------------------------------------------------
import asyncio  # ha még nincs felül importálva

USE_RUNPOD = os.getenv("USE_RUNPOD", "0")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")  # pl. wy5mho4k9gqorm
RUNPOD_TOKEN = os.getenv("RUNPOD_TOKEN", "")
RUNPOD_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

async def call_runpod_queue(mode: str, image_url: str, questions: Optional[List[str]] = None) -> Dict[str, Any]:
    if not RUNPOD_ENDPOINT_ID:
        raise HTTPException(status_code=503, detail="RUNPOD_ENDPOINT_ID not configured")

    headers = {"Authorization": f"Bearer {RUNPOD_TOKEN}"}
    payload: Dict[str, Any] = {"input": {"mode": mode, "id": "proxy", "image": image_url}}
    if questions:
        payload["input"]["questions"] = questions

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        r = await client.post(f"{RUNPOD_BASE}/run", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"RunPod run error: {r.text}")
        rid = r.json()["id"]

        while True:
            s = await client.get(f"{RUNPOD_BASE}/status/{rid}", headers=headers)
            if s.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"RunPod status error: {s.text}")
            data = s.json()
            st = data.get("status", "")
            if st == "COMPLETED":
                return data.get("output", {})
            if st in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                raise HTTPException(status_code=502, detail=f"RunPod job {st.lower()}")
            await asyncio.sleep(1.0)

# --- minimal health endpoints for RunPod LB ---
@app.get("/")
def root():
    return {"ok": True}

@app.get("/ping")
def ping():
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True}
# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": all(x is not None for x in [caption_model, caption_processor, vqa_model, vqa_processor]),
        "model_ready_at": MODEL_READY_AT,
        "uptime_s": time.time() - APP_START_TS,
    }

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True}

@app.post("/caption", response_model=CaptionResponse)
async def caption(body: CaptionRequest):
    # PROXY mód Renderen: küldd a RunPod Queue-ra
    if USE_RUNPOD == "1":
        out = await call_runpod_queue("caption", str(body.image))
        return CaptionResponse(
            id=body.id,
            caption=str(out.get("caption", "")),
            model_ready_at=float(out.get("model_ready_at", 0.0)),
        )

    # LOKÁLIS inferencia (RunPodon USE_RUNPOD=0)
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

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest):
    # PROXY mód Renderen → RunPod Queue
    if USE_RUNPOD == "1":
        out = await call_runpod_queue(
            "analyze",
            str(body.image),
            body.questions or DEFAULT_QUESTIONS
        )
        return AnalyzeResponse(
    id=body.id,
    answers=answers,
    led_modes=led_modes,
    rationale=str(out.get("rationale", "")),
    model_ready_at=float(out.get("model_ready_at", 0.0)),
    metrics=metrics,
    auto_flags=auto_flags,
        )

    # ── LOKÁLIS VQA + postprocess ──
    if vqa_model is None or vqa_processor is None:
        raise HTTPException(status_code=503, detail="VQA model not loaded")

    t0 = time.time()
    img = await fetch_image(str(body.image))
    t1 = time.time()

    qs: List[str] = body.questions or DEFAULT_QUESTIONS
    answers: Dict[str, str] = {}
    

    with torch.inference_mode():
        for q in qs:
            vqa_inputs = vqa_processor(
                images=img,
                text=f"Question: {q} Answer:",
                return_tensors="pt"
            ).to(DEVICE)
            out = vqa_model.generate(
                **vqa_inputs,
                max_new_tokens=5,
                num_beams=3,
                length_penalty=0.0,
            )
            raw = vqa_processor.decode(out[0], skip_special_tokens=True)
            answers[q] = yes_no_from_text(raw)

    positives = [k for k, v in answers.items() if v == "yes"]
    rationale = ", ".join(positives) if positives else "none with high confidence"
    led_modes = leds_from_flags(answers)

    t2 = time.time()
    logger.info(
        "analyze: timings download=%.3f infer=%.3f post=%.3f total=%.3f flags=%s",
        t1 - t0, t2 - t1, time.time() - t2, time.time() - t0, answers
    )

    return AnalyzeResponse(
        id=body.id,
        answers=answers,
        led_modes=led_modes,
        rationale=rationale,
        model_ready_at=MODEL_READY_AT,
    )

    # --- LOKÁLIS (Render gépen futó) BLIP VQA + postprocess — eredeti logika ---
    if vqa_model is None or vqa_processor is None:
        raise HTTPException(status_code=503, detail="VQA model not loaded")

    t0 = time.time()
    # 1) Kép letöltés
    img = await fetch_image(str(body.image))
    t1 = time.time()
    try:
        image_rgb = np.array(img.convert('RGB'))
        dc = detect_dark_circles(image_rgb)
        metrics    = {'dark_circles_score': float(dc.get('dark_circles_score', 0.0))}
        auto_flags = {'dark_circles': bool(dc.get('dark_circles', False))}
    except Exception as e:
        logger.exception("dark circle calc error: %s", e)
        metrics, auto_flags = {}, {}

    logger.info("dc_measure id=%s auto_flags=%s metrics=%s", body.id, auto_flags, metrics)

    # 2) Kérdés–válasz inferencia
    qs: List[str] = body.questions or DEFAULT_QUESTIONS
    answers: Dict[str, str] = {}
    with torch.inference_mode():
        for q in qs:
            inputs = vqa_processor(images=img, text=f"Question: {q} Answer:", return_tensors="pt").to(DEVICE)
            out = vqa_model.generate(
                **inputs,
                max_new_tokens=5,
                num_beams=3,
                length_penalty=0.0,
            )
            raw = vqa_processor.decode(out[0], skip_special_tokens=True)
            answers[q] = yes_no_from_text(raw)

    t2 = time.time()

    # 3) postprocess
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

# -----------------------------------------------------------------------------
# --- Eredmények ideiglenes tárhelye (HWP) ---
# -----------------------------------------------------------------------------
RESULTS: Dict[str, Dict[str, Any]] = {}

@app.post("/receive", response_model=ReceiveResponse)
async def receive(body: ReceiveRequest):
    RESULTS[body.analysis_id or ""] = {
        "rid": body.analysis_id,
        "email": body.user_email,
        "image_url": str(body.image_url) if body.image_url else None,
        "final_text": body.final_text,
        "leds": body.leds or [],
        "rationale": body.rationale,
        "questions": body.questions or [],
        "ts": time.time(),
    }
    return ReceiveResponse(ok=True, payload=RESULTS[body.analysis_id or ""])

# A TYPEDREAM LEKÉR: ezzel húzza be az oldal a tartalmat
@app.get("/result", response_model=Dict[str, Any])
async def get_result(rid: str):
    data = RESULTS.get(rid, {})
    return {"ok": bool(data), "rid": rid, "data": data}

# -----------------------------------------------------------------------------
# CORS a Typedream és a saját domainekre
# -----------------------------------------------------------------------------
try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://skincheck.mediglowx.com",  # Typedream aldomain
            "https://mediglowx.com",            # fődomain
            "https://www.mediglowx.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# -----------------------------------------------------------------------------
# Lokális futtatás (dev)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

# worker.py  — RunPod Queue worker MediglowX BLIP-hez
import os, time
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
import httpx

from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

import runpod  # pip: runpod

# ---------------- Config + cache (a Dockerfile már /runpod-volume-ra irányít)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_MODEL_NAME = os.getenv("BLIP_CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
VQA_MODEL_NAME     = os.getenv("BLIP_VQA_MODEL", "Salesforce/blip-vqa-base")
HTTP_TIMEOUT       = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)

DEFAULT_QUESTIONS: List[str] = [
    "Fine lines visible?", "Skin looks tired?", "Uneven skin texture?",
    "Acne or pimples visible?", "Clogged pores visible?", "Skin looks oily?",
    "Skin discoloration visible?", "Pigmentation spots visible?",
    "Aging signs with breakouts?", "Cheeks skin textured?",
    "Skin looks stressed?", "Multiple mild skin issues visible?",
    "Dry or flaky skin visible?",
]

# ---------------- Load models (egyszer, induláskor)
print(f"[worker] loading models on {DEVICE}…")
caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_NAME)
caption_model     = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME).to(DEVICE)

vqa_processor     = AutoProcessor.from_pretrained(VQA_MODEL_NAME)
vqa_model         = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME).to(DEVICE)

MODEL_READY_AT = time.time()
print(f"[worker] models ready at {MODEL_READY_AT}")

# ---------------- Helpers
async def fetch_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url)
    if r.status_code >= 400:
        raise ValueError(f"Failed to download image: {r.status_code}")
    return Image.open(r.content if hasattr(r, "content") else r).convert("RGB")

def yes_no_from_text(t: str) -> str:
    t = (t or "").strip().lower()
    if t in {"yes", "true", "yup", "y", "yeah"}: return "yes"
    if t in {"no", "false", "nope", "n"}: return "no"
    return "unsure"

def leds_from_flags(flags: Dict[str, str]) -> List[str]:
    """Egyszerű LED-javaslat logika a korábbi szabályaid alapján."""
    leds: List[str] = []
    Y = lambda k: flags.get(k, "unsure") == "yes"

    if Y("Acne or pimples visible?") or Y("Clogged pores visible?") or Y("Skin looks oily?"):
        leds.append("Blue 465 nm")

    if Y("Fine lines visible?") or Y("Skin looks saggy?"):
        leds.append("Red 633 nm")

    if Y("Pigmentation spots visible?") or Y("Skin discoloration visible?"):
        leds.append("Green 538 nm")

    if Y("Aging signs with breakouts?") or Y("Cheeks skin textured?") or Y("Skin looks oily?"):
        leds.append("Yellow 590 nm")

    if (Y("Aging signs with breakouts?") and Y("Cheeks skin textured?")) or Y("Acne or pimples visible?"):
        leds.append("Purple 638+405 nm")

    if Y("Multiple mild skin issues visible?") or Y("Dry or flaky skin visible?"):
        leds.append("White 380–780 nm")

    # dedup, sorrend tartása
    seen, out = set(), []
    for l in leds:
        if l not in seen:
            seen.add(l); out.append(l)
    return out

# ---------------- Inference
def do_caption(img: Image.Image) -> str:
    inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
    out_ids = caption_model.generate(
        **inputs, max_new_tokens=32, num_beams=3, repetition_penalty=1.2
    )
    return caption_processor.decode(out_ids[0], skip_special_tokens=True).strip()

def do_analyze(img: Image.Image, questions: Optional[List[str]]) -> Dict[str, Any]:
    qs = questions or DEFAULT_QUESTIONS
    answers: Dict[str, str] = {}
    for q in qs:
        inputs = vqa_processor(images=img, text=f"Question: {q} Answer:", return_tensors="pt").to(DEVICE)
        out_ids = vqa_model.generate(**inputs, max_new_tokens=5, num_beams=3)
        raw = vqa_processor.decode(out_ids[0], skip_special_tokens=True)
        answers[q] = yes_no_from_text(raw)
    positives = [k for k, v in answers.items() if v == "yes"]
    rationale = ", ".join(positives) if positives else "none with high confidence"
    led_modes = leds_from_flags(answers)
    return {"answers": answers, "led_modes": led_modes, "rationale": rationale}

# ---------------- RunPod handler
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    event: {"input": {...}} a RunPod Queue-tól
    """
    try:
        data = event.get("input", {})
        mode = (data.get("mode") or "caption").lower()
        img_url = data.get("image")
        if not img_url:
            return {"error": "Missing 'image' URL in input."}

        # Kép letöltés
        import asyncio
        img = asyncio.get_event_loop().run_until_complete(fetch_image(img_url))

        if mode == "analyze":
            res = do_analyze(img, data.get("questions"))
            out = {"id": data.get("id") or "", **res, "model_ready_at": MODEL_READY_AT}
        else:
            caption = do_caption(img)
            out = {"id": data.get("id") or "", "caption": caption, "model_ready_at": MODEL_READY_AT}

        return out
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})

# worker.py — RunPod Queue handler (SYNC, nincs asyncio)
import os, io, time, urllib.request
from typing import Dict, List, Any, Optional
from PIL import Image

import torch
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import runpod

# ---- Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_MODEL_NAME = os.getenv("BLIP_CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
VQA_MODEL_NAME     = os.getenv("BLIP_VQA_MODEL",     "Salesforce/blip-vqa-base")

# Hugging Face cache helye (RunPod Network Volume-hoz igazítva)
HF_HOME = os.getenv("HF_HOME", "/root/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", HF_HOME)

MODEL_READY_AT = 0.0

# ---- Default kérdések (ugyanaz a lista, mint az app.py-ben)
DEFAULT_QUESTIONS: List[str] = [
    "Fine lines visible?", "Skin looks dull?", "Deep wrinkles visible?",
    "Skin looks saggy?", "Skin looks tired?", "Acne or pimples visible?",
    "Clogged pores visible?", "Skin looks oily?", "Skin discoloration visible?",
    "Pigmentation spots visible?", "Aging signs with breakouts?",
    "Cheeks skin textured?", "Skin looks stressed?",
    "Multiple mild skin issues visible?", "Dry or flaky skin visible?",
]

# ---- Helper: kép letöltés (szinkron)
def fetch_image_sync(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=30) as r:
        if r.status >= 400:
            raise RuntimeError(f"Failed to download image: {r.status}")
        return Image.open(io.BytesIO(r.read())).convert("RGB")

# ---- Szöveg → igen/nem
def yes_no_from_text(s: str) -> str:
    t = (s or "").strip().lower()
    if t in {"yes", "true", "y", "yep", "yeah"}:
        return "yes"
    if t in {"no", "false", "n", "nope"}:
        return "no"
    return "unsure"

# ---- LED javaslat szabályok (mint az app.py-ben)
def leds_from_flags(flags: Dict[str, str]) -> List[str]:
    Y = lambda k: flags.get(k, "") == "yes"
    leds: List[str] = []

    # Blue
    if Y("Acne or pimples visible?") or Y("Clogged pores visible?") or Y("Skin looks oily?"):
        leds.append("Blue 465 nm")

    # Red+NIR
    if Y("Deep wrinkles visible?") or Y("Skin looks saggy?"):
        leds.append("Near-IR 638–850+ nm")

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
    if Y("Skin looks stressed?") or Y("Multiple mild skin issues visible?") or Y("Dry or flaky skin visible?"):
        leds.append("Indigo 465+530 nm")

    # White (mindenből kicsi)
    if Y("Multiple mild skin issues visible?") or Y("Dry or flaky skin visible?"):
        leds.append("White 638+530+405 nm")

    # dedup, sorrend megtartás
    seen, ordered = set(), []
    for l in leds:
        if l not in seen:
            ordered.append(l); seen.add(l)
    return ordered

# ---- Modellek betöltése (induláskor)
caption_processor = None
caption_model = None
vqa_processor = None
vqa_model = None

def load_models():
    global caption_processor, caption_model, vqa_processor, vqa_model, MODEL_READY_AT

    caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_NAME)
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME).to(DEVICE)

    vqa_processor = AutoProcessor.from_pretrained(VQA_MODEL_NAME)
    vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME).to(DEVICE)

    # mini warmup
    with torch.inference_mode():
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        cap_inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
        _ = caption_model.generate(**cap_inputs, max_new_tokens=5)

        vqa_inputs = vqa_processor(images=img, text="Is anything visible?", return_tensors="pt").to(DEVICE)
        _ = vqa_model.generate(**vqa_inputs, max_new_tokens=3)

    MODEL_READY_AT = time.time()

load_models()

# ---- Függvények a feladatokra
def do_caption(image_url: str) -> Dict[str, Any]:
    img = fetch_image_sync(image_url)
    with torch.inference_mode():
        inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
        out_ids = caption_model.generate(
            **inputs, max_new_tokens=32, num_beams=3, repetition_penalty=1.2
        )
        text = caption_processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return {"caption": text, "model_ready_at": MODEL_READY_AT}

def do_analyze(image_url: str, questions: Optional[List[str]] = None) -> Dict[str, Any]:
    img = fetch_image_sync(image_url)
    questions = questions or DEFAULT_QUESTIONS
    answers: Dict[str, str] = {}

    t0 = time.time()
    with torch.inference_mode():
        for q in questions:
            inputs = vqa_processor(
                images=img, text=f"Question: {q} Answer:", return_tensors="pt"
            ).to(DEVICE)
            out = vqa_model.generate(**inputs, max_new_tokens=5, num_beams=3)
            raw = vqa_processor.batch_decode(out, skip_special_tokens=True)[0]
            answers[q] = yes_no_from_text(raw)

    positives = ", ".join([k for k, v in answers.items() if v == "yes"]) or "none with high confidence"
    led_modes = leds_from_flags(answers)

    return {
        "answers": answers,
        "led_modes": led_modes,
        "rationale": positives,
        "model_ready_at": MODEL_READY_AT,
        "elapsed_sec": round(time.time() - t0, 3),
    }

# ---- RunPod handler (SYNC!)
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Várja:
    {
      "input": {
        "mode": "caption" | "analyze",
        "id": "test-1",
        "image": "https://.../zidane.jpg",
        "questions": [...]   # opcionális az analyze-hoz
      }
    }
    """
    inp = event.get("input", {})
    mode = (inp.get("mode") or "caption").lower()
    image_url = inp.get("image")

    if not image_url:
        return {"ok": False, "error": "Missing 'image' in input."}

    if mode == "caption":
        out = do_caption(image_url)
        return {"ok": True, "id": inp.get("id"), **out}

    # analyze
    out = do_analyze(image_url, inp.get("questions"))
    return {"ok": True, "id": inp.get("id"), **out}

runpod.serverless.start({"handler": handler})

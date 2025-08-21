# worker.py
import time, os
from typing import Dict, Any, Optional, List
import runpod
from PIL import Image
import httpx

# Újrahasznosítjuk az app.py modelljeit/függvényeit
from app import (
    load_models, caption_processor, caption_model,
    vqa_processor, vqa_model, DEVICE,
    fetch_image, yes_no_from_text, leds_from_flags
)

# Cold start: modellek beolvasása
load_models()

async def do_caption(image_url: str, req_id: str) -> Dict[str, Any]:
    img: Image.Image = await fetch_image(image_url)
    inputs = caption_processor(images=img, return_tensors="pt").to(DEVICE)
    out_ids = caption_model.generate(**inputs, max_new_tokens=32, num_beams=3, repetition_penalty=1.2)
    text = caption_processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return {"id": req_id, "caption": text, "model_ready_at": time.time()}

async def do_analyze(image_url: str, questions: Optional[List[str]], req_id: str) -> Dict[str, Any]:
    img: Image.Image = await fetch_image(image_url)
    qs = questions or [
        "Fine lines visible?", "Skin looks oily?", "Uneven skin texture?",
        "Acne or pimples visible?", "Clogged pores visible?", "Deep wrinkles visible?",
        "Skin looks saggy?", "Skin looks tired?", "Skin discoloration visible?",
        "Pigmentation spots visible?", "Aging signs with breakouts?", "Cheeks skin textured?",
        "Skin looks stressed?", "Multiple mild skin issues visible?", "Dry or flaky skin visible?"
    ]
    answers = {}
    for q in qs:
        inputs = vqa_processor(images=img, text=f"Question: {q} Answer:", return_tensors="pt").to(DEVICE)
        out = vqa_model.generate(**inputs, max_new_tokens=5, num_beams=3, length_penalty=0.0)
        raw = vqa_processor.decode(out[0], skip_special_tokens=True)
        answers[q] = yes_no_from_text(raw)
    led_modes = leds_from_flags(answers)
    return {
        "id": req_id,
        "answers": answers,
        "led_modes": led_modes,
        "rationale": ", ".join([k for k, v in answers.items() if v == "yes"]) or "none with high confidence",
        "model_ready_at": time.time(),
    }

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Várt bemenet (Queue):
    {
      "input": {
        "task": "caption" | "analyze",
        "id": "test-1",
        "image": "https://...jpg",
        "questions": ["..."]   # csak analyze-hoz, opcionális
      }
    }
    """
    inp = event.get("input") or {}
    task = (inp.get("task") or "caption").lower()
    req_id = str(inp.get("id") or "")
    image = inp.get("image")
    questions = inp.get("questions")

    if not image:
        return {"error": "missing field: image"}

    import asyncio
    if task == "analyze":
        return asyncio.run(do_analyze(image, questions, req_id))
    else:
        return asyncio.run(do_caption(image, req_id))

runpod.serverless.start({"handler": handler})

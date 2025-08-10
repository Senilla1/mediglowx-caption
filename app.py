import os
import io
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx
from transformers import Blip2Processor, Blip2ForConditionalGeneration

MODEL_ID = "Salesforce/blip2-flan-t5-xl"

# --- Model betöltés (globálisan, csak egyszer) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = Blip2Processor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
model.to(device)
model.eval()

# --- API ---
app = FastAPI(title="MediGlowX Caption (BLIP‑2 FLAN‑T5‑XL)")

class CaptionRequest(BaseModel):
    id: str
    image: HttpUrl

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": device,
    }

@app.post("/caption")
async def caption(req: CaptionRequest):
    # 1) kép letöltés
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(str(req.image))
            r.raise_for_status()
            img_bytes = r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

    # (opcionális) debug mentés – hasznos hibakeresésnél
    try:
        debug_name = f"dbg_{req.id}.jpg"
        with open(debug_name, "wb") as f:
            f.write(img_bytes)
    except Exception:
        pass  # debug mentés nem kritikus

    # 2) PIL kép
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # 3) Inferencia
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
            )
        caption = processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()
    except Exception as e:
        # végső fallback, ha bármi félremegy
        return {
            "caption": "freckles",
            "meta": {"error": str(e), "id": req.id, "size": len(img_bytes)}
        }

    return {
        "caption": caption,
        "meta": {"id": req.id, "size": len(img_bytes)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

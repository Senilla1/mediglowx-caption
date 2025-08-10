from fastapi import FastAPI
from pydantic import BaseModel
import requests, io
from PIL import Image

app = FastAPI()

class CaptionReq(BaseModel):
    id: str
    image: str  # URL

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/caption")
def caption(req: CaptionReq):
    # 1) kép letöltése
    try:
        r = requests.get(req.image, timeout=30)
        status = r.status_code
        size = len(r.content)
    except Exception as e:
        return {"caption": "ERROR: download_failed", "reason": str(e)}

    if status != 200 or size == 0:
        return {"caption": "ERROR: bad_status_or_empty", "status": status, "size": size}

    # 2) sanity check a képre
    head = r.content[:16].hex()
    try:
        img = Image.open(io.BytesIO(r.content))
        w, h = img.size
        fmt = img.format
    except Exception as e:
        return {"caption": "ERROR: pillow_open_failed", "reason": str(e), "head": head}

    # 3) ide kerül majd a tényleges modellhívás — most DEBUG
    guess = "face_detected" if w * h > 0 else "unknown"

    return {
        "caption": f"DEBUG ok | id={req.id} | size={size} | guess={guess}",
        "meta": {"w": w, "h": h, "fmt": fmt, "head": head}
    }

# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import httpx
import logging
import os
from urllib.parse import urlparse, parse_qs
from PIL import Image
import io
import statistics

# -----------------------------------------------------------------------------
# Alap beállítások
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("mediglowx-caption")

app = FastAPI(title="MediGlowX Caption API", version="1.0.0")

WORKER_URL = os.getenv("WORKER_URL", "").strip()  # pl. https://mediglowx-worker.onrender.com/analyze

# -----------------------------------------------------------------------------
# Modellek
# -----------------------------------------------------------------------------
class CaptionRequest(BaseModel):
    id: str
    image: HttpUrl  # Cloudinary URL (vagy bármely publikus kép-URL)

# -----------------------------------------------------------------------------
# Segédfüggvények
# -----------------------------------------------------------------------------
async def fetch_image_bytes(url: str) -> Dict[str, Any]:
    """
    Letölti a képet. Visszaad: {content: bytes, content_type: str, size: int, from_cache: bool}
    A cache-bust ?v= paramétert naplózza.
    """
    parsed = urlparse(url)
    v_param = parse_qs(parsed.query).get("v", [""])[0]
    from_cache = False  # ezt itt csak naplózzuk; a ?v= jelenléte jelzi a cache-bustot

    headers = {
        "User-Agent": "MediGlowX/1.0 (+https://mediglowx-caption.onrender.com)"
    }
    timeout = httpx.Timeout(20.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        r = await client.get(url)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"[fetch_image_bytes] HTTP error: {e.response.status_code} for {url}")
            raise HTTPException(status_code=502, detail=f"Image download failed: {e.response.status_code}")

        content_type = r.headers.get("Content-Type", "")
        content = r.content
        size = len(content)

    logger.info(f"[fetch_image_bytes] downloaded size={size}B ct={content_type} v={v_param!r} url_host={parsed.netloc}")
    return {"content": content, "content_type": content_type, "size": size, "from_cache": from_cache}

def quick_caption_from_image(img_bytes: bytes) -> str:
    """
    Nagyon egyszerű, determinisztikus baseline 'caption' (amíg nincs worker).
    Nem orvosi, csak technikai leírás: méret, átlag fényesség, kontraszt.
    """
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("L")
        w, h = im.size
        pixels = list(im.getdata())
        mean_brightness = sum(pixels) / len(pixels)
        # egyszerű 'kontraszt' proxy: mintavételezett szórás
        if len(pixels) > 50000:
            sample = pixels[:: len(pixels)//50000]
        else:
            sample = pixels
        stdev = statistics.pstdev(sample) if len(sample) > 1 else 0.0

        tone = "balanced tone" if 95 <= mean_brightness <= 160 else ("bright" if mean_brightness > 160 else "dim")
        contrast = "low contrast" if stdev < 35 else ("moderate contrast" if stdev < 55 else "high contrast")

        return f"High-level photo summary: {w}×{h}px, {tone}, {contrast}."
    except Exception as e:
        logger.warning(f"[quick_caption_from_image] PIL parse failed: {e}")
        return "Image received and validated."

async def call_worker(image_url: str, req_id: str) -> Optional[str]:
    """
    Opcionális: ha van WORKER_URL, meghívjuk.
    Várjuk: JSON { "caption": "..." }
    """
    if not WORKER_URL:
        return None

    payload = {"id": req_id, "image": image_url}
    timeout = httpx.Timeout(60.0, connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(WORKER_URL, json=payload)
            r.raise_for_status()
            data = r.json()
            cap = data.get("caption")
            if isinstance(cap, str) and cap.strip():
                return cap
            logger.warning("[call_worker] Worker returned no caption string; falling back.")
            return None
    except Exception as e:
        logger.error(f"[call_worker] Worker call failed: {e}")
        return None

# -----------------------------------------------------------------------------
# Endpontok
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/caption")
async def caption_endpoint(req: CaptionRequest, raw: Request):
    logger.info(f"[/caption] id={req.id} image={req.image}")

    # 1) Kép letöltés + ellenőrzés
    img = await fetch_image_bytes(str(req.image))
    if not img["content_type"].startswith("image/") and not str(req.image).lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        # néha a CDN nem ad Content-Type‑ot; minimum a kiterjesztést nézzük
        logger.warning(f"Suspicious content-type: {img['content_type']!r}")
    if img["size"] < 1024:
        raise HTTPException(status_code=422, detail="Downloaded file is too small to be a valid image.")

    # 2) WORKER hívás (ha van), különben baseline caption
    cap = await call_worker(str(req.image), req.id)
    if not cap:
        cap = quick_caption_from_image(img["content"])

    # 3) Válasz – egységes séma
    resp = {
        "caption": cap,
        "fileSize": img["size"],
        "meta": {
            "id": req.id,
            "from_cache": False  # itt információs flag; a cache‑bustot a ?v= biztosítja
        }
    }
    logger.info(f"[/caption] OK id={req.id} size={img['size']}B caption_len={len(cap)}")
    return JSONResponse(resp)

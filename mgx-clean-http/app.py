from fastapi import FastAPI, Request, HTTPException, Body
from pydantic import BaseModel
import os

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

X_API_KEY = os.getenv("X_API_KEY", "")

@app.post("/caption")
async def caption(req: Request, payload: dict = Body(...)):
    # opcionális fejléces védelem – most nincs kulcs beállítva, így nem ellenőriz
    if X_API_KEY and req.headers.get("x-api-key") != X_API_KEY:
        raise HTTPException(status_code=401, detail="bad x-api-key")

    # Normalizálás: elfogadjuk a két formátumot
    image_url = None
    if isinstance(payload, dict):
        if "image_url" in payload:
            image_url = payload["image_url"]
        elif "input" in payload and isinstance(payload["input"], dict):
            image_url = payload["input"].get("image_url")

    if not image_url:
        raise HTTPException(
            status_code=422,
            detail="image_url hiányzik (elfogadott: {'image_url': ...} vagy {'input': {'image_url': ...}})"
        )

    # (itt jöhet majd a valódi feldolgozás)
    return {"received": image_url, "status": "OK"}

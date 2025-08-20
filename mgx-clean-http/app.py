from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

class In(BaseModel):
    image_url: str

@app.get("/health")
def health():
    return {"ok": True}

# NINCS kötelező auth most
X_API_KEY = os.getenv("X_API_KEY", "")

@app.post("/caption")
def caption(req: Request, body: In):
    if X_API_KEY and req.headers.get("x-api-key") != X_API_KEY:
        raise HTTPException(status_code=401, detail="bad x-api-key")
    return {"received": body.image_url, "status": "OK"}

"""MedLens web API + static frontend.

    uvicorn app.main:app --reload
"""
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from medlens.pipeline import MedLens

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 8 * 1024 * 1024

app = FastAPI(title="MedLens", version="0.1.0")
engine: MedLens | None = None


@app.on_event("startup")
def load_engine() -> None:
    global engine
    engine = MedLens()


def _matches(matches) -> list[dict]:
    cat = engine.catalog
    return [
        {
            "id": int(cat.at[m.index, "id"]),
            "name": m.name,
            "composition": m.composition,
            "manufacturer": cat.at[m.index, "manufacturer_name"],
            "pack": cat.at[m.index, "pack_size_label"],
            "price": float(cat.at[m.index, "price"]),
            "score": m.score,
        }
        for m in matches
    ]


@app.get("/api/health")
def health() -> dict:
    return {"ok": engine is not None, "medicines": 0 if engine is None else len(engine.catalog)}


@app.get("/api/search")
def search(q: str = Query(..., min_length=2, max_length=500), limit: int = Query(5, ge=1, le=20)) -> dict:
    return {"query": q, "matches": _matches(engine.search(q, limit=limit))}


@app.post("/api/scan")
async def scan(image: UploadFile = File(...), limit: int = Query(5, ge=1, le=20)) -> dict:
    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 8 MB)")
    try:
        text, matches = engine.scan(data, limit=limit)
    except Exception as exc:  # unreadable / non-image upload
        raise HTTPException(400, f"Could not read image: {exc}") from exc
    return {"ocr_text": text, "matches": _matches(matches)}


@app.get("/api/medicine/{medicine_id}")
def medicine(medicine_id: int) -> dict:
    try:
        return asdict(engine.medicine(medicine_id))
    except KeyError as exc:
        raise HTTPException(404, "Medicine not found") from exc


class ReportRequest(BaseModel):
    ids: list[int] = Field(..., min_length=1, max_length=20)


@app.post("/api/report")
def report(req: ReportRequest) -> dict:
    try:
        return engine.report(req.ids)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown medicine id {exc}") from exc


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

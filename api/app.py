"""FastAPI routes for glare-removal inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from api.predictor import DeglarePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    predictor = DeglarePredictor()
    predictor.load_model()
    app.state.predictor = predictor
    yield


app = FastAPI(title="HTX Deglare Service", lifespan=lifespan)


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Simple health endpoint."""
    return {"message": "pong"}


@app.post("/infer")
async def infer(request: Request, image: UploadFile = File(...)) -> dict[str, str]:
    """Run de-glaring inference and return base64-encoded grayscale PNG."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    predictor: DeglarePredictor = request.app.state.predictor

    try:
        output_b64 = predictor.predict_base64_png(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    return {"image": output_b64}

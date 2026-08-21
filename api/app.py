"""FastAPI para serving del score de propensión."""
from fastapi import FastAPI, HTTPException

from api.predictor import ModelPredictor
from api.schemas import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="Seguro Antirrobos - Propensión de Compra",
    version="1.0.0",
    description="API MLOps para scoring de propensión de compra.",
)
predictor = ModelPredictor()


@app.get("/health")
def health():
    return {"status": "ok", "model_available": predictor.available}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    try:
        return predictor.predict(features.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

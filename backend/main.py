import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import os
import warnings
warnings.filterwarnings("ignore")

from model_registry import ALGORITHM_REGISTRY

app = FastAPI(title="MLVerse — Supervised Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Supervised Learning")

pipelines: dict = {}


@app.on_event("startup")
def load_models():
    for key, cfg in ALGORITHM_REGISTRY.items():
        pkl = cfg.get("pkl_path")
        if pkl:
            full_path = os.path.join(BASE_DIR, pkl)
            try:
                print(f"Loading {key} from {full_path} ...")
                pipelines[key] = joblib.load(full_path)
                print(f"  ✓ {key} loaded.")
            except Exception as e:
                print(f"  ✗ {key} failed: {e}")


# ── GET /models — full registry for frontend ────────────────────────────
@app.get("/models")
def get_models():
    result = {}
    for key, cfg in ALGORITHM_REGISTRY.items():
        pipeline = pipelines.get(key)
        features = {}

        if isinstance(pipeline, dict):
            fnames = pipeline.get('feature_names', [])
            display = pipeline.get('feature_names_display', {})
            le_dict = pipeline.get('label_encoders', {})

            for fname in fnames:
                label = display.get(fname, fname)
                if fname in le_dict:
                    le = le_dict[fname]
                    features[fname] = {
                        "label": label,
                        "type": "select",
                        "options": [str(c) for c in le.classes_],
                        "default": str(le.classes_[0]),
                    }
                else:
                    features[fname] = {
                        "label": label,
                        "type": "number",
                        "default": 0,
                    }
        else:
            # CatBoost/XGBoost pkl is raw model, use fallback features from registry
            pass

        result[key] = {
            "name": cfg["name"],
            "algorithm": cfg["algorithm"],
            "dataset": cfg["dataset"],
            "target": cfg["target"],
            "loaded": key in pipelines,
            "features": features,
        }
    return result


# ── GET /chart-data/{model_id} — pre-computed chart data ────────────────
@app.get("/chart-data/{model_id}")
def get_chart_data(model_id: str):
    if model_id not in pipelines:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    pipeline = pipelines[model_id]
    if isinstance(pipeline, dict) and 'chart_data' in pipeline:
        return pipeline['chart_data']

    return {"error": "No chart data available for this model"}


# ── POST /predict — unified prediction ──────────────────────────────────
class PredictionRequest(BaseModel):
    model: str
    data: Dict[str, Any]


@app.post("/predict")
def predict(request: PredictionRequest):
    model_name = request.model

    if model_name not in pipelines:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded.")

    pipeline = pipelines[model_name]

    try:
        if isinstance(pipeline, dict):
            model = pipeline['model']
            scaler = pipeline.get('scaler')
            le_dict = pipeline.get('label_encoders', {})
            fnames = pipeline.get('feature_names', [])

            row = {}
            for col in fnames:
                val = request.data.get(col, 0)
                if col in le_dict:
                    le = le_dict[col]
                    s = str(val)
                    if s in list(le.classes_):
                        row[col] = int(le.transform([s])[0])
                    else:
                        row[col] = 0
                else:
                    try:
                        row[col] = float(val)
                    except:
                        row[col] = 0.0

            df = pd.DataFrame([row])[fnames]

            if scaler is not None:
                X = scaler.transform(df)
            else:
                X = df.values

            prediction = model.predict(X)
            return {"prediction": round(float(prediction[0]), 4)}

        else:
            # Raw model (CatBoost/XGBoost legacy pkl)
            df = pd.DataFrame([request.data])
            prediction = pipeline.predict(df)
            return {"prediction": round(float(prediction[0]), 4)}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")


@app.get("/")
def read_root():
    return {"message": "MLVerse API", "total_models": len(ALGORITHM_REGISTRY)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

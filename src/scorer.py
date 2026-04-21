import joblib
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ml.features import (
    engineer_features,
    features_to_array,
    get_score_category
    )

load_dotenv()

MODEL_PATH = Path(__file__).parent.parent / os.getenv(
    "MODEL_PATH", "data/models/credit_model.joblib"
)

# Variable global para el modelo - se carga una vez al arrancar
# Cargarlo en cada request sería muy lento
_model = None

def load_model():
    """
    Carga el modelo desde disco si no está en memoria.
    Patrón singleton - una sola instancia del modelo en toda la app
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {MODEL_PATH}. "
                f"Ejecuta src/ml/trainer.py primero."
            )
        _model = joblib.load(MODEL_PATH)
    return _model

def probability_to_score(default_prob: float) -> int:
    """
    Convierte probabilidad de default a score en escala 300-850
    Relación inversa - mayor probabilidad de default = menor score

    La fórmula mapea linealmente:
    prob=0.0 -> score=850 (sin riesgo)
    prob=1.0 -> score=300 (máxima riesgo)
    """
    score = 850 - int(default_prob * 550)
    return max(300, min(850, score))

def calculate_score(input_data: dict) -> dict:
    """
    Punto de entrada del scorer
    Toma los datos crudos del cliente y retorna el score completo
    """
    model    = load_model()
    features = engineer_features(input_data)
    X        = features_to_array(features)

    # predict_proba retorna [prob_no_default, prob_default]
    # Tomamos el índice 1 - probabilidad de default
    proba = model.predict_proba(X)[0]
    default_prob = float(proba[1])
    score = probability_to_score(default_prob)
    category = get_score_category(score)

    # Feature importance — qué factores influyeron más en el score
    # RandomForest expone esto directamente via feature_importances_
    rf_model = model.named_steps["model"]
    importances = rf_model.feature_importances_

    feature_names = [
        "age", "income_monthly", "debt_to_income_ratio",
        "credit_history_years", "num_late_payments",
        "num_credit_accounts", "employment_years",
        "loan_amount", "loan_to_income_ratio"
    ]

    # Top 3 factores más influyentes para este perfil
    importance_dict = dict(zip(feature_names, importances))
    top_factors = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return {
        "score":            score,
        "default_prob":     round(default_prob, 4),
        "category":         category["category"],
        "decision":         category["decision"],
        "description":      category["description"],
        "top_factors":      top_factors,
        "features":         features
    }
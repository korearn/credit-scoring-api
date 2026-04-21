from fastapi import APIRouter, HTTPException, Query
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import CreditRequest, ScoreResponse, HealthResponse
from scorer import calculate_score, load_model
from explainer import get_explanation
from database import save_score, get_history, get_stats

load_dotenv()
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Estado del servicio y verificación del modelo."""
    try:
        load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    return HealthResponse(
        status="ok",
        version=os.getenv("APP_VERSION", "1.0.0"),
        model_loaded=model_loaded
    )


@router.post("/score", response_model=ScoreResponse)
def calculate_credit_score(request: CreditRequest):
    """
    Calcula el score crediticio y genera una explicación con IA.
    POST porque enviamos datos sensibles del cliente — nunca en la URL.
    """
    try:
        input_data   = request.model_dump()
        score_result = calculate_score(input_data)
        explanation  = get_explanation(score_result, input_data)
        request_id   = save_score(score_result, input_data, explanation)

        return ScoreResponse(
            score=score_result["score"],
            category=score_result["category"],
            decision=score_result["decision"],
            description=score_result["description"],
            default_prob=score_result["default_prob"],
            explanation=explanation,
            request_id=request_id
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando score: {e}")


@router.get("/history")
def score_history(limit: int = Query(20, ge=1, le=100)):
    """Historial de scores calculados para auditoría."""
    history = get_history(limit)
    return {"history": history, "total": len(history)}


@router.get("/stats")
def scoring_stats():
    """Estadísticas generales del sistema."""
    stats = get_stats()
    return {"stats": stats}
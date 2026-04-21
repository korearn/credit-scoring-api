from pydantic import BaseModel, Field
from typing import Optional


class CreditRequest(BaseModel):
    """
    Datos del solicitante que llegan via POST /score.
    Pydantic valida automáticamente tipos y rangos antes de
    que el código de negocio siquiera vea los datos.
    """
    age:                    int   = Field(..., ge=18, le=75,       description="Edad del solicitante")
    income_monthly:         float = Field(..., gt=0,               description="Ingreso mensual en MXN")
    loan_amount:            float = Field(..., gt=0,               description="Monto del crédito solicitado")
    total_debt:             float = Field(0,   ge=0,               description="Deuda total actual en MXN")
    credit_history_years:   float = Field(0,   ge=0, le=50,        description="Años de historial crediticio")
    num_late_payments:      int   = Field(0,   ge=0, le=50,        description="Pagos tardíos en últimos 24 meses")
    num_credit_accounts:    int   = Field(1,   ge=1, le=30,        description="Cuentas de crédito activas")
    employment_years:       float = Field(0,   ge=0, le=50,        description="Años en empleo actual")


class ScoreResponse(BaseModel):
    """Respuesta completa del endpoint /score."""
    score:          int
    category:       str
    decision:       str
    description:    str
    default_prob:   float
    explanation:    str
    request_id:     int


class HealthResponse(BaseModel):
    status:         str
    version:        str
    model_loaded:   bool


class ScoreHistoryItem(BaseModel):
    """Item del historial de scores para auditoría."""
    id:             int
    score:          int
    category:       str
    decision:       str
    age:            int
    income_monthly: float
    loan_amount:    float
    created_at:     str
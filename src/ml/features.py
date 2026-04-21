import numpy as np
from dataclasses import dataclass

@dataclass
class CreditFeatures:
    """
    Features numéricas que el modelo ML usa para calcular el score.
    Cada campo representa un factor de riesgo crediticio real.
    """
    age:                    float  # edad del solicitante
    income_monthly:         float  # ingreso mensual en MXN
    debt_to_income_ratio:   float  # deuda total / ingreso anual (0-1)
    credit_history_years:   float  # años de historial crediticio
    num_late_payments:      int    # pagos tardíos en últimos 24 meses
    num_credit_accounts:    int    # número de cuentas de crédito activas
    employment_years:       float  # años en empleo actual
    loan_amount:            float  # monto del crédito solicitado
    loan_to_income_ratio:   float  # monto solicitado / ingreso anual

def engineer_features(raw_input: dict) -> CreditFeatures:
    """
    Transforma los datos crudos del cliente en features para el modelo.
    Calcula ratios derivados que el modelo encuentra más informativos
    que los valores absolutos.
    """
    income_monthly = raw_input['income_monthly']
    income_annual = income_monthly * 12
    total_debt = raw_input.get('total_debt', 0)
    loan_amount = raw_input['loan_amount']

    # Debt-to-income ratio — métrica estándar en análisis crediticio
    # Un ratio > 0.43 es generalmente considerado riesgoso por los bancos
    dti = total_debt / income_annual if income_annual > 0 else 1.0
    dti = min(dti, 2.0)  # capear a 2.0 para evitar valores extremos

    # Loan-to-income ratio — qué tan grande es el préstamo vs ingresos
    lti = loan_amount / income_annual if income_annual > 0 else 1.0
    lti = min(lti, 5.0)

    return CreditFeatures(
        age=raw_input['age'],
        income_monthly=income_monthly,
        debt_to_income_ratio=dti,
        credit_history_years=raw_input.get('credit_history_years', 0),
        num_late_payments=raw_input.get('num_late_payments', 0),
        num_credit_accounts=raw_input.get('num_credit_accounts', 1),
        employment_years=raw_input.get('employment_years', 0),
        loan_amount=loan_amount,
        loan_to_income_ratio=lti
    )

def features_to_array(features: CreditFeatures) -> np.ndarray:
    """
    Convierte el dataclass CreditFeatures a un array de numpy.
    El orden de las features debe coincidir con el orden esperado por el modelo ML.
    """
    return np.array([[
        features.age,
        features.income_monthly,
        features.debt_to_income_ratio,
        features.credit_history_years,
        features.num_late_payments,
        features.num_credit_accounts,
        features.employment_years,
        features.loan_amount,
        features.loan_to_income_ratio
    ]])

def get_feature_names() -> list:
    """
    Retorna la lista de nombres de features en el orden que el modelo espera.
    Esto es útil para debugging y para asegurar que el preprocesamiento es consistente.
    """
    return [
        'age',
        'income_monthly',
        'debt_to_income_ratio',
        'credit_history_years',
        'num_late_payments',
        'num_credit_accounts',
        'employment_years',
        'loan_amount',
        'loan_to_income_ratio'
    ]

def get_score_category(score: int) -> dict:
    """
    Clasifica el score numérico en categorías de riesgo.
    Estas categorías pueden ser usadas para tomar decisiones de negocio o para explicar el resultado al cliente.
    """
    if score >= 750:
        return {"category": "Excelente", "color": "green", "decision": "Aprobado", "description": "Historial credicio excepcional. Acceso a mejores tasas."}
    elif score >= 670:
        return {"category": "Bueno", "color": "blue", "decision": "Aprobado", "description": "Buen historial. Elegible para la mayoría de productos."}
    elif score >= 580:
        return {"category": "Regular", "color": "yellow", "decision": "Condicional", "description": "Historial con algunos factores de riesgo. Requiere garantías."}
    else:
        return {"category": "Malo", "color": "red", "decision": "Rechazado", "description": "Alto riesgo de incumplimiento. No elegible actualmente."}
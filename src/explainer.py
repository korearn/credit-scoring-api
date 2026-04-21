import requests
import os
from dotenv import load_dotenv

load_dotenv()

LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions")

def build_explanation_prompt(score_result: dict, input_data: dict) -> str:
    """
    Construye el prompt con el contexto completo del scoring.
    El LLM recibe tanto los datos del cliente como el resultado
    para generar una explicación relevante y personaliada.
    """
    features  = score_result["features"]
    top_facts = "\n".join([
        f"- {factor}: importancia {importance:.3f}"
        for factor, importance in score_result["top_factors"]
    ])
    
    return f"""Eres un analista de crédito experto explicando una decisión crediticia a un cliente.

RESULTADO DEL ANÁLISIS:
- Score crediticio: {score_result['score']}/850
- Categoría: {score_result['category']}
- Decisión: {score_result['decision']}
- Probabilidad de incumplimiento: {score_result['default_prob']*100:.1f}%

PERFIL DEL SOLICITANTE:
- Edad: {input_data['age']} años
- Ingreso mensual: ${input_data['income_monthly']:,.0f} MXN
- Monto solicitado: ${input_data['loan_amount']:,.0f} MXN
- Años de historial crediticio: {input_data.get('credit_history_years', 0)}
- Pagos tardíos (24 meses): {input_data.get('num_late_payments', 0)}
- Años en empleo actual: {input_data.get('employment_years', 0)}
- Ratio deuda/ingreso: {features.debt_to_income_ratio:.2f}

FACTORES MÁS INFLUYENTES EN EL SCORE:
{top_facts}

Genera una explicación clara y profesional (máximo 3 párrafos) que incluya:
1. Por qué el cliente obtuvo este score
2. Los factores positivos y negativos más importantes
3. Recomendaciones concretas para mejorar el score

Responde en español, con tono profesional pero accesible."""

def get_explanation(score_result: dict, input_data: dict) -> str:
    """
    Llama al LLM para generar la explicación.
    Si LMStudio no está disponible, da una explicación básica.
    """
    prompt = build_explanation_prompt(score_result, input_data)

    payload = {
        "model": "local-mo0del",
        "messages": [
            {
                "role": "system", 
             "content": "Eres un analista financiero experto. Explicas decisiones crediticias de forma clara, profesional y empática."
            },
            {
                "role":"user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    try:
        response = requests.post(
            LMSTUDIO_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return (
            f"Score {score_result['score']}/850 — {score_result['category']}. "
            f"Decisión: {score_result['decision']}. "
            f"LMStudio no disponible para explicación detallada."
        )
    except Exception as e:
        return f"Explicación no disponible: {e}"

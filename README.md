# Credit Scoring API 🏦

API REST para scoring crediticio con Machine Learning e IA explicable.
Calcula un score entre 300-850, clasifica el riesgo y genera una explicación
en lenguaje natural usando un LLM local via LMStudio.

## ¿Qué hace?
  
- Recibe datos financieros del solicitante via POST
- Calcula score crediticio con un modelo Random Forest entrenado
- Genera explicación personalizada usando un LLM local
- Registra cada decisión en SQLite para auditoría regulatoria
- Expone estadísticas y historial de scores 

## Stack técnico

- **Python 3.12** — lenguaje principal
- **FastAPI** — framework REST con documentación Swagger automática
- **scikit-learn** — modelo Random Forest para predicción de riesgo
- **joblib** — serialización del modelo ML entrenado
- **LMStudio** — LLM local para explicaciones en lenguaje natural
- **Pydantic** — validación de datos de entrada
- **SQLite** — auditoría de decisiones crediticias
- **Uvicorn** — servidor ASGI
 
## Arquitectura

```bash
POST /score

│
├── Pydantic valida los datos
├── Feature engineering (DTI, LTI, ratios)
├── Random Forest → probabilidad de default
├── Conversión a score 300-850
├── LLM local → explicación en lenguaje natural
├── SQLite → registro de auditoría
└── JSON response con score + explicación
```

## Categorías de score

```bash
| Rango | Categoría | Decisión |
|---|---|---|
| 750-850 | Excelente | Aprobado — mejor tasa |
| 670-749 | Bueno | Aprobado |
| 580-669 | Regular | Condicional |
| 300-579 | Malo | Rechazado | 
```

## Estructura

```bash
credit-scoring-api/
├── data/
│   └── models/             # Modelo ML serializado (.joblib)
├── src/
│   ├── ml/
│   │   ├── features.py     # Feature engineering y categorías
│   │   └── trainer.py      # Entrenamiento del modelo
│   ├── scorer.py           # Carga modelo y calcula score
│   ├── explainer.py        # Explicación via LLM local
│   ├── database.py         # Auditoría SQLite
│   ├── models.py           # Schemas Pydantic
│   ├── routes.py           # Endpoints REST
│   └── main.py             # Servidor FastAPI
└── requirements.txt
```

## Instalación

```bash
git clone https://github.com/korearn/credit-scoring-api
cd credit-scoring-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenar el modelo

```bash
cd src/ml
python trainer.py
```

Genera 5,000 perfiles sintéticos y entrena un Random Forest.
El modelo se guarda en `data/models/credit_model.joblib`.

## Uso

```bash
cd src
uvicorn main:app --reload
```

Documentación interactiva en `http://localhost:8000/docs`


## Endpoints

```bash
| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/v1/health` | Estado del servicio y modelo |
| POST | `/api/v1/score` | Calcula score con explicación IA |
| GET | `/api/v1/history` | Historial para auditoría |
| GET | `/api/v1/stats` | Estadísticas generales |
```

## Ejemplo de request

```bash
curl -X POST "http://localhost:8000/api/v1/score" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income_monthly": 45000,
    "loan_amount": 150000,
    "total_debt": 50000,
    "credit_history_years": 10,
    "num_late_payments": 0,
    "num_credit_accounts": 3,
    "employment_years": 8
  }'
```

## Ejemplo de respuesta

```json
{
  "score": 820,
  "category": "Excelente",
  "decision": "Aprobado",
  "description": "Historial crediticio excepcional. Acceso a mejores tasas.",
  "default_prob": 0.0562,
  "explanation": "El solicitante presenta un perfil crediticio sólido...",
  "request_id": 1
}
```

## Notas técnicas
 
- El modelo usa `StandardScaler + RandomForest` en un Pipeline de scikit-learn
- `predict_proba()` retorna probabilidad de default convertida a escala 300-850
- `feature_importances_` expone qué factores influyeron más en cada decisión
- Si LMStudio no está disponible el sistema usa un fallback automático
- Cada score queda registrado en SQLite para auditoría regulatoria
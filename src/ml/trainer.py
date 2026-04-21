import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from features import get_feature_names

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "credit_model.joblib"

def generate_training_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Genera un dataset sintético para entrenamiento del modelo de crédito.
    Las features se generan con distribuciones realistas basadas en datos de crédito típicos.
    """
    np.random.seed(42)  # Para reproducibilidad
    n = n_samples

    # Features del solicitante
    age                  = np.random.normal(38, 12, n).clip(18, 75)
    income_monthly       = np.random.lognormal(10, 0.5, n).clip(5000, 200000)
    credit_history_years = np.random.exponential(7, n).clip(0, 40)
    num_late_payments    = np.random.poisson(1.5, n).clip(0, 20)
    num_credit_accounts  = np.random.poisson(3, n).clip(1, 15)
    employment_years     = np.random.exponential(5, n).clip(0, 40)
    loan_amount          = np.random.lognormal(11, 0.8, n).clip(10000, 2000000)

    income_annual = income_monthly * 12
    total_debt = np.random.uniform(0, income_annual * 1.5, n)

    dti = (total_debt / income_annual).clip(0, 2.0)
    lti = (loan_amount / income_annual).clip(0, 5.0)

    # Probabilidad de default basada en factores de riesgo reales
    # Cada factor contribuye positiva o negativamente al riesgo de default
    default_prob = (
        0.05                                        # base rate
        + 0.15 * (num_late_payments / 10)           # pagos tardíos aumentan riesgo
        + 0.20 * dti                                # alto DTI aumenta riesgo
        + 0.10 * lti                                # alto LTI aumenta riesgo
        - 0.10 * (credit_history_years / 40)        # más historial reduce riesgo
        - 0.05 * (employment_years / 40)            # más empleo reduce riesgo
        - 0.05 * (income_monthly / 200000)          # más ingreso reduce riesgo
        + np.random.normal(0, 0.05, n)              # ruido aleatorio
    ).clip(0, 1)

    default = (np.random.random(n) < default_prob).astype(int)

    return pd.DataFrame({
        "age": age,
        "income_monthly": income_monthly,
        "debt_to_income_ratio": dti,
        "credit_history_years": credit_history_years,
        "num_late_payments": num_late_payments,
        "num_credit_accounts": num_credit_accounts,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_to_income_ratio": lti,
        "default": default
    })

def train_model() -> dict:
    """
    Entrena un modelo de clasificación para predecir el riesgo de crédito.
    Utiliza un pipeline con escalado de features y un Random Forest.
    Devuelve el modelo entrenado y un reporte de clasificación en el set de prueba.
    """
    print("Generando datos de entrenammiento...")
    df = generate_training_data(5000)

    feature_names = get_feature_names()
    X = df[feature_names].values
    y = df["default"].values

    # Split 80% entrenamiento, 20% evaluación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entrenando modelo Random Forest...")
    pipeline = Pipeline([
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(
            n_estimators=100,   # 100 árboles en el ensemble
            max_depth=10,       # profundidad máxima de cada árbol
            random_state=42,
            class_weight="balanced"  # compensa el desbalance de clases
        ))
    ])

    pipeline.fit(X_train, y_train)
    
    # Evaluación del modelo
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Guardar modelo en disco
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✓ Modelo guardado en: {MODEL_PATH}")

    return {
        "accuracy":  report["accuracy"],
        "model_path": str(MODEL_PATH)
    }


if __name__ == "__main__":
    result = train_model()
    print(f"\nAccuracy: {result['accuracy']:.3f}")
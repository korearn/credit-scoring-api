import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent))

from routes import router
from database import init_db
from scorer import load_model

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa BD y carga el modelo al arrancar."""
    print("🚀 Iniciando Credit Scoring API...")
    init_db()
    print("✓ Base de datos lista")
    try:
        load_model()
        print("✓ Modelo ML cargado")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
    yield
    print("👋 Servidor apagado")

app = FastAPI(
    title=os.getenv("APP_NAME", "Credit Scoring API"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="""
API REST para scoring crediticio con IA explicable.

## Endpoints

- **/health** — Estado del servicio y modelo ML
- **/score** — Calcula score crediticio con explicación IA
- **/history** — Historial de scores para auditoría
- **/stats** — Estadísticas generales del sistema
    """,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "message": "Credit Scoring API",
        "docs":    "http://localhost:8000/docs",
        "version": os.getenv("APP_VERSION", "1.0.0")
    }
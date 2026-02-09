# Configuration automatique pour l'API CV Classifier
# Compatible macOS/Linux/Windows

from pathlib import Path

# Détection automatique des chemins
API_DIR = Path(__file__).parent.resolve()
BASE_DIR = API_DIR.parent
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"

# Informations sur le modèle
MODEL_INFO = {
    "name": "Gradient_Boosting",
    "version": "2.0.0",
    "trained_on": "962 CVs",
    "categories": 25,
    "test_f1_score": 1.0,
    "cv_f1_score": 0.9896,
    "features": 8000
}

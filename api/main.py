"""
api/main.py - VERSION COMPLÈTE FINALE
API REST pour la classification de CV avec toutes les fonctionnalités avancées
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

print("\n" + "="*80)
print(" INITIALISATION DE L'API - VERSION COMPLÈTE")
print("="*80 + "\n")

# ============================================
# ÉTAPE 1: DÉTERMINER LES CHEMINS ABSOLUS
# ============================================

print(" ÉTAPE 1: Détection des chemins...")

# Chemin de ce fichier (api/main.py)
THIS_FILE = Path(__file__).resolve()
print(f"   Ce fichier: {THIS_FILE}")

# Dossier api
API_DIR = THIS_FILE.parent
print(f"   Dossier API: {API_DIR}")

# Dossier racine du projet (parent de api/)
BASE_DIR = API_DIR.parent
print(f"   Dossier projet: {BASE_DIR}")

# Dossier des modèles
MODELS_DIR = BASE_DIR / "models"
print(f"   Dossier modèles: {MODELS_DIR}")

# Dossier src
SRC_DIR = BASE_DIR / "src"
print(f"   Dossier src: {SRC_DIR}")

# Dossier data
DATA_DIR = BASE_DIR / "data"
print(f"   Dossier data: {DATA_DIR}")

print()

# ============================================
# ÉTAPE 2: VÉRIFICATIONS
# ============================================

print(" ÉTAPE 2: Vérifications...")

# Vérifier que models existe
if not MODELS_DIR.exists():
    print(f" ERREUR CRITIQUE: {MODELS_DIR} n'existe pas!")
    print(f"   Créez ce dossier et placez-y vos modèles")
    sys.exit(1)
else:
    print(f" Dossier models trouvé")

# Créer le dossier data s'il n'existe pas
DATA_DIR.mkdir(exist_ok=True)
print(f" Dossier data vérifié")

# Lister les fichiers de modèles
print(f"\n Fichiers dans {MODELS_DIR}:")
for file in MODELS_DIR.iterdir():
    if file.is_file():
        print(f"   - {file.name}")

# Ajouter src au path Python
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
    print(f"\n {SRC_DIR} ajouté au path Python")
else:
    print(f"\n  {SRC_DIR} n'existe pas")

print()

# ============================================
# IMPORTS DES MODULES
# ============================================

# Import du TextCleaner
try:
    from preprocessing.text_cleaner import TextCleaner
    TEXT_CLEANER_AVAILABLE = True
    print(" TextCleaner importé avec succès")
except ImportError as e:
    TEXT_CLEANER_AVAILABLE = False
    print(f"  Impossible d'importer TextCleaner: {e}")
    print("   L'API fonctionnera sans nettoyage de texte")

# Import des modules avancés (optionnels)
try:
    sys.path.insert(0, str(BASE_DIR / "src" / "database"))
    from db_manager import CVDatabaseManager
    DATABASE_AVAILABLE = True
    print(" Database Manager importé")
except ImportError:
    DATABASE_AVAILABLE = False
    CVDatabaseManager = None
    print("  Database Manager non disponible")

try:
    sys.path.insert(0, str(BASE_DIR / "src" / "pdf_processing"))
    from pdf_extractor import AdvancedPDFExtractor
    PDF_EXTRACTOR_AVAILABLE = True
    print(" PDF Extractor importé")
except ImportError:
    PDF_EXTRACTOR_AVAILABLE = False
    AdvancedPDFExtractor = None
    print("  PDF Extractor non disponible")

try:
    sys.path.insert(0, str(BASE_DIR / "src" / "skills_extraction"))
    from skills_detector import SkillsDetector
    SKILLS_DETECTOR_AVAILABLE = True
    print(" Skills Detector importé")
except ImportError:
    SKILLS_DETECTOR_AVAILABLE = False
    SkillsDetector = None
    print("  Skills Detector non disponible")

print()

# ============================================
# MODÈLES PYDANTIC
# ============================================

class CVInput(BaseModel):
    resume_text: str = Field(..., description="Texte complet du CV")

class CVPrediction(BaseModel):
    category: str
    confidence: float
    all_probabilities: Optional[dict] = None

class BatchCVInput(BaseModel):
    resumes: List[str]

class BatchCVPrediction(BaseModel):
    predictions: List[CVPrediction]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    label_encoder_loaded: bool
    text_cleaner_loaded: bool
    database_available: bool
    pdf_extractor_available: bool
    skills_detector_available: bool
    version: str
    base_dir: str
    models_dir: str

class CVUploadResponse(BaseModel):
    classification_id: Optional[int]
    filename: str
    predicted_category: str
    confidence: float
    extraction_method: Optional[str]
    extraction_confidence: Optional[float]
    skills_summary: Optional[Dict]
    experience_info: Optional[Dict]
    job_recommendations: Optional[List[Dict]]
    processing_time_ms: int

class SkillsAnalysisResponse(BaseModel):
    skills_summary: Dict
    detailed_skills: Dict
    experience_analysis: Dict
    job_recommendations: List[Dict]
    top_strengths: List[str]

class FeedbackUpdate(BaseModel):
    user_feedback: str = Field(..., description="'Correct' ou 'Incorrect'")
    correct_category: Optional[str] = None
    notes: Optional[str] = None

class StatisticsResponse(BaseModel):
    total_classifications: int
    avg_confidence: float
    category_distribution: List[Dict]
    top_skills: Optional[List[Dict]]
    accuracy_from_feedback: Optional[float]

# ============================================
# INITIALISATION FASTAPI
# ============================================

app = FastAPI(
    title="CV Classification API - Version Complète",
    description="""
API professionnelle pour classifier automatiquement des CV.

## Modèle
- **Type**: Gradient Boosting Classifier
- **Entraîné sur**: 962 CVs
- **Performance**: 100% accuracy, 98.96% CV F1-score
- **Catégories**: 25 métiers

## Fonctionnalités
- Classification de CV (texte ou PDF)
- Extraction PDF avec OCR
- Détection de 1000+ compétences
- Recommandations de postes
- Historique et statistiques
""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CHARGEMENT DES MODÈLES ML
# ============================================

print("="*80)
print(" CHARGEMENT DES MODÈLES ML")
print("="*80 + "\n")

# Variables globales pour les modèles
MODEL = None
VECTORIZER = None
LABEL_ENCODER = None
TEXT_CLEANER = None

# Variables pour les modules avancés
db_manager = None
pdf_extractor = None
skills_detector = None

def load_pickle(filename):
    """Charger un fichier pickle de manière sécurisée"""
    filepath = MODELS_DIR / filename
    
    if not filepath.exists():
        print(f" {filename} non trouvé dans {MODELS_DIR}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f" {filename} chargé avec succès")
        return obj
    except Exception as e:
        print(f" Erreur lors du chargement de {filename}: {e}")
        return None

# Charger le modèle ML
print(" Chargement du modèle ML...")
MODEL = load_pickle("best_model.pkl")
if MODEL is None:
    for name in ["Random_Forest_model.pkl", "Logistic_Regression_model.pkl", "SVM_model.pkl"]:
        print(f"   Essai avec {name}...")
        MODEL = load_pickle(name)
        if MODEL is not None:
            break

# Charger le vectorizer
print("\n Chargement du vectorizer...")
VECTORIZER = load_pickle("tfidf_vectorizer.pkl")

# Charger le label encoder
print("\n Chargement du label encoder...")
LABEL_ENCODER = load_pickle("label_encoder.pkl")
if LABEL_ENCODER is not None and hasattr(LABEL_ENCODER, 'classes_'):
    print(f"   Catégories: {len(LABEL_ENCODER.classes_)}")

# Initialiser le text cleaner
print("\n Initialisation du text cleaner...")
if TEXT_CLEANER_AVAILABLE:
    try:
        TEXT_CLEANER = TextCleaner()
        print(" TextCleaner initialisé")
    except Exception as e:
        print(f" Erreur: {e}")
else:
    print("  TextCleaner non disponible - utilisation d'un nettoyage basique")

# Résumé ML
print("\n" + "="*80)
all_ml_loaded = all([MODEL, VECTORIZER, LABEL_ENCODER])
if all_ml_loaded:
    print("  TOUS LES MODÈLES ML CHARGÉS AVEC SUCCÈS!")
else:
    print("  CERTAINS MODÈLES ML SONT MANQUANTS:")
    if not MODEL:
        print("    Modèle ML")
    if not VECTORIZER:
        print("    Vectorizer")
    if not LABEL_ENCODER:
        print("    Label Encoder")
print("="*80 + "\n")

# ============================================
# INITIALISATION DES MODULES AVANCÉS
# ============================================

print("="*80)
print(" INITIALISATION DES MODULES AVANCÉS")
print("="*80 + "\n")

# Base de données
if DATABASE_AVAILABLE and CVDatabaseManager:
    try:
        db_manager = CVDatabaseManager(str(DATA_DIR / "cv_history.db"))
        print(" Base de données initialisée")
    except Exception as e:
        print(f" Erreur base de données: {e}")
        db_manager = None

# Extracteur PDF
if PDF_EXTRACTOR_AVAILABLE and AdvancedPDFExtractor:
    try:
        pdf_extractor = AdvancedPDFExtractor()
        print(" Extracteur PDF initialisé")
    except Exception as e:
        print(f" Erreur extracteur PDF: {e}")
        pdf_extractor = None

# Détecteur de compétences
if SKILLS_DETECTOR_AVAILABLE and SkillsDetector:
    try:
        skills_detector = SkillsDetector()
        print(" Détecteur de compétences initialisé")
    except Exception as e:
        print(f" Erreur détecteur de compétences: {e}")
        skills_detector = None

print("="*80 + "\n")

# ============================================
# FONCTION DE NETTOYAGE BASIQUE (fallback)
# ============================================

def basic_clean(text):
    """Nettoyage basique si TextCleaner n'est pas disponible"""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================
# ENDPOINTS DE BASE
# ============================================

@app.get("/")
def root():
    """Endpoint racine avec informations sur l'API"""
    return {
        "message": "CV Classification API - Version Complète",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "ml_classification": all([MODEL, VECTORIZER, LABEL_ENCODER]),
            "pdf_extraction": pdf_extractor is not None,
            "skills_detection": skills_detector is not None,
            "database_history": db_manager is not None
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "upload_cv": "/upload-cv",
            "analyze_skills": "/analyze-skills",
            "history": "/history",
            "statistics": "/statistics",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health():
    """Vérifier l'état de santé de l'API"""
    return HealthResponse(
        status="healthy" if all([MODEL, VECTORIZER, LABEL_ENCODER]) else "degraded",
        model_loaded=MODEL is not None,
        vectorizer_loaded=VECTORIZER is not None,
        label_encoder_loaded=LABEL_ENCODER is not None,
        text_cleaner_loaded=TEXT_CLEANER is not None,
        database_available=db_manager is not None,
        pdf_extractor_available=pdf_extractor is not None,
        skills_detector_available=skills_detector is not None,
        version="2.0.0",
        base_dir=str(BASE_DIR),
        models_dir=str(MODELS_DIR)
    )

@app.post("/predict", response_model=CVPrediction, tags=["Classification"])
def predict(cv: CVInput, include_all_probabilities: bool = False):
    """Prédire la catégorie d'un CV depuis un texte"""
    # Vérifier que tous les composants sont chargés
    if not all([MODEL, VECTORIZER, LABEL_ENCODER]):
        missing = []
        if not MODEL:
            missing.append("Modèle ML")
        if not VECTORIZER:
            missing.append("Vectorizer")
        if not LABEL_ENCODER:
            missing.append("Label Encoder")
        
        raise HTTPException(
            status_code=503,
            detail=f"Composants manquants: {', '.join(missing)}. Vérifiez {MODELS_DIR}"
        )
    
    # Vérifier le texte
    if not cv.resume_text or len(cv.resume_text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Le CV doit contenir au moins 10 caractères"
        )
    
    try:
        # Nettoyer le texte
        if TEXT_CLEANER:
            cleaned_text = TEXT_CLEANER.clean_text(cv.resume_text)
        else:
            cleaned_text = basic_clean(cv.resume_text)
        
        # Vectoriser
        X = VECTORIZER.transform([cleaned_text])
        
        # Prédire
        prediction = MODEL.predict(X)[0]
        probabilities = MODEL.predict_proba(X)[0]
        
        # Décoder
        category = LABEL_ENCODER.inverse_transform([prediction])[0]
        confidence = float(probabilities.max())
        
        # Probabilités pour toutes les catégories
        all_probs = None
        if include_all_probabilities:
            all_probs = {
                LABEL_ENCODER.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return CVPrediction(
            category=category,
            confidence=confidence,
            all_probabilities=all_probs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/batch-predict", response_model=BatchCVPrediction, tags=["Classification"])
def batch_predict(batch: BatchCVInput, include_all_probabilities: bool = False):
    """Prédire les catégories de plusieurs CV en batch"""
    predictions = []
    
    for resume_text in batch.resumes:
        if resume_text and len(resume_text.strip()) >= 10:
            try:
                result = predict(
                    CVInput(resume_text=resume_text),
                    include_all_probabilities
                )
                predictions.append(result)
            except:
                predictions.append(CVPrediction(
                    category="ERROR",
                    confidence=0.0
                ))
        else:
            predictions.append(CVPrediction(
                category="INVALID",
                confidence=0.0
            ))
    
    return BatchCVPrediction(
        predictions=predictions,
        total_processed=len(predictions)
    )

@app.get("/categories", tags=["Information"])
def get_categories():
    """Obtenir la liste de toutes les catégories disponibles"""
    if not LABEL_ENCODER:
        raise HTTPException(status_code=503, detail="Label encoder non chargé")
    
    categories = LABEL_ENCODER.classes_.tolist()
    return {
        "total_categories": len(categories),
        "categories": sorted(categories)
    }

@app.get("/model-info", tags=["Information"])
def get_model_info():
    """Obtenir des informations sur le modèle chargé"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    return {
        "model_type": type(MODEL).__name__,
        "model_version": "2.0.0",
        "trained_on": "962 CVs",
        "n_features": VECTORIZER.max_features if VECTORIZER else None,
        "n_categories": len(LABEL_ENCODER.classes_) if LABEL_ENCODER else None,
        "categories": LABEL_ENCODER.classes_.tolist() if LABEL_ENCODER else None,
        "performance": {
            "test_accuracy": "100%",
            "test_f1_score": "100%",
            "cv_f1_score": "98.96%",
            "note": "Validated with 5-fold cross-validation"
        }
    }

# ============================================
# ENDPOINTS AVANCÉS - UPLOAD PDF
# ============================================

@app.post("/upload-cv", response_model=CVUploadResponse, tags=["Advanced - PDF"])
async def upload_and_classify_cv(
    file: UploadFile = File(...),
    extract_skills: bool = Query(True, description="Extraire les compétences"),
    recommend_jobs: bool = Query(True, description="Recommander des postes"),
    save_to_history: bool = Query(True, description="Sauvegarder dans l'historique")
):
    """
    Upload et classification complète d'un CV PDF
    
    Fonctionnalités:
    - Extraction du texte PDF (avec OCR si nécessaire)
    - Classification du CV
    - Extraction des compétences (optionnel)
    - Recommandations de postes (optionnel)
    - Sauvegarde dans l'historique (optionnel)
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers PDF sont acceptés"
        )
    
    if not pdf_extractor:
        raise HTTPException(
            status_code=503,
            detail="PDF Extractor non disponible. Installez: pip install pdfplumber pypdf pytesseract pdf2image"
        )
    
    start_time = datetime.now()
    
    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # 1. Extraire le texte du PDF
        print(f" Extraction du PDF: {file.filename}")
        pdf_result = pdf_extractor.extract_from_pdf(tmp_path, method='auto')
        
        # 2. Classifier le CV
        print(f" Classification...")
        classification_result = predict(
            CVInput(resume_text=pdf_result.text),
            include_all_probabilities=True
        )
        
        predicted_category = classification_result.category
        confidence = classification_result.confidence
        all_probabilities = classification_result.all_probabilities
        
        # 3. Extraire les compétences si demandé
        skills_summary = None
        experience_info = None
        job_recommendations = None
        
        if extract_skills and skills_detector:
            print(" Extraction des compétences...")
            skills = skills_detector.extract_skills(pdf_result.text)
            experience_info = skills_detector.analyze_experience(pdf_result.text)
            
            skills_summary = {
                'total_technical_skills': len(skills['technical_skills']),
                'total_soft_skills': len(skills['soft_skills']),
                'total_frameworks': len(skills['frameworks']),
                'total_tools': len(skills['tools']),
                'total_languages': len(skills['languages'])
            }
            
            # 4. Recommander des postes si demandé
            if recommend_jobs:
                print(" Génération de recommandations...")
                job_recommendations = skills_detector.recommend_jobs(
                    skills,
                    experience_info,
                    top_n=5
                )
        
        # 5. Sauvegarder dans l'historique si demandé
        classification_id = None
        if save_to_history and db_manager:
            print(" Sauvegarde dans l'historique...")
            
            # Préparer les compétences pour la DB
            extracted_skills_list = []
            if extract_skills and skills_detector:
                for skill in skills.get('technical_skills', []):
                    extracted_skills_list.append({
                        'name': skill['skill'],
                        'category': skill['category'],
                        'confidence': skill['confidence']
                    })
            
            classification_id = db_manager.add_classification(
                cv_text=pdf_result.text,
                predicted_category=predicted_category,
                confidence_score=confidence,
                cv_filename=file.filename,
                all_probabilities=all_probabilities,
                model_used=type(MODEL).__name__,
                model_version="1.0",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                extracted_skills=extracted_skills_list
            )
        
        # Nettoyer le fichier temporaire
        Path(tmp_path).unlink()
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return CVUploadResponse(
            classification_id=classification_id,
            filename=file.filename,
            predicted_category=predicted_category,
            confidence=confidence,
            extraction_method=pdf_result.extraction_method,
            extraction_confidence=pdf_result.confidence,
            skills_summary=skills_summary,
            experience_info=experience_info,
            job_recommendations=job_recommendations,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        # Nettoyer en cas d'erreur
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )

# ============================================
# ENDPOINTS AVANCÉS - SKILLS ANALYSIS
# ============================================

@app.post("/analyze-skills", response_model=SkillsAnalysisResponse, tags=["Advanced - Skills"])
async def analyze_skills_from_text(
    cv: CVInput,
    recommend_jobs: bool = Query(True, description="Recommander des postes")
):
    """Analyser les compétences depuis un texte brut"""
    if not skills_detector:
        raise HTTPException(
            status_code=503,
            detail="Skills Detector non disponible"
        )
    
    try:
        # Extraire les compétences
        skills = skills_detector.extract_skills(cv.resume_text)
        
        # Analyser l'expérience
        experience = skills_detector.analyze_experience(cv.resume_text)
        
        # Recommandations
        recommendations = []
        if recommend_jobs:
            recommendations = skills_detector.recommend_jobs(
                skills,
                experience,
                top_n=5
            )
        
        # Top strengths
        top_strengths = skills_detector._identify_top_strengths(skills)
        
        return SkillsAnalysisResponse(
            skills_summary={
                'total_technical_skills': len(skills['technical_skills']),
                'total_soft_skills': len(skills['soft_skills']),
                'total_frameworks': len(skills['frameworks']),
                'total_tools': len(skills['tools']),
                'total_languages': len(skills['languages'])
            },
            detailed_skills=skills,
            experience_analysis=experience,
            job_recommendations=recommendations,
            top_strengths=top_strengths
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )

# ============================================
# ENDPOINTS AVANCÉS - HISTORY
# ============================================

@app.get("/history", tags=["Advanced - History"])
async def get_classification_history(
    limit: int = Query(10, ge=1, le=100, description="Nombre de résultats"),
    category: Optional[str] = Query(None, description="Filtrer par catégorie"),
    start_date: Optional[str] = Query(None, description="Date de début (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Date de fin (YYYY-MM-DD)")
):
    """Récupérer l'historique des classifications"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        if category:
            results = db_manager.get_classifications_by_category(category)[:limit]
        elif start_date and end_date:
            results = db_manager.get_classifications_by_date_range(start_date, end_date)
        else:
            results = db_manager.get_recent_classifications(limit)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.put("/history/{classification_id}/feedback", tags=["Advanced - History"])
async def update_classification_feedback(
    classification_id: int,
    feedback: FeedbackUpdate
):
    """Mettre à jour le feedback utilisateur pour une classification"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        db_manager.update_feedback(
            classification_id,
            feedback.user_feedback,
            feedback.correct_category,
            feedback.notes
        )
        
        return {
            "status": "success",
            "message": "Feedback mis à jour",
            "classification_id": classification_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.get("/statistics", response_model=StatisticsResponse, tags=["Advanced - Statistics"])
async def get_statistics():
    """Obtenir les statistiques globales"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        stats = db_manager.get_statistics()
        
        return StatisticsResponse(
            total_classifications=stats['total_classifications'],
            avg_confidence=stats['avg_confidence'],
            category_distribution=stats['category_distribution'],
            top_skills=stats.get('top_skills'),
            accuracy_from_feedback=stats.get('accuracy_from_feedback')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.post("/export", tags=["Advanced - Export"])
async def export_history_to_csv(
    include_skills: bool = Query(False, description="Inclure les compétences"),
    output_filename: str = Query("cv_history_export.csv", description="Nom du fichier")
):
    """Exporter l'historique vers un fichier CSV"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        output_dir = BASE_DIR / "outputs" / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        db_manager.export_to_csv(str(output_path), include_skills)
        
        return {
            "status": "success",
            "message": "Export réussi",
            "file_path": str(output_path)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.get("/search-skill/{skill_name}", tags=["Advanced - Search"])
async def search_by_skill(skill_name: str):
    """Rechercher tous les CV contenant une compétence spécifique"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        results = db_manager.search_by_skill(skill_name)
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

# ============================================
# LANCEMENT DU SERVEUR
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print(" DÉMARRAGE DU SERVEUR API")
    print("="*80)
    print("\n L'API sera accessible sur:")
    print("   - http://localhost:8000")
    print("   - http://localhost:8000/docs (Documentation interactive)")
    print("   - http://localhost:8000/redoc (Documentation alternative)")
    print("\n Fonctionnalités disponibles:")
    print(f"   - Classification ML: {'Bon' if all([MODEL, VECTORIZER, LABEL_ENCODER]) else 'Mauvais'}")
    print(f"   - Extraction PDF: {'Bon' if pdf_extractor else 'Mauvais'}")
    print(f"   - Détection compétences: {'Bon' if skills_detector else 'Mauvais'}")
    print(f"   - Base de données: {'Bon' if db_manager else 'Mauvais'}")
    print("\n  Pour arrêter: Ctrl+C")
    print("="*80 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
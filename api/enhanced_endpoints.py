"""
api/enhanced_endpoints.py
Endpoints API améliorés avec extraction PDF, détection de compétences et historique
À ajouter à votre api/main.py existant
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import sys
from pathlib import Path
import tempfile
import shutil

# Ajouter les modules au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import CVDatabaseManager
from src.pdf_processing.pdf_extractor import AdvancedPDFExtractor
from src.skills_extraction.skills_detector import SkillsDetector


# ============================================
# MODÈLES PYDANTIC
# ============================================

class CVUploadResponse(BaseModel):
    """Réponse pour l'upload et le traitement d'un CV"""
    classification_id: int
    filename: str
    predicted_category: str
    confidence: float
    extraction_method: str
    extraction_confidence: float
    skills_summary: Dict
    experience_info: Dict
    job_recommendations: List[Dict]
    processing_time_ms: int

class SkillsAnalysisResponse(BaseModel):
    """Réponse pour l'analyse de compétences"""
    skills_summary: Dict
    detailed_skills: Dict
    experience_analysis: Dict
    job_recommendations: List[Dict]
    top_strengths: List[str]

class HistoryQuery(BaseModel):
    """Requête pour l'historique"""
    limit: Optional[int] = 10
    category: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class FeedbackUpdate(BaseModel):
    """Mise à jour du feedback utilisateur"""
    user_feedback: str = Field(..., description="'Correct' ou 'Incorrect'")
    correct_category: Optional[str] = None
    notes: Optional[str] = None

class StatisticsResponse(BaseModel):
    """Réponse pour les statistiques"""
    total_classifications: int
    avg_confidence: float
    category_distribution: List[Dict]
    top_skills: List[Dict]
    accuracy_from_feedback: Optional[float]


# ============================================
# INITIALISATION DES COMPOSANTS
# ============================================

# Initialiser les gestionnaires (à faire au démarrage de l'API)
db_manager = None
pdf_extractor = None
skills_detector = None

def initialize_enhanced_components():
    """Initialiser tous les composants améliorés"""
    global db_manager, pdf_extractor, skills_detector
    
    print("\n" + "="*70)
    print(" INITIALISATION DES COMPOSANTS AMÉLIORÉS")
    print("="*70)
    
    # Base de données
    try:
        db_manager = CVDatabaseManager("data/cv_history.db")
        print(" Base de données initialisée")
    except Exception as e:
        print(f" Erreur base de données: {e}")
    
    # Extracteur PDF
    try:
        pdf_extractor = AdvancedPDFExtractor()
        print(" Extracteur PDF initialisé")
    except Exception as e:
        print(f" Erreur extracteur PDF: {e}")
    
    # Détecteur de compétences
    try:
        skills_detector = SkillsDetector()
        print(" Détecteur de compétences initialisé")
    except Exception as e:
        print(f" Erreur détecteur de compétences: {e}")
    
    print("="*70 + "\n")


# ============================================
# NOUVEAUX ENDPOINTS
# ============================================

async def upload_and_classify_cv(
    file: UploadFile = File(...),
    extract_skills: bool = Query(True, description="Extraire les compétences"),
    recommend_jobs: bool = Query(True, description="Recommander des postes"),
    save_to_history: bool = Query(True, description="Sauvegarder dans l'historique")
) -> CVUploadResponse:
    """
    Upload et classification complète d'un CV PDF
    
    - Extrait le texte du PDF (avec OCR si nécessaire)
    - Classifie le CV
    - Extrait les compétences
    - Recommande des postes
    - Sauvegarde dans l'historique
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers PDF sont acceptés"
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
        
        # 2. Classifier le CV (utilisez votre logique existante)
        # NOTE: Remplacez ceci par votre fonction de classification
        from your_classification_module import classify_cv  # À adapter
        
        classification_result = classify_cv(pdf_result.text)
        predicted_category = classification_result['category']
        confidence = classification_result['confidence']
        all_probabilities = classification_result.get('all_probabilities')
        
        # 3. Extraire les compétences si demandé
        skills_summary = {}
        experience_info = {}
        job_recommendations = []
        
        if extract_skills:
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
            if extract_skills:
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
                model_used="YourModelName",  # À adapter
                model_version="1.0",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                extracted_skills=extracted_skills_list
            )
        
        # Nettoyer le fichier temporaire
        Path(tmp_path).unlink()
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return CVUploadResponse(
            classification_id=classification_id or 0,
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


async def analyze_skills_from_text(
    text: str,
    recommend_jobs: bool = Query(True)
) -> SkillsAnalysisResponse:
    """
    Analyser les compétences depuis un texte brut
    """
    try:
        # Extraire les compétences
        skills = skills_detector.extract_skills(text)
        
        # Analyser l'expérience
        experience = skills_detector.analyze_experience(text)
        
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


async def get_classification_history(
    limit: int = Query(10, ge=1, le=100),
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict]:
    """
    Récupérer l'historique des classifications
    """
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        if category:
            results = db_manager.get_classifications_by_category(category)[:limit]
        elif start_date and end_date:
            results = db_manager.get_classifications_by_date_range(
                start_date,
                end_date
            )
        else:
            results = db_manager.get_recent_classifications(limit)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération: {str(e)}"
        )


async def update_classification_feedback(
    classification_id: int,
    feedback: FeedbackUpdate
) -> Dict:
    """
    Mettre à jour le feedback utilisateur pour une classification
    """
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
            detail=f"Erreur lors de la mise à jour: {str(e)}"
        )


async def get_statistics() -> StatisticsResponse:
    """
    Obtenir les statistiques globales
    """
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
            top_skills=stats['top_skills'],
            accuracy_from_feedback=stats.get('accuracy_from_feedback')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des stats: {str(e)}"
        )


async def export_history_to_csv(
    include_skills: bool = Query(False),
    output_filename: str = Query("cv_history_export.csv")
) -> Dict:
    """
    Exporter l'historique vers un fichier CSV
    """
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        output_path = f"outputs/exports/{output_filename}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        db_manager.export_to_csv(output_path, include_skills)
        
        return {
            "status": "success",
            "message": "Export réussi",
            "file_path": output_path
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'export: {str(e)}"
        )


async def search_by_skill(skill_name: str) -> List[Dict]:
    """
    Rechercher tous les CV contenant une compétence spécifique
    """
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
            detail=f"Erreur lors de la recherche: {str(e)}"
        )


# ============================================
# ENREGISTREMENT DES ROUTES
# ============================================

def register_enhanced_routes(app: FastAPI):
    """
    Enregistrer tous les nouveaux endpoints dans l'app FastAPI
    
    Usage:
        from api.enhanced_endpoints import register_enhanced_routes
        register_enhanced_routes(app)
    """
    
    # Upload et classification complète
    app.post(
        "/upload-cv",
        response_model=CVUploadResponse,
        tags=["CV Processing"],
        summary="Upload et classification complète d'un CV PDF"
    )(upload_and_classify_cv)
    
    # Analyse de compétences
    app.post(
        "/analyze-skills",
        response_model=SkillsAnalysisResponse,
        tags=["Skills Analysis"],
        summary="Analyser les compétences depuis un texte"
    )(analyze_skills_from_text)
    
    # Historique
    app.get(
        "/history",
        response_model=List[Dict],
        tags=["History"],
        summary="Récupérer l'historique des classifications"
    )(get_classification_history)
    
    # Feedback
    app.put(
        "/history/{classification_id}/feedback",
        response_model=Dict,
        tags=["History"],
        summary="Mettre à jour le feedback utilisateur"
    )(update_classification_feedback)
    
    # Statistiques
    app.get(
        "/statistics",
        response_model=StatisticsResponse,
        tags=["Statistics"],
        summary="Obtenir les statistiques globales"
    )(get_statistics)
    
    # Export
    app.post(
        "/export",
        response_model=Dict,
        tags=["Export"],
        summary="Exporter l'historique vers CSV"
    )(export_history_to_csv)
    
    # Recherche par compétence
    app.get(
        "/search-skill/{skill_name}",
        response_model=List[Dict],
        tags=["Search"],
        summary="Rechercher les CV par compétence"
    )(search_by_skill)
    
    print(" Tous les endpoints améliorés enregistrés")


# ============================================
# EXEMPLE D'UTILISATION DANS MAIN.PY
# ============================================

"""
Dans votre api/main.py, ajoutez:

from api.enhanced_endpoints import initialize_enhanced_components, register_enhanced_routes

# Au démarrage de l'app
@app.on_event("startup")
async def startup_event():
    initialize_enhanced_components()
    register_enhanced_routes(app)

Puis l'API aura automatiquement tous les nouveaux endpoints !
"""

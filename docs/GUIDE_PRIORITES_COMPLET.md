#  GUIDE COMPLET - FONCTIONNALITÉS PRIORITAIRES

##  TABLE DES MATIÈRES
1. [Installation](#installation)
2. [Base de Données - Historique](#base-de-données)
3. [Extraction PDF Améliorée](#extraction-pdf)
4. [Détection de Compétences](#détection-de-compétences)
5. [Intégration API](#intégration-api)
6. [Exemples d'Utilisation](#exemples)

---

## I INSTALLATION

### Dépendances Requises

```bash
# Base de données (inclus avec Python)
# sqlite3 est déjà inclus

# Extraction PDF
pip install pdfplumber pypdf pytesseract pdf2image

# OCR (Tesseract) - Installation système requise
# Windows: Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract

# Pour les images PDF
pip install Pillow poppler-utils

# Packages supplémentaires
pip install pandas openpyxl
```

### Structure du Projet Mise à Jour

```
Projet_NLPfinal/
├── api/
│   ├── main.py                    # API principale
│   └── enhanced_endpoints.py      # Nouveaux endpoints (NOUVEAU)
├── src/
│   ├── database/
│   │   └── db_manager.py          # Gestionnaire DB (NOUVEAU)
│   ├── pdf_processing/
│   │   └── pdf_extractor.py       # Extracteur PDF (NOUVEAU)
│   ├── skills_extraction/
│   │   └── skills_detector.py     # Détecteur compétences (NOUVEAU)
│   └── preprocessing/
│       └── text_cleaner.py        # Existant
├── data/
│   ├── cv_history.db              # Base de données (NOUVEAU)
│   └── ...
├── outputs/
│   ├── reports/                   # Rapports générés
│   └── exports/                   # Exports CSV
└── models_saved/
    └── ...
```

---

## II BASE DE DONNÉES - HISTORIQUE

### A. Fonctionnalités

- **Stockage complet** de toutes les classifications
- **Historique** avec date, heure, modèle utilisé
- **Feedback utilisateur** (Correct/Incorrect)
- **Statistiques** automatiques (quotidiennes et globales)
- **Export** vers CSV/Excel
- **Recherche** par compétences, catégories, dates

### B. Utilisation de Base

```python
from src.database.db_manager import CVDatabaseManager

# Créer le gestionnaire
db = CVDatabaseManager("data/cv_history.db")

# Ajouter une classification
classification_id = db.add_classification(
    cv_text="Experienced Python developer...",
    predicted_category="Data Scientist",
    confidence_score=0.95,
    cv_filename="john_doe_cv.pdf",
    all_probabilities={
        "Data Scientist": 0.95,
        "ML Engineer": 0.03
    },
    model_used="Random_Forest",
    model_version="1.0",
    processing_time_ms=150,
    extracted_skills=[
        {"name": "Python", "category": "Programming", "confidence": 0.98},
        {"name": "TensorFlow", "category": "Framework", "confidence": 0.90}
    ]
)

# Récupérer les stats
stats = db.get_statistics()
print(f"Total classifications: {stats['total_classifications']}")
print(f"Confiance moyenne: {stats['avg_confidence']:.2%}")

# Exporter vers CSV
db.export_to_csv("outputs/exports/history.csv", include_skills=True)

# Exporter un rapport
db.export_statistics_report("outputs/reports/statistics.txt")

# Rechercher par compétence
results = db.search_by_skill("Python")
print(f"Trouvé {len(results)} CV avec Python")

# Fermer la connexion
db.close()
```

### C. API Endpoints

```bash
# Récupérer l'historique (10 dernières classifications)
GET /history?limit=10

# Récupérer par catégorie
GET /history?category=Data+Scientist

# Récupérer par période
GET /history?start_date=2024-01-01&end_date=2024-12-31

# Statistiques globales
GET /statistics

# Export CSV
POST /export?include_skills=true

# Recherche par compétence
GET /search-skill/Python
```

### D. Schéma de Base de Données

**Table: classifications**
- id (PRIMARY KEY)
- cv_filename
- cv_text
- cv_text_preview (premiers 500 chars)
- predicted_category
- confidence_score
- all_probabilities (JSON)
- classification_date
- model_used
- model_version
- processing_time_ms
- user_feedback
- correct_category
- notes

**Table: extracted_skills**
- id (PRIMARY KEY)
- classification_id (FOREIGN KEY)
- skill_name
- skill_category
- confidence

**Table: daily_stats**
- date (PRIMARY KEY)
- total_classifications
- unique_categories
- avg_confidence
- most_common_category

---

## III EXTRACTION PDF AMÉLIORÉE

### A. Fonctionnalités

- **Détection automatique** du type de PDF (natif ou scanné)
- **Extraction multi-méthodes** (pdfplumber, pypdf, OCR)
- **OCR automatique** pour les PDFs scannés
- **Extraction de tableaux** depuis les PDFs
- **Extraction de métadonnées**
- **Extraction d'informations de contact**
- **Détection de sections** (Expérience, Éducation, Compétences)

### B. Utilisation de Base

```python
from src.pdf_processing.pdf_extractor import AdvancedPDFExtractor

# Créer l'extracteur
extractor = AdvancedPDFExtractor()

# Extraction automatique (détecte la meilleure méthode)
result = extractor.extract_from_pdf(
    "data/uploads/cv.pdf",
    method='auto',           # 'auto', 'pdfplumber', 'pypdf', 'ocr'
    extract_tables=True,
    ocr_lang='eng'          # 'eng', 'fra', etc.
)

# Résultats
print(f"Méthode utilisée: {result.extraction_method}")
print(f"PDF scanné: {result.is_scanned}")
print(f"Confiance: {result.confidence:.2%}")
print(f"Nombre de pages: {len(result.pages)}")
print(f"Texte extrait: {result.text[:200]}...")

# Extraire les informations de contact
contact = extractor.extract_contact_info(result.text)
print(f"Email: {contact['email']}")
print(f"Téléphone: {contact['phone']}")
print(f"LinkedIn: {contact['linkedin']}")

# Extraire les sections
sections = extractor.extract_sections(result.text)
for section_name in sections.keys():
    print(f"- {section_name}")
    print(f"  {sections[section_name][:100]}...")

# Traitement en batch
pdf_files = ["cv1.pdf", "cv2.pdf", "cv3.pdf"]
results = extractor.batch_extract(
    pdf_files,
    output_dir="data/extracted_texts"
)
```

### C. API Endpoint

```bash
# Upload d'un CV PDF avec extraction complète
POST /upload-cv

# Paramètres:
# - file: Le fichier PDF
# - extract_skills: true/false (extraire les compétences)
# - recommend_jobs: true/false (recommander des postes)
# - save_to_history: true/false (sauvegarder dans la DB)
```

Exemple avec curl:
```bash
curl -X POST "http://localhost:8000/upload-cv?extract_skills=true&recommend_jobs=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@cv.pdf"
```

### D. Formats Supportés

| Type de PDF | Méthode | Qualité |
|-------------|---------|---------|
| PDF natif (texte) | pdfplumber | Excellente |
| PDF natif simple | pypdf | Très bonne |
| PDF scanné (image) | OCR | Bonne* |
| PDF avec tableaux | pdfplumber | Excellente |
| PDF multi-colonnes | pdfplumber | Bonne |

*Dépend de la qualité du scan original

---

## IV DÉTECTION DE COMPÉTENCES

### A. Fonctionnalités

- **Détection automatique** de 1000+ compétences techniques
- **Extraction de soft skills**
- **Identification de frameworks et outils**
- **Langages de programmation**
- **Certifications**
- **Analyse de l'expérience** (années, niveau)
- **Recommandations de postes** basées sur les compétences
- **Score de correspondance** pour chaque poste
- **Identification des lacunes** de compétences

### B. Compétences Détectées

**Catégories Techniques:**
- Machine Learning (deep learning, NLP, computer vision, etc.)
- Data Science (statistics, analytics, visualization, etc.)
- Programming (Python, Java, JavaScript, etc.)
- Web Development (frontend, backend, full-stack, etc.)
- DevOps (CI/CD, containerization, cloud, etc.)
- Databases (SQL, NoSQL, design, etc.)
- Cloud (AWS, Azure, GCP, etc.)
- Security (cybersecurity, encryption, etc.)

**Frameworks:**
- ML/DL: TensorFlow, PyTorch, scikit-learn
- Web: React, Angular, Vue, Django, Flask
- Mobile: React Native, Flutter
- Testing: pytest, Jest, Selenium

**Outils:**
- Version Control: Git, GitHub, GitLab
- Containers: Docker, Kubernetes
- CI/CD: Jenkins, GitHub Actions
- Monitoring: Prometheus, Grafana
- Infrastructure: Terraform, Ansible

### C. Utilisation de Base

```python
from src.skills_extraction.skills_detector import SkillsDetector

# Créer le détecteur
detector = SkillsDetector()

# Exemple de texte de CV
cv_text = """
Senior Data Scientist with 7 years of experience...
SKILLS: Python, TensorFlow, PyTorch, AWS...
"""

# Extraire les compétences
skills = detector.extract_skills(cv_text)

print(f"Compétences techniques: {len(skills['technical_skills'])}")
print(f"Soft skills: {len(skills['soft_skills'])}")
print(f"Frameworks: {len(skills['frameworks'])}")

# Analyser l'expérience
experience = detector.analyze_experience(cv_text)
print(f"Années d'expérience: {experience['estimated_experience_years']}")
print(f"Niveau: {experience['experience_level']}")

# Recommander des postes
recommendations = detector.recommend_jobs(skills, experience, top_n=5)
for rec in recommendations:
    print(f"\n{rec['job_title']}")
    print(f"  Match: {rec['match_score']:.0%}")
    print(f"  Fit expérience: {rec['experience_level_fit']}")
    print(f"  Compétences manquantes: {rec['missing_required_skills'][:3]}")

# Générer un rapport complet
report = detector.generate_skills_report(
    cv_text,
    output_path="outputs/reports/skills_analysis.txt"
)
```

### D. API Endpoint

```bash
# Analyser les compétences depuis un texte
POST /analyze-skills

# Body:
{
  "text": "Texte du CV...",
  "recommend_jobs": true
}
```

### E. Personnalisation

Créez un fichier JSON personnalisé:

```json
{
  "technical_skills": {
    "Custom Category": ["skill1", "skill2", "skill3"]
  },
  "job_skill_mapping": {
    "Custom Job Title": {
      "required": ["skill1", "skill2"],
      "preferred": ["skill3", "skill4"],
      "soft_skills": ["communication", "leadership"]
    }
  }
}
```

Puis chargez-le:

```python
detector = SkillsDetector("path/to/custom_skills.json")
```

---

## V INTÉGRATION API

### A. Mise à Jour de main.py

Ajoutez à votre `api/main.py`:

```python
from api.enhanced_endpoints import initialize_enhanced_components, register_enhanced_routes

# Au démarrage de l'app
@app.on_event("startup")
async def startup_event():
    # Initialiser les composants
    initialize_enhanced_components()
    
    # Enregistrer les routes
    register_enhanced_routes(app)
```

### B. Nouveaux Endpoints Disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| /upload-cv | POST | Upload et analyse complète PDF |
| /analyze-skills | POST | Analyse de compétences |
| /history | GET | Historique des classifications |
| /history/{id}/feedback | PUT | Mise à jour feedback |
| /statistics | GET | Statistiques globales |
| /export | POST | Export CSV |
| /search-skill/{skill} | GET | Recherche par compétence |

### C. Documentation Interactive

Une fois l'API lancée:
```
http://localhost:8000/docs
```

---

## VI EXEMPLES D'UTILISATION

### Exemple 1: Upload et Classification Complète

```python
import requests

url = "http://localhost:8000/upload-cv"

# Upload du fichier
with open("cv_john_doe.pdf", "rb") as f:
    files = {"file": f}
    params = {
        "extract_skills": True,
        "recommend_jobs": True,
        "save_to_history": True
    }
    
    response = requests.post(url, files=files, params=params)

result = response.json()

print(f"Catégorie: {result['predicted_category']}")
print(f"Confiance: {result['confidence']:.2%}")
print(f"Compétences techniques: {result['skills_summary']['total_technical_skills']}")

print("\nTop 3 recommandations:")
for i, rec in enumerate(result['job_recommendations'][:3], 1):
    print(f"{i}. {rec['job_title']} - {rec['match_score']:.0%}")
```

### Exemple 2: Analyse de Compétences Uniquement

```python
url = "http://localhost:8000/analyze-skills"

data = {
    "text": """
    Software Engineer with 5 years of experience.
    Skills: Python, JavaScript, React, Node.js, Docker, AWS
    """,
    "recommend_jobs": True
}

response = requests.post(url, json=data)
result = response.json()

print("Compétences détectées:")
for skill in result['detailed_skills']['technical_skills'][:5]:
    print(f"  - {skill['skill']} ({skill['category']})")
```

### Exemple 3: Consultation de l'Historique

```python
# Obtenir les 20 dernières classifications
response = requests.get("http://localhost:8000/history?limit=20")
history = response.json()

for item in history:
    print(f"{item['classification_date']}: {item['predicted_category']} ({item['confidence']:.2%})")
```

### Exemple 4: Statistiques et Rapports

```python
# Statistiques
response = requests.get("http://localhost:8000/statistics")
stats = response.json()

print(f"Total: {stats['total_classifications']}")
print(f"Confiance moyenne: {stats['avg_confidence']:.2%}")

print("\nTop catégories:")
for cat in stats['category_distribution'][:5]:
    print(f"  {cat['category']}: {cat['count']}")

print("\nTop compétences:")
for skill in stats['top_skills'][:5]:
    print(f"  {skill['skill']}: {skill['count']}")

# Export CSV
response = requests.post(
    "http://localhost:8000/export",
    params={"include_skills": True}
)
print(f"Export: {response.json()['file_path']}")
```

---

##  WORKFLOW COMPLET

```
1. Upload CV PDF
   ↓
2. Extraction automatique du texte
   ├─ PDF natif → pdfplumber
   └─ PDF scanné → OCR
   ↓
3. Classification du CV
   ↓
4. Extraction des compétences
   ├─ Compétences techniques
   ├─ Soft skills
   ├─ Frameworks & outils
   └─ Langages
   ↓
5. Analyse de l'expérience
   ├─ Années d'expérience
   └─ Niveau (Junior/Mid/Senior)
   ↓
6. Recommandations de postes
   ├─ Score de correspondance
   └─ Compétences manquantes
   ↓
7. Sauvegarde dans l'historique
   ├─ Base de données SQLite
   └─ Statistiques automatiques
```

---

##  AVANTAGES

### Base de Données
-  Traçabilité complète
-  Amélioration continue (feedback)
-  Statistiques en temps réel
-  Export facile pour analyse

### Extraction PDF
-  Support universel (natif + scanné)
-  Haute qualité d'extraction
-  Métadonnées enrichies
-  Détection de sections

### Détection de Compétences
-  1000+ compétences reconnues
-  Recommandations intelligentes
-  Analyse de l'expérience
-  Identification des lacunes

---

##  DÉPANNAGE

### Problème: OCR ne fonctionne pas

```bash
# Vérifier Tesseract
tesseract --version

# Windows: Ajouter au PATH
# Chemin: C:\Program Files\Tesseract-OCR

# Réinstaller
pip uninstall pytesseract pdf2image
pip install pytesseract pdf2image
```

### Problème: Base de données verrouillée

```python
# Fermer proprement les connexions
with CVDatabaseManager("data/cv_history.db") as db:
    # Vos opérations
    pass  # Auto-fermeture
```

### Problème: Extraction PDF lente

```python
# Utiliser pypdf pour des PDFs simples (plus rapide)
result = extractor.extract_from_pdf(
    "cv.pdf",
    method='pypdf'  # Plus rapide que pdfplumber
)
```

---

##  RESSOURCES

- Documentation pdfplumber: https://github.com/jsvine/pdfplumber
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- FastAPI: https://fastapi.tiangolo.com/
- SQLite: https://www.sqlite.org/docs.html

---

**Projet prêt à l'emploi ! **

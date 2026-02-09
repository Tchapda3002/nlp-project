#  GUIDE DE DÉPANNAGE RAPIDE - ERREUR 503

## LE PROBLÈME
```
Status Code: 503
"Composants manquants: Modèle ML, Vectorizer, Label Encoder..."
```

##  SOLUTION EN 5 ÉTAPES

### ÉTAPE 1: Exécuter le diagnostic
```bash
cd C:\Users\Easy Services Pro\Projet_NLPfinal
python diagnostic_api.py
```

Ce script va:
-  Vérifier que vos fichiers existent
-  Tester leur chargement
-  Identifier le problème exact
-  Créer un fichier de configuration

### ÉTAPE 2: Remplacer main.py

**Remplacez votre `api/main.py` par `main_v2.py`**

Pourquoi ? Le nouveau fichier:
-  Utilise des chemins absolus (fonctionne peu importe d'où vous lancez l'API)
-  Affiche des logs détaillés au démarrage
-  Teste chaque composant individuellement
-  A un fallback si TextCleaner n'est pas disponible

### ÉTAPE 3: Lancer l'API DEPUIS LE TERMINAL (PAS JUPYTER!)

** NE PAS FAIRE:**
```python
# Dans Jupyter Notebook
!cd api
!python main.py
```

** FAIRE:**

**Option A - Windows CMD:**
```cmd
cd C:\Users\Easy Services Pro\Projet_NLPfinal
cd api
python main.py
```

**Option B - PowerShell:**
```powershell
cd "C:\Users\Easy Services Pro\Projet_NLPfinal"
cd api
python main.py
```

**Option C - Git Bash:**
```bash
cd /c/Users/Easy\ Services\ Pro/Projet_NLPfinal
cd api
python main.py
```

### ÉTAPE 4: Vérifier les logs

Vous devriez voir:
```
============================================================
 CHARGEMENT DES MODÈLES
============================================================

 Chargement du modèle ML...
 best_model.pkl chargé avec succès

 Chargement du vectorizer...
 tfidf_vectorizer.pkl chargé avec succès

 Chargement du label encoder...
 label_encoder.pkl chargé avec succès
   Catégories: 25

 Initialisation du text cleaner...
 TextCleaner initialisé

============================================================
  TOUS LES COMPOSANTS CHARGÉS AVEC SUCCÈS!
============================================================
```

### ÉTAPE 5: Tester l'API

Ouvrez votre navigateur:
```
http://localhost:8000/health
```

Ou en Python:
```python
import requests

# Test de santé
response = requests.get("http://localhost:8000/health")
print(response.json())

# Test de prédiction
data = {
    "resume_text": "Experienced Python developer with machine learning skills..."
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

---

##  DIAGNOSTICS AVANCÉS

### Si les fichiers ne se chargent toujours pas:

#### 1. Vérifier les permissions
```bash
# Les fichiers sont-ils accessibles en lecture?
ls -l models_saved/
```

#### 2. Vérifier l'intégrité des fichiers
```python
import pickle
from pathlib import Path

models_dir = Path("models_saved")

# Tester chaque fichier
for file in ["best_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl"]:
    filepath = models_dir / file
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f" {file} OK")
    except Exception as e:
        print(f" {file} ERREUR: {e}")
```

#### 3. Recréer les modèles
Si les fichiers sont corrompus:
```bash
cd notebooks
jupyter notebook 04_modeling.ipynb
# Exécutez TOUTES les cellules
```

---

##  PROBLÈMES COURANTS

### Problème: "ModuleNotFoundError: No module named 'preprocessing'"

**Cause:** Le dossier `src/` n'est pas dans le Python path

**Solution:**
```python
# Ajoutez au début de main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### Problème: "FileNotFoundError" même si les fichiers existent

**Cause:** Vous lancez l'API depuis le mauvais dossier

**Solution:** Toujours lancer depuis le dossier `api/`:
```bash
cd C:\Users\Easy Services Pro\Projet_NLPfinal\api
python main.py
```

### Problème: TextCleaner ne fonctionne pas

**Solution temporaire:** Le nouveau main.py a un nettoyage basique en fallback

**Solution permanente:** Vérifiez que `src/preprocessing/text_cleaner.py` existe

---

##  CHECKLIST DE VÉRIFICATION

Avant de lancer l'API, vérifiez:

- [ ]  Je suis dans le bon dossier (`api/`)
- [ ]  Les fichiers existent dans `models_saved/`:
  - [ ] `best_model.pkl` (ou `Random_Forest_model.pkl`)
  - [ ] `tfidf_vectorizer.pkl`
  - [ ] `label_encoder.pkl`
- [ ]  Le fichier `src/preprocessing/text_cleaner.py` existe
- [ ]  J'utilise le nouveau `main.py` (ou `main_v2.py`)
- [ ]  Je lance depuis le TERMINAL (pas Jupyter)
- [ ]  Aucun autre processus n'utilise le port 8000

---

##  COMMANDES ESSENTIELLES

**Diagnostic complet:**
```bash
python diagnostic_api.py
```

**Lancer l'API:**
```bash
cd api
python main.py
```

**Tester l'API:**
```bash
python test_api.py
```

**Voir les processus sur le port 8000:**
```bash
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

**Tuer un processus:**
```bash
# Windows
taskkill /PID <numero_PID> /F

# Linux/Mac
kill <PID>
```

---

##  ASTUCE FINALE

Si RIEN ne fonctionne, créez un environnement virtuel propre:

```bash
# Créer un nouvel environnement
python -m venv venv_test

# Activer (Windows)
venv_test\Scripts\activate

# Installer les dépendances
pip install fastapi uvicorn scikit-learn pandas numpy

# Lancer l'API
cd api
python main.py
```

---

**Besoin d'aide?** Envoyez:
1. La sortie complète de `python diagnostic_api.py`
2. Les logs de démarrage de l'API
3. Une capture d'écran du dossier `models_saved/`

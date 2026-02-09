"""
diagnostic_api.py
Script de diagnostic approfondi pour identifier pourquoi l'API ne charge pas les modèles
"""

import sys
from pathlib import Path
import pickle

print("\n" + "="*80)
print(" DIAGNOSTIC APPROFONDI - PROBLÈME DE CHARGEMENT DES MODÈLES")
print("="*80 + "\n")

# 1. Vérifier le répertoire de travail actuel
print(" 1. RÉPERTOIRE DE TRAVAIL ACTUEL")
print("-" * 80)
current_dir = Path.cwd()
print(f"   Répertoire actuel: {current_dir}")
print()

# 2. Trouver le dossier du projet
print(" 2. STRUCTURE DU PROJET")
print("-" * 80)

# Essayer de trouver le dossier api
api_dir = current_dir / "api"
if api_dir.exists():
    print(f" Dossier api trouvé: {api_dir}")
    base_dir = current_dir
else:
    # Peut-être qu'on est dans le dossier api
    if current_dir.name == "api":
        print(f"  Vous êtes dans le dossier api")
        base_dir = current_dir.parent
        api_dir = current_dir
    else:
        print(f" Dossier api non trouvé")
        # Chercher dans le parent
        parent_api = current_dir.parent / "api"
        if parent_api.exists():
            print(f" Dossier api trouvé dans le parent: {parent_api}")
            base_dir = current_dir.parent
            api_dir = parent_api
        else:
            print(f" Structure de projet non standard")
            base_dir = current_dir

print(f"   Base du projet: {base_dir}")
print()

# 3. Vérifier le dossier models_saved
print(" 3. DOSSIER MODELS_SAVED")
print("-" * 80)

models_dir = base_dir / "models_saved"
print(f"   Chemin attendu: {models_dir}")
print(f"   Existe: {models_dir.exists()}")

if models_dir.exists():
    print(f"\n    Fichiers dans {models_dir}:")
    for file in sorted(models_dir.iterdir()):
        if file.is_file():
            size = file.stat().st_size / 1024
            print(f"       {file.name:40s} ({size:.1f} KB)")
else:
    print(f"    Le dossier n'existe pas!")
    # Chercher dans d'autres endroits
    print(f"\n    Recherche dans d'autres emplacements...")
    
    for potential_path in [
        current_dir / "models_saved",
        current_dir.parent / "models_saved",
        Path(__file__).parent / "models_saved",
    ]:
        if potential_path.exists():
            print(f"       Trouvé: {potential_path}")
            models_dir = potential_path
            break
print()

# 4. Essayer de charger chaque fichier individuellement
print(" 4. TEST DE CHARGEMENT DES FICHIERS")
print("-" * 80)

if not models_dir.exists():
    print(" Impossible de continuer - dossier models_saved non trouvé")
    sys.exit(1)

files_to_test = {
    "best_model.pkl": "Modèle ML principal",
    "tfidf_vectorizer.pkl": "Vectorizer TF-IDF",
    "label_encoder.pkl": "Encodeur de labels",
}

loaded_objects = {}

for filename, description in files_to_test.items():
    filepath = models_dir / filename
    print(f"\n    {filename} ({description})")
    print(f"      Chemin: {filepath}")
    
    if not filepath.exists():
        print(f"       Fichier n'existe pas")
        continue
    
    print(f"       Fichier existe")
    
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"       Chargement réussi")
        print(f"      Type: {type(obj).__name__}")
        loaded_objects[filename] = obj
        
        # Afficher des infos supplémentaires
        if filename == "tfidf_vectorizer.pkl":
            if hasattr(obj, 'max_features'):
                print(f"      Features: {obj.max_features}")
        elif filename == "label_encoder.pkl":
            if hasattr(obj, 'classes_'):
                print(f"      Catégories: {len(obj.classes_)}")
                print(f"      Exemples: {', '.join(obj.classes_[:3])}...")
        
    except Exception as e:
        print(f"       Erreur de chargement: {e}")

print()

# 5. Vérifier le module TextCleaner
print(" 5. MODULE TEXT_CLEANER")
print("-" * 80)

src_dir = base_dir / "src" / "preprocessing"
text_cleaner_path = src_dir / "text_cleaner.py"

print(f"   Chemin attendu: {text_cleaner_path}")
print(f"   Existe: {text_cleaner_path.exists()}")

if text_cleaner_path.exists():
    print(f"    Fichier trouvé")
    
    # Essayer d'importer
    sys.path.insert(0, str(base_dir / "src"))
    try:
        from preprocessing.text_cleaner import TextCleaner
        print(f"    Import réussi")
        
        # Tester l'initialisation
        cleaner = TextCleaner()
        print(f"    Initialisation réussie")
        
        # Tester le nettoyage
        test_text = "This is a TEST with 123 numbers!"
        cleaned = cleaner.clean_text(test_text)
        print(f"    Test de nettoyage réussi")
        print(f"      Original: {test_text}")
        print(f"      Nettoyé: {cleaned}")
        
    except Exception as e:
        print(f"    Erreur d'import/utilisation: {e}")
else:
    print(f"    Fichier non trouvé")

print()

# 6. Créer un fichier de configuration
print("  6. CRÉATION D'UN FICHIER DE CONFIGURATION")
print("-" * 80)

config_content = f"""# Configuration automatique générée par diagnostic_api.py

BASE_DIR = r"{base_dir}"
MODELS_DIR = r"{models_dir}"
API_DIR = r"{api_dir}"
SRC_DIR = r"{base_dir / 'src'}"

# À utiliser dans main.py:
# from pathlib import Path
# from config import BASE_DIR, MODELS_DIR, API_DIR, SRC_DIR
"""

config_path = api_dir / "config.py"
try:
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f" Fichier de configuration créé: {config_path}")
except Exception as e:
    print(f" Erreur: {e}")

print()

# 7. Tester une prédiction complète
print(" 7. TEST DE PRÉDICTION COMPLÈTE")
print("-" * 80)

if len(loaded_objects) == 3:
    print(" Tous les composants sont chargés - Test de prédiction...")
    
    try:
        from preprocessing.text_cleaner import TextCleaner
        
        model = loaded_objects["best_model.pkl"]
        vectorizer = loaded_objects["tfidf_vectorizer.pkl"]
        label_encoder = loaded_objects["label_encoder.pkl"]
        cleaner = TextCleaner()
        
        test_cv = """
        Experienced Python developer with 5 years in machine learning.
        Skills: Python, TensorFlow, scikit-learn, pandas, numpy.
        Built predictive models and data pipelines.
        """
        
        print(f"\n   CV de test: {test_cv[:100]}...")
        
        # Pipeline complet
        cleaned = cleaner.clean_text(test_cv)
        print(f"    Texte nettoyé")
        
        X = vectorizer.transform([cleaned])
        print(f"    Texte vectorisé: {X.shape}")
        
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        category = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities.max()
        
        print(f"\n    PRÉDICTION RÉUSSIE!")
        print(f"      Catégorie: {category}")
        print(f"      Confiance: {confidence:.4f}")
        
    except Exception as e:
        print(f"    Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f" Impossible de tester - tous les composants ne sont pas chargés")

print()

# 8. Recommandations
print(" 8. RECOMMANDATIONS")
print("-" * 80)

if len(loaded_objects) == 3 and text_cleaner_path.exists():
    print("  TOUS LES COMPOSANTS SONT FONCTIONNELS!")
    print()
    print(" SOLUTION AU PROBLÈME:")
    print("   Le problème vient probablement de la façon dont vous lancez l'API.")
    print()
    print("    NE PAS FAIRE:")
    print("      - Lancer l'API depuis Jupyter Notebook")
    print("      - Lancer depuis un mauvais répertoire")
    print()
    print("    À FAIRE:")
    print("      1. Ouvrir un terminal (CMD ou PowerShell)")
    print(f"      2. cd {base_dir}")
    print("      3. cd api")
    print("      4. python main.py")
    print()
    print("   OU utiliser le nouveau main.py corrigé que je vous ai fourni")
    print()
else:
    print("  Problèmes détectés:")
    if "best_model.pkl" not in loaded_objects:
        print("    Modèle ML non chargé")
    if "tfidf_vectorizer.pkl" not in loaded_objects:
        print("    Vectorizer non chargé")
    if "label_encoder.pkl" not in loaded_objects:
        print("    Label encoder non chargé")
    if not text_cleaner_path.exists():
        print("    TextCleaner non trouvé")
    print()
    print("    Actions nécessaires:")
    print("      1. Vérifiez que tous les notebooks ont été exécutés")
    print("      2. Vérifiez la structure du projet")

print()
print("="*80)
print(" DIAGNOSTIC TERMINÉ")
print("="*80)
print()
print(f" Chemins importants:")
print(f"   Base du projet: {base_dir}")
print(f"   Modèles: {models_dir}")
print(f"   API: {api_dir}")
print()

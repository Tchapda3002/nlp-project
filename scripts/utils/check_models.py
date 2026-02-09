"""
check_models.py
Script de diagnostic pour vérifier la présence de tous les fichiers nécessaires
À exécuter depuis le dossier racine du projet
"""

from pathlib import Path
import pickle
import sys

def check_structure():
    """Vérifier la structure du projet"""
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC DE LA STRUCTURE DU PROJET")
    print("="*70 + "\n")
    
    # Déterminer le répertoire du projet
    current_dir = Path.cwd()
    print(f"📁 Répertoire actuel: {current_dir}\n")
    
    # Vérifier les dossiers principaux
    folders_to_check = {
        "api": "Dossier de l'API",
        "models_saved": "Dossier des modèles sauvegardés",
        "src": "Dossier du code source",
        "data": "Dossier des données",
        "notebooks": "Dossier des notebooks"
    }
    
    print("📂 STRUCTURE DES DOSSIERS:")
    print("-" * 70)
    all_folders_ok = True
    for folder, description in folders_to_check.items():
        folder_path = current_dir / folder
        exists = folder_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {folder:20s} - {description}")
        if not exists:
            all_folders_ok = False
    
    print("\n" + "="*70)
    
    # Vérifier les fichiers de modèles
    print("\n🤖 FICHIERS DE MODÈLES:")
    print("-" * 70)
    
    models_dir = current_dir / "models_saved"
    
    if not models_dir.exists():
        print(f"❌ Le dossier {models_dir} n'existe pas!")
        print("   Vous devez d'abord exécuter le notebook 04_modeling.ipynb")
        return False
    
    # Fichiers essentiels
    essential_files = {
        "tfidf_vectorizer.pkl": "Vectorizer TF-IDF (créé par 03_feature_extraction.ipynb)",
        "label_encoder.pkl": "Encodeur de labels (créé par 04_modeling.ipynb)",
    }
    
    # Fichiers de modèles possibles
    model_files = [
        "best_model.pkl",
        "Random_Forest_model.pkl",
        "Logistic_Regression_model.pkl",
        "SVM_model.pkl",
        "Naive_Bayes_model.pkl"
    ]
    
    all_files_ok = True
    
    # Vérifier les fichiers essentiels
    for filename, description in essential_files.items():
        filepath = models_dir / filename
        exists = filepath.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {filename:30s} - {description}")
        if exists:
            size = filepath.stat().st_size / 1024  # Taille en KB
            print(f"     Taille: {size:.1f} KB")
        if not exists:
            all_files_ok = False
    
    # Vérifier au moins un fichier de modèle
    print(f"\n🎯 MODÈLES ML DISPONIBLES:")
    model_found = False
    for model_file in model_files:
        filepath = models_dir / model_file
        if filepath.exists():
            size = filepath.stat().st_size / 1024
            print(f"✅ {model_file:30s} - Taille: {size:.1f} KB")
            model_found = True
    
    if not model_found:
        print(f"❌ Aucun modèle ML trouvé!")
        print(f"   Noms recherchés: {', '.join(model_files)}")
        all_files_ok = False
    
    # Lister tous les fichiers présents
    print(f"\n📋 TOUS LES FICHIERS DANS {models_dir}:")
    print("-" * 70)
    if list(models_dir.iterdir()):
        for file in sorted(models_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size / 1024
                print(f"   - {file.name:40s} ({size:.1f} KB)")
    else:
        print("   (Dossier vide)")
    
    # Vérifier src/preprocessing/text_cleaner.py
    print(f"\n📄 FICHIERS DE CODE SOURCE:")
    print("-" * 70)
    
    text_cleaner_path = current_dir / "src" / "preprocessing" / "text_cleaner.py"
    exists = text_cleaner_path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} src/preprocessing/text_cleaner.py")
    if not exists:
        print("   ⚠️  Ce fichier est nécessaire pour le nettoyage du texte")
        all_files_ok = False
    
    # Vérifier api/main.py
    api_main_path = current_dir / "api" / "main.py"
    exists = api_main_path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} api/main.py")
    
    print("\n" + "="*70)
    
    # Résumé final
    print("\n📊 RÉSUMÉ:")
    print("-" * 70)
    
    if all_folders_ok and all_files_ok and model_found:
        print("✅ ✅ ✅  TOUT EST PRÊT!")
        print("\n📌 PROCHAINES ÉTAPES:")
        print("   1. Démarrer l'API:")
        print("      cd api")
        print("      python main.py")
        print("      OU")
        print("      uvicorn main:app --reload --port 8000")
        print("\n   2. Tester l'API:")
        print("      http://localhost:8000/docs")
        return True
    else:
        print("❌ ❌ ❌  DES FICHIERS SONT MANQUANTS!")
        print("\n📌 ACTIONS À EFFECTUER:")
        
        if not model_found:
            print("   1. Exécutez le notebook: notebooks/04_modeling.ipynb")
            print("      → Cela créera les fichiers de modèles")
        
        if not (models_dir / "tfidf_vectorizer.pkl").exists():
            print("   2. Exécutez le notebook: notebooks/03_feature_extraction.ipynb")
            print("      → Cela créera le vectorizer TF-IDF")
        
        if not text_cleaner_path.exists():
            print("   3. Vérifiez que le fichier src/preprocessing/text_cleaner.py existe")
            print("      → Créez-le si nécessaire")
        
        return False


def test_model_loading():
    """Tester le chargement des modèles"""
    print("\n" + "="*70)
    print("🧪 TEST DE CHARGEMENT DES MODÈLES")
    print("="*70 + "\n")
    
    current_dir = Path.cwd()
    models_dir = current_dir / "models_saved"
    
    # Tester le chargement du vectorizer
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    if vectorizer_path.exists():
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"✅ Vectorizer chargé avec succès")
            print(f"   Type: {type(vectorizer).__name__}")
            if hasattr(vectorizer, 'max_features'):
                print(f"   Max features: {vectorizer.max_features}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du vectorizer: {e}")
    
    # Tester le chargement du label encoder
    encoder_path = models_dir / "label_encoder.pkl"
    if encoder_path.exists():
        try:
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print(f"✅ Label Encoder chargé avec succès")
            print(f"   Nombre de catégories: {len(label_encoder.classes_)}")
            print(f"   Catégories: {', '.join(label_encoder.classes_[:5])}...")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du label encoder: {e}")
    
    # Tester le chargement d'un modèle
    model_files = ["best_model.pkl", "Random_Forest_model.pkl"]
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ Modèle {model_file} chargé avec succès")
                print(f"   Type: {type(model).__name__}")
                break
            except Exception as e:
                print(f"❌ Erreur lors du chargement du modèle {model_file}: {e}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔧 OUTIL DE DIAGNOSTIC - PROJET CV CLASSIFICATION")
    print("="*70)
    
    # Vérifier la structure
    structure_ok = check_structure()
    
    # Si la structure est OK, tester le chargement
    if structure_ok:
        test_model_loading()
    
    print("\n" + "="*70)
    print("✨ DIAGNOSTIC TERMINÉ")
    print("="*70 + "\n")

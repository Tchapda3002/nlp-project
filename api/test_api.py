"""
test_api.py
Script pour tester l'API de classification de CV
Peut être exécuté directement ou depuis un notebook
"""

import requests
import json
from typing import Dict, Any

# ============================================
# CONFIGURATION
# ============================================

API_URL = "http://localhost:8000"

# Exemples de CV pour différentes catégories
SAMPLE_CVS = {
    "Data Scientist": """
        Experienced Data Scientist with 5+ years in machine learning and AI.
        
        SKILLS:
        - Python, R, SQL
        - TensorFlow, PyTorch, Scikit-learn
        - Data visualization with Matplotlib, Seaborn
        - Statistical analysis and modeling
        - Deep learning, NLP, Computer Vision
        
        EXPERIENCE:
        - Built predictive models for customer churn
        - Developed recommendation systems
        - Led data science projects end-to-end
    """,
    
    "Web Developer": """
        Full-stack Web Developer with expertise in modern frameworks.
        
        SKILLS:
        - JavaScript, TypeScript, Python
        - React, Angular, Vue.js
        - Node.js, Express
        - HTML5, CSS3, Sass
        - MongoDB, PostgreSQL
        - RESTful APIs, GraphQL
        
        EXPERIENCE:
        - Built responsive web applications
        - Developed e-commerce platforms
        - Implemented user authentication systems
    """,
    
    "DevOps Engineer": """
        DevOps Engineer specializing in CI/CD and cloud infrastructure.
        
        SKILLS:
        - AWS, Azure, Google Cloud
        - Docker, Kubernetes
        - Jenkins, GitLab CI, GitHub Actions
        - Terraform, Ansible
        - Linux administration
        - Monitoring with Prometheus, Grafana
        
        EXPERIENCE:
        - Automated deployment pipelines
        - Managed cloud infrastructure
        - Implemented monitoring solutions
    """,
    
    "Short CV": "I am a developer.",  # CV trop court - devrait échouer
}


# ============================================
# FONCTIONS DE TEST
# ============================================

def test_health_check():
    """Test 1: Vérifier que l'API est en ligne"""
    print("\n" + "="*60)
    print("TEST 1: HEALTH CHECK")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print(" API en ligne!")
            print(f"\nStatut: {data.get('status')}")
            print(f"Modèle chargé: {data.get('model_loaded')}")
            print(f"Vectorizer chargé: {data.get('vectorizer_loaded')}")
            print(f"Label Encoder chargé: {data.get('label_encoder_loaded')}")
            print(f"Text Cleaner chargé: {data.get('text_cleaner_loaded')}")
            print(f"Version: {data.get('version')}")
            print(f"Dossier modèles: {data.get('models_directory')}")
            
            if data.get('status') == 'healthy':
                return True
            else:
                print("\n  API en mode dégradé - certains composants manquent")
                return False
        else:
            print(f" Erreur: Status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(" Impossible de se connecter à l'API")
        print(f"   Vérifiez que l'API est lancée sur {API_URL}")
        return False
    except Exception as e:
        print(f" Erreur: {e}")
        return False


def test_get_categories():
    """Test 2: Récupérer la liste des catégories"""
    print("\n" + "="*60)
    print("TEST 2: GET CATEGORIES")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/categories")
        
        if response.status_code == 200:
            data = response.json()
            print(f" {data.get('total_categories')} catégories disponibles:")
            for i, cat in enumerate(data.get('categories', []), 1):
                print(f"   {i}. {cat}")
            return True
        else:
            print(f" Erreur: Status {response.status_code}")
            print(f"   Réponse: {response.text}")
            return False
            
    except Exception as e:
        print(f" Erreur: {e}")
        return False


def test_model_info():
    """Test 3: Obtenir les infos du modèle"""
    print("\n" + "="*60)
    print("TEST 3: MODEL INFO")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/model-info")
        
        if response.status_code == 200:
            data = response.json()
            print(" Informations du modèle:")
            print(f"   Type: {data.get('model_type')}")
            print(f"   Nombre de features: {data.get('n_features')}")
            print(f"   Nombre de catégories: {data.get('n_categories')}")
            print(f"   Type de vectorizer: {data.get('vectorizer_type')}")
            return True
        else:
            print(f" Erreur: Status {response.status_code}")
            print(f"   Réponse: {response.text}")
            return False
            
    except Exception as e:
        print(f" Erreur: {e}")
        return False


def test_single_prediction(cv_text: str, expected_category: str = None):
    """Test 4: Prédiction simple"""
    print("\n" + "="*60)
    print(f"TEST 4: PRÉDICTION - {expected_category if expected_category else 'CV'}")
    print("="*60)
    
    try:
        # Requête sans probabilités
        payload = {
            "resume_text": cv_text
        }
        
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(" Prédiction réussie!")
            print(f"\n   Catégorie prédite: {data.get('category')}")
            print(f"   Confiance: {data.get('confidence'):.4f}")
            
            if expected_category:
                if data.get('category') == expected_category:
                    print(f"    Correspond à l'attendu: {expected_category}")
                else:
                    print(f"     Attendu: {expected_category}, Obtenu: {data.get('category')}")
            
            return True
        else:
            print(f" Erreur: Status {response.status_code}")
            print(f"   Réponse: {response.json()}")
            return False
            
    except Exception as e:
        print(f" Erreur: {e}")
        return False


def test_prediction_with_probabilities(cv_text: str):
    """Test 5: Prédiction avec toutes les probabilités"""
    print("\n" + "="*60)
    print("TEST 5: PRÉDICTION AVEC PROBABILITÉS")
    print("="*60)
    
    try:
        payload = {
            "resume_text": cv_text
        }
        
        response = requests.post(
            f"{API_URL}/predict?include_all_probabilities=true",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(" Prédiction réussie!")
            print(f"\n   Catégorie prédite: {data.get('category')}")
            print(f"   Confiance: {data.get('confidence'):.4f}")
            
            if data.get('all_probabilities'):
                print("\n   Top 5 des probabilités:")
                probs = data.get('all_probabilities')
                for i, (cat, prob) in enumerate(list(probs.items())[:5], 1):
                    print(f"      {i}. {cat:30s} {prob:.4f}")
            
            return True
        else:
            print(f" Erreur: Status {response.status_code}")
            print(f"   Réponse: {response.json()}")
            return False
            
    except Exception as e:
        print(f" Erreur: {e}")
        return False


def test_batch_prediction():
    """Test 6: Prédiction en batch"""
    print("\n" + "="*60)
    print("TEST 6: PRÉDICTION BATCH")
    print("="*60)
    
    try:
        # Prendre les 3 premiers CV valides
        resumes_list = [
            SAMPLE_CVS["Data Scientist"],
            SAMPLE_CVS["Web Developer"],
            SAMPLE_CVS["DevOps Engineer"]
        ]
        
        payload = {
            "resumes": resumes_list
        }
        
        response = requests.post(f"{API_URL}/batch-predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f" Prédiction batch réussie!")
            print(f"\n   Total traité: {data.get('total_processed')}")
            print("\n   Résultats:")
            
            for i, pred in enumerate(data.get('predictions', []), 1):
                print(f"      CV {i}: {pred.get('category'):30s} (confiance: {pred.get('confidence'):.4f})")
            
            return True
        else:
            print(f" Erreur: Status {response.status_code}")
            print(f"   Réponse: {response.json()}")
            return False
            
    except Exception as e:
        print(f" Erreur: {e}")
        return False


def test_invalid_input():
    """Test 7: Gestion des erreurs (CV trop court)"""
    print("\n" + "="*60)
    print("TEST 7: GESTION DES ERREURS")
    print("="*60)
    
    try:
        payload = {
            "resume_text": "Short"  # Trop court (< 10 caractères)
        }
        
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 400:
            print(" Erreur correctement détectée!")
            print(f"   Message: {response.json().get('detail')}")
            return True
        else:
            print(f"  Status inattendu: {response.status_code}")
            print(f"   Réponse: {response.json()}")
            return False
            
    except Exception as e:
        print(f" Erreur: {e}")
        return False


# ============================================
# FONCTION PRINCIPALE
# ============================================

def run_all_tests():
    """Exécuter tous les tests"""
    print("\n" + "="*60)
    print(" DÉBUT DES TESTS DE L'API")
    print("="*60)
    print(f"\nURL de l'API: {API_URL}")
    
    results = {}
    
    # Test 1: Health Check
    results['health_check'] = test_health_check()
    
    if not results['health_check']:
        print("\n" + "="*60)
        print(" L'API n'est pas prête. Arrêt des tests.")
        print("="*60)
        return
    
    # Test 2: Get Categories
    results['categories'] = test_get_categories()
    
    # Test 3: Model Info
    results['model_info'] = test_model_info()
    
    # Test 4: Prédictions simples
    results['pred_data_scientist'] = test_single_prediction(
        SAMPLE_CVS["Data Scientist"], 
        "Data Scientist"
    )
    
    results['pred_web_dev'] = test_single_prediction(
        SAMPLE_CVS["Web Developer"], 
        "Web Developer"
    )
    
    # Test 5: Prédiction avec probabilités
    results['pred_with_probs'] = test_prediction_with_probabilities(
        SAMPLE_CVS["DevOps Engineer"]
    )
    
    # Test 6: Batch prediction
    results['batch_prediction'] = test_batch_prediction()
    
    # Test 7: Gestion des erreurs
    results['error_handling'] = test_invalid_input()
    
    # Résumé
    print("\n" + "="*60)
    print(" RÉSUMÉ DES TESTS")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "True" if result else "False"
        print(f"{status} {test_name}")
    
    print("\n" + "="*60)
    print(f"Résultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print(" TOUS LES TESTS SONT PASSÉS!")
    else:
        print("  Certains tests ont échoué")
    
    print("="*60 + "\n")


# ============================================
# EXÉCUTION
# ============================================

if __name__ == "__main__":
    run_all_tests()

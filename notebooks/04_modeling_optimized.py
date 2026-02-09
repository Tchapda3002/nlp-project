"""
MODÉLISATION OPTIMISÉE - CV Classifier
======================================
Script d'amélioration des performances du modèle
"""

import pandas as pd
import numpy as np
from scipy.sparse import load_npz, hstack, csr_matrix
import pickle
import warnings
from pathlib import Path

# ML imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# Pour le resampling
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ imbalanced-learn non installé. Installer avec: pip install imbalanced-learn")

# Pour hyperparameter tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna non installé. Installer avec: pip install optuna")

# MLflow
import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = Path('../data/processed')
MODELS_PATH = Path('../models')
RANDOM_STATE = 42
CV_FOLDS = 5

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("=" * 70)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 70)

df = pd.read_csv(DATA_PATH / 'resume_cleaned.csv')

# Vérifier les colonnes disponibles
print(f"Colonnes disponibles: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# Identifier la colonne texte et catégorie
text_col = 'Resume_clean' if 'Resume_clean' in df.columns else 'cleaned_text'
category_col = 'Category'

print(f"\nDistribution des catégories:")
print(df[category_col].value_counts())

# =============================================================================
# 2. AMÉLIORATION DU FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING AMÉLIORÉ")
print("=" * 70)

# TF-IDF amélioré avec n-grams
vectorizer_improved = TfidfVectorizer(
    max_features=8000,           # Plus de features
    ngram_range=(1, 2),          # Unigrams + bigrams
    min_df=2,                    # Ignorer termes très rares
    max_df=0.90,                 # Ignorer termes dans >90% des docs
    sublinear_tf=True,           # log(tf) pour réduire l'impact des termes fréquents
    norm='l2',
    strip_accents='unicode',
    lowercase=True
)

# Extraire les features
X_tfidf = vectorizer_improved.fit_transform(df[text_col])
print(f"TF-IDF Shape: {X_tfidf.shape}")

# Features additionnelles
def extract_meta_features(texts):
    """Extraire des features métadonnées du texte"""
    features = []
    for text in texts:
        words = str(text).split()
        features.append([
            len(words),                                    # Nombre de mots
            len(str(text)),                               # Nombre de caractères
            np.mean([len(w) for w in words]) if words else 0,  # Longueur moyenne des mots
            len(set(words)) / len(words) if words else 0, # Ratio vocabulaire unique
            str(text).count('\n'),                        # Nombre de lignes
            sum(1 for c in str(text) if c.isupper()) / len(str(text)) if text else 0  # Ratio majuscules
        ])
    return np.array(features)

X_meta = extract_meta_features(df[text_col])
scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)
print(f"Meta Features Shape: {X_meta_scaled.shape}")

# Combiner les features
X_combined = hstack([X_tfidf, csr_matrix(X_meta_scaled)])
print(f"Combined Features Shape: {X_combined.shape}")

# Encoder les labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[category_col])
print(f"\nNombre de classes: {len(label_encoder.classes_)}")

# =============================================================================
# 3. GESTION DU DÉSÉQUILIBRE DES CLASSES
# =============================================================================
print("\n" + "=" * 70)
print("3. GESTION DU DÉSÉQUILIBRE")
print("=" * 70)

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Calculer les poids des classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"\nClass weights calculés (balanced)")

# Appliquer SMOTE si disponible
if IMBLEARN_AVAILABLE:
    # Utiliser RandomOverSampler (plus sûr avec peu de données)
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    print(f"\nAprès resampling: {X_train_resampled.shape[0]} samples")
else:
    X_train_resampled, y_train_resampled = X_train, y_train

# =============================================================================
# 4. DÉFINITION DES MODÈLES OPTIMISÉS
# =============================================================================
print("\n" + "=" * 70)
print("4. MODÈLES OPTIMISÉS")
print("=" * 70)

models = {
    'Random_Forest_Balanced': RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'Logistic_Regression_L2': LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight='balanced',
        solver='lbfgs',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'Gradient_Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        subsample=0.8,
        random_state=RANDOM_STATE
    ),
    'SVM_RBF': SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=RANDOM_STATE
    )
}

# =============================================================================
# 5. CROSS-VALIDATION ROBUSTE
# =============================================================================
print("\n" + "=" * 70)
print("5. CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

results_cv = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Évaluation: {name}")
    print(f"{'='*50}")

    cv_results = cross_validate(
        model, X_train_resampled, y_train_resampled,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    results_cv[name] = {
        'train_accuracy': cv_results['train_accuracy'].mean(),
        'test_accuracy': cv_results['test_accuracy'].mean(),
        'test_accuracy_std': cv_results['test_accuracy'].std(),
        'f1_weighted': cv_results['test_f1_weighted'].mean(),
        'f1_weighted_std': cv_results['test_f1_weighted'].std(),
        'precision': cv_results['test_precision_weighted'].mean(),
        'recall': cv_results['test_recall_weighted'].mean()
    }

    print(f"  Train Accuracy: {results_cv[name]['train_accuracy']:.4f}")
    print(f"  CV Accuracy:    {results_cv[name]['test_accuracy']:.4f} (+/- {results_cv[name]['test_accuracy_std']:.4f})")
    print(f"  CV F1-Score:    {results_cv[name]['f1_weighted']:.4f} (+/- {results_cv[name]['f1_weighted_std']:.4f})")
    print(f"  CV Precision:   {results_cv[name]['precision']:.4f}")
    print(f"  CV Recall:      {results_cv[name]['recall']:.4f}")

# =============================================================================
# 6. HYPERPARAMETER TUNING AVEC OPTUNA
# =============================================================================
if OPTUNA_AVAILABLE:
    print("\n" + "=" * 70)
    print("6. HYPERPARAMETER TUNING (Optuna)")
    print("=" * 70)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }

        model = RandomForestClassifier(**params)
        scores = cross_val_score(
            model, X_train_resampled, y_train_resampled,
            cv=cv, scoring='f1_weighted', n_jobs=-1
        )
        return scores.mean()

    # Créer et lancer l'étude
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )

    # Moins de trials pour la démo, augmenter pour de meilleurs résultats
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\nMeilleurs paramètres trouvés:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nMeilleur F1-Score CV: {study.best_value:.4f}")

    # Créer le modèle optimisé
    best_rf_params = study.best_params
    best_rf_params['class_weight'] = 'balanced'
    best_rf_params['random_state'] = RANDOM_STATE
    best_rf_params['n_jobs'] = -1

    models['RF_Optuna_Optimized'] = RandomForestClassifier(**best_rf_params)

# =============================================================================
# 7. ENSEMBLE VOTING
# =============================================================================
print("\n" + "=" * 70)
print("7. ENSEMBLE VOTING")
print("=" * 70)

ensemble = VotingClassifier(
    estimators=[
        ('rf', models['Random_Forest_Balanced']),
        ('lr', models['Logistic_Regression_L2']),
        ('svm', models['SVM_RBF'])
    ],
    voting='soft',
    n_jobs=-1
)

# Évaluer l'ensemble
ensemble_scores = cross_val_score(
    ensemble, X_train_resampled, y_train_resampled,
    cv=cv, scoring='f1_weighted', n_jobs=-1
)
print(f"Ensemble Voting F1-Score: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")

# =============================================================================
# 8. ENTRAÎNEMENT FINAL ET ÉVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("8. ENTRAÎNEMENT FINAL")
print("=" * 70)

# Trouver le meilleur modèle basé sur F1-score CV
best_model_name = max(results_cv, key=lambda k: results_cv[k]['f1_weighted'])
print(f"\nMeilleur modèle: {best_model_name}")

# Entraîner le meilleur modèle sur toutes les données d'entraînement
best_model = models[best_model_name]
best_model.fit(X_train_resampled, y_train_resampled)

# Évaluer sur le test set
y_pred = best_model.predict(X_test)

print(f"\n{'='*50}")
print(f"RÉSULTATS SUR TEST SET")
print(f"{'='*50}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# =============================================================================
# 9. SAUVEGARDE
# =============================================================================
print("\n" + "=" * 70)
print("9. SAUVEGARDE DES MODÈLES")
print("=" * 70)

MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Sauvegarder le meilleur modèle
with open(MODELS_PATH / f'{best_model_name}_optimized.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Sauvegarder le vectorizer amélioré
with open(MODELS_PATH / 'tfidf_vectorizer_optimized.pkl', 'wb') as f:
    pickle.dump(vectorizer_improved, f)

# Sauvegarder le scaler
with open(MODELS_PATH / 'meta_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Sauvegarder le label encoder
with open(MODELS_PATH / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"✅ Modèles sauvegardés dans {MODELS_PATH}")

# =============================================================================
# 10. RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)

print(f"""
📊 COMPARAISON DES MODÈLES (F1-Score CV):
""")

for name, metrics in sorted(results_cv.items(), key=lambda x: x[1]['f1_weighted'], reverse=True):
    print(f"  {name:30s}: {metrics['f1_weighted']:.4f} (+/- {metrics['f1_weighted_std']:.4f})")

print(f"""

🏆 MEILLEUR MODÈLE: {best_model_name}
   - CV F1-Score: {results_cv[best_model_name]['f1_weighted']:.4f}
   - Test F1-Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}

📁 FICHIERS SAUVEGARDÉS:
   - {MODELS_PATH}/{best_model_name}_optimized.pkl
   - {MODELS_PATH}/tfidf_vectorizer_optimized.pkl
   - {MODELS_PATH}/meta_scaler.pkl
   - {MODELS_PATH}/label_encoder.pkl

💡 PROCHAINES AMÉLIORATIONS POSSIBLES:
   1. Ajouter plus de données (data augmentation, datasets externes)
   2. Utiliser des embeddings BERT (sentence-transformers)
   3. Fine-tuner un modèle Transformer sur tes données
   4. Fusionner des catégories similaires pour réduire le déséquilibre
""")

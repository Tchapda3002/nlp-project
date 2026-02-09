"""
ENTRAÎNEMENT OPTIMISÉ DES MODÈLES
=================================
Dataset: 962 CVs, 25 catégories
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import pickle
from pathlib import Path
import time
import warnings

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
RANDOM_STATE = 42

# =============================================================================
# 1. CHARGER LES DONNÉES
# =============================================================================
print("=" * 70)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 70)

X = load_npz(DATA_PROCESSED / 'X_features.npz')
y = np.load(DATA_PROCESSED / 'y_labels.npy')

with open(MODELS_DIR / 'label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print(f"Features: {X.shape}")
print(f"Labels: {y.shape}")
print(f"Classes: {len(label_encoder.classes_)}")

# =============================================================================
# 2. SPLIT TRAIN/TEST
# =============================================================================
print("\n" + "=" * 70)
print("2. SPLIT DES DONNÉES")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# =============================================================================
# 3. DÉFINITION DES MODÈLES
# =============================================================================
print("\n" + "=" * 70)
print("3. MODÈLES À ENTRAÎNER")
print("=" * 70)

models = {
    'Random_Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'Logistic_Regression': LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        solver='lbfgs',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'SVM_Linear': SVC(
        kernel='linear',
        C=1.0,
        class_weight='balanced',
        probability=True,
        random_state=RANDOM_STATE
    ),
    'Naive_Bayes': MultinomialNB(alpha=0.1),
    'Gradient_Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
}

for name in models:
    print(f"  - {name}")

# =============================================================================
# 4. CROSS-VALIDATION
# =============================================================================
print("\n" + "=" * 70)
print("4. CROSS-VALIDATION (5-fold)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, model in models.items():
    print(f"\n{name}:")

    start = time.time()

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)

    # Entraîner sur tout le train set
    model.fit(X_train, y_train)

    # Prédire sur test
    y_pred = model.predict(X_test)

    elapsed = time.time() - start

    # Métriques
    results[name] = {
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'test_recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'test_f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'time': elapsed,
        'model': model
    }

    print(f"  CV F1-Score:   {results[name]['cv_f1_mean']:.4f} (+/- {results[name]['cv_f1_std']:.4f})")
    print(f"  Test Accuracy: {results[name]['test_accuracy']:.4f}")
    print(f"  Test F1-Score: {results[name]['test_f1']:.4f}")
    print(f"  Time: {elapsed:.2f}s")

# =============================================================================
# 5. COMPARAISON
# =============================================================================
print("\n" + "=" * 70)
print("5. COMPARAISON DES MODÈLES")
print("=" * 70)

# Trier par F1-score test
sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)

print(f"\n{'Modèle':<25} {'CV F1':>10} {'Test Acc':>10} {'Test F1':>10} {'Time':>8}")
print("-" * 70)

for name, metrics in sorted_results:
    print(f"{name:<25} {metrics['cv_f1_mean']:>10.4f} {metrics['test_accuracy']:>10.4f} {metrics['test_f1']:>10.4f} {metrics['time']:>7.2f}s")

# =============================================================================
# 6. MEILLEUR MODÈLE
# =============================================================================
print("\n" + "=" * 70)
print("6. MEILLEUR MODÈLE")
print("=" * 70)

best_name, best_metrics = sorted_results[0]
best_model = best_metrics['model']

print(f"\nMeilleur modèle: {best_name}")
print(f"  Test Accuracy: {best_metrics['test_accuracy']:.4f}")
print(f"  Test F1-Score: {best_metrics['test_f1']:.4f}")
print(f"  Test Precision: {best_metrics['test_precision']:.4f}")
print(f"  Test Recall: {best_metrics['test_recall']:.4f}")

# Classification report
print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_, zero_division=0))

# =============================================================================
# 7. ENSEMBLE VOTING
# =============================================================================
print("\n" + "=" * 70)
print("7. ENSEMBLE VOTING")
print("=" * 70)

# Créer un ensemble avec les 3 meilleurs modèles
top_3 = sorted_results[:3]
estimators = [(name, results[name]['model']) for name, _ in top_3]

ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
ensemble.fit(X_train, y_train)

y_pred_ensemble = ensemble.predict(X_test)
ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)

print(f"Ensemble ({', '.join([n for n, _ in top_3])})")
print(f"  Test Accuracy: {ensemble_acc:.4f}")
print(f"  Test F1-Score: {ensemble_f1:.4f}")

# =============================================================================
# 8. SAUVEGARDE
# =============================================================================
print("\n" + "=" * 70)
print("8. SAUVEGARDE")
print("=" * 70)

# Déterminer le meilleur modèle final
if ensemble_f1 > best_metrics['test_f1']:
    final_model = ensemble
    final_name = "Ensemble"
    final_f1 = ensemble_f1
else:
    final_model = best_model
    final_name = best_name
    final_f1 = best_metrics['test_f1']

# Sauvegarder le meilleur modèle
with open(MODELS_DIR / 'best_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print(f"✓ Modèle sauvegardé: {MODELS_DIR / 'best_model.pkl'}")

# Sauvegarder aussi avec son nom
with open(MODELS_DIR / f'{final_name}_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print(f"✓ Copie sauvegardée: {MODELS_DIR / f'{final_name}_model.pkl'}")

# =============================================================================
# 9. RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)

print(f"""
ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS

Dataset:
  - Train: {X_train.shape[0]} CVs
  - Test: {X_test.shape[0]} CVs
  - Features: {X.shape[1]}
  - Catégories: {len(label_encoder.classes_)}

Meilleur Modèle: {final_name}
  - Test F1-Score: {final_f1:.4f}

Comparaison avec avant (169 CVs):
  - Avant: ~73.5% accuracy, ~68.8% F1
  - Maintenant: {best_metrics['test_accuracy']*100:.1f}% accuracy, {best_metrics['test_f1']*100:.1f}% F1

Amélioration: +{(best_metrics['test_f1'] - 0.688) * 100:.1f}% F1-Score

Fichiers sauvegardés:
  - {MODELS_DIR / 'best_model.pkl'}
  - {MODELS_DIR / 'tfidf_vectorizer.pkl'}
  - {MODELS_DIR / 'label_encoder.pkl'}
""")

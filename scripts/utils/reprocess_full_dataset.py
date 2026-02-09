"""
REPROCESSING COMPLET DU DATASET (962 CVs)
=========================================
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import save_npz
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Créer les dossiers
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. CHARGER LE DATASET
# =============================================================================
print("=" * 70)
print("1. CHARGEMENT DU DATASET")
print("=" * 70)

df = pd.read_csv(DATA_RAW / 'resume_dataset.csv')
print(f"CVs chargés: {len(df)}")
print(f"Colonnes: {df.columns.tolist()}")

# Utiliser la colonne Resume (cleaned_resume est vide)
text_col = 'Resume'
print(f"Colonne texte utilisée: {text_col}")

# =============================================================================
# 2. NETTOYAGE SUPPLÉMENTAIRE
# =============================================================================
print("\n" + "=" * 70)
print("2. NETTOYAGE DU TEXTE")
print("=" * 70)

def clean_text(text):
    """Nettoyage du texte pour ML"""
    if not isinstance(text, str):
        return ""

    # Minuscules
    text = text.lower()

    # Supprimer URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Supprimer emails
    text = re.sub(r'\S+@\S+', '', text)

    # Supprimer caractères spéciaux (garder lettres, chiffres et espaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Appliquer le nettoyage (sans tqdm pour éviter les conflits)
print("Nettoyage en cours...")
df['cleaned_text'] = df[text_col].apply(clean_text)
print("Nettoyage terminé.")

# Filtrer les CVs trop courts
min_length = 100
df = df[df['cleaned_text'].str.len() >= min_length]
print(f"\nCVs après filtrage (>= {min_length} chars): {len(df)}")

# =============================================================================
# 3. ANALYSE DES CATÉGORIES
# =============================================================================
print("\n" + "=" * 70)
print("3. DISTRIBUTION DES CATÉGORIES")
print("=" * 70)

print(df['Category'].value_counts())
print(f"\nTotal catégories: {df['Category'].nunique()}")

# =============================================================================
# 4. EXTRACTION TF-IDF
# =============================================================================
print("\n" + "=" * 70)
print("4. EXTRACTION TF-IDF")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    norm='l2',
    strip_accents='unicode'
)

X = vectorizer.fit_transform(df['cleaned_text'])
print(f"Matrice TF-IDF: {X.shape}")
print(f"Sparsité: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")

# Encoder les labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])
print(f"Classes encodées: {len(label_encoder.classes_)}")

# =============================================================================
# 5. SAUVEGARDE
# =============================================================================
print("\n" + "=" * 70)
print("5. SAUVEGARDE")
print("=" * 70)

# CSV nettoyé
df.to_csv(DATA_PROCESSED / 'resume_cleaned.csv', index=False)
print(f"✓ {DATA_PROCESSED / 'resume_cleaned.csv'}")

# Matrice TF-IDF
save_npz(DATA_PROCESSED / 'X_features.npz', X)
print(f"✓ {DATA_PROCESSED / 'X_features.npz'}")

# Labels encodés
np.save(DATA_PROCESSED / 'y_labels.npy', y)
print(f"✓ {DATA_PROCESSED / 'y_labels.npy'}")

# Vectorizer
with open(MODELS_DIR / 'tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"✓ {MODELS_DIR / 'tfidf_vectorizer.pkl'}")

# Label encoder
with open(MODELS_DIR / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✓ {MODELS_DIR / 'label_encoder.pkl'}")

# =============================================================================
# 6. RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)

print(f"""
DATASET RETRAITÉ AVEC SUCCÈS

  CVs traités: {len(df)}
  Features TF-IDF: {X.shape[1]}
  Catégories: {len(label_encoder.classes_)}

  CVs par catégorie (moyenne): {len(df) / len(label_encoder.classes_):.1f}
  CVs par catégorie (min): {df['Category'].value_counts().min()}
  CVs par catégorie (max): {df['Category'].value_counts().max()}

PROCHAINE ÉTAPE:
  python scripts/train_optimized.py
""")

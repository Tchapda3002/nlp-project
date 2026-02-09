# SOLUTION AU PROBLÈME NUMPY 2.x

## LE PROBLÈME
Vos modèles ont été créés avec NumPy 1.x, mais vous avez NumPy 2.4.1 installé.
Cette incompatibilité empêche le chargement des modèles.

## SOLUTION 1 (RAPIDE): Downgrade NumPy

Ouvrez un terminal et exécutez:

```bash
pip uninstall numpy -y
pip install "numpy<2.0"
```

Puis réinstallez les autres dépendances si nécessaire:

```bash
pip install scikit-learn pandas pyarrow --upgrade
```

## SOLUTION 2 (RECOMMANDÉE): Recréer les modèles avec NumPy 2.x

### Étape 1: Vérifier vos versions
```bash
pip list | findstr numpy
pip list | findstr scikit-learn
pip list | findstr pandas
```

### Étape 2: Mettre à jour tous les packages
```bash
pip install --upgrade numpy scikit-learn pandas pyarrow
```

### Étape 3: Réentraîner les modèles
```bash
cd notebooks
jupyter notebook 04_modeling.ipynb
# Exécutez TOUTES les cellules pour recréer les modèles
```

Cela recréera les fichiers .pkl compatibles avec NumPy 2.x.

## SOLUTION 3: Utiliser un environnement virtuel propre

```bash
# Créer un nouvel environnement
python -m venv venv_compatible

# Activer (Windows)
venv_compatible\Scripts\activate

# Installer avec des versions compatibles
pip install numpy==1.26.4
pip install scikit-learn==1.5.1
pip install pandas==2.2.2
pip install fastapi uvicorn nltk

# Lancer l'API depuis cet environnement
cd api
python main.py
```

## VÉRIFICATION

Après avoir appliqué la solution, testez:

```python
import numpy as np
import sklearn
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"pandas: {pd.__version__}")
```

Versions recommandées:
- NumPy: 1.26.4 (pour compatibilité) OU 2.0+ (si vous recréez les modèles)
- scikit-learn: 1.5+
- pandas: 2.2+

## QUE FAIRE MAINTENANT?

**Option A (RAPIDE - 5 minutes):**
```bash
pip uninstall numpy -y
pip install "numpy<2.0"
cd api
python main.py
```

**Option B (PROPRE - 30 minutes):**
```bash
pip install --upgrade numpy scikit-learn pandas
cd notebooks
jupyter notebook 04_modeling.ipynb
# Réexécuter le notebook pour recréer les modèles
cd ../api
python main.py
```

Je recommande l'**Option A** pour tester rapidement, puis l'**Option B** pour une solution durable.

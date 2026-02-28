# diabetes-decision-tree
Projet de classification du diabète avec GridSearchCV et arbre de décision
# Système de Diagnostic du Diabète avec Arbre de Décision

## Description
Ce projet implémente un **système de diagnostic du diabète** utilisant un **arbre de décision** pour prédire si un patient est diabétique ou non à partir de ses données médicales.  
Le modèle est simple, interprétable et peut être testé via une interface utilisateur (optionnelle).

-----

## Objectif
- Prédire le diagnostic diabétique (Diabétique / Non diabétique) à partir de variables telles que : âge, IMC, tension artérielle, taux de glucose, antécédents familiaux, etc.
- Fournir un modèle interprétable avec visualisation de l’arbre de décision.
- Optionnel : permettre à un utilisateur de tester son diagnostic via une interface Streamlit.

---

## Données
**Variables d’entrée :**  
- Âge  
- Sexe  
- IMC (Indice de Masse Corporelle)  
- Tension artérielle  
- Taux de glucose  
- Antécédents familiaux  
- Autres mesures médicales disponibles  

**Variable de sortie :**  
- Diagnostic (Diabétique / Non diabétique)  

**Source :**  
- Dataset public (exemple : Pima Indians Diabetes Dataset)

---

## Technologies et Outils
- **Langage :** Python  
- **Bibliothèques :**  
  - `pandas` → manipulation et nettoyage des données  
  - `numpy` → calculs numériques  
  - `scikit-learn` → création et évaluation de l’arbre de décision  
  - `matplotlib` / `seaborn` → visualisation des données et de l’arbre  
  - `joblib` → sauvegarde du modèle  
  - `streamlit` (optionnel) → interface utilisateur  

- **IDE recommandé :** VS Code, PyCharm, Jupyter Notebook

---

## Structure du projet
projetbigdata/
│
├── data/ # Dataset original et nettoyé
│ └── dataset_clean.csv
│
├── model/ # Scripts pour l’entraînement et l’évaluation
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── decision_tree.joblib
│
├── evaluation/ # Résultats d’évaluation et visualisation
│ ├── classification_report.txt
│ ├── confusion_matrix.png
│ ├── feature_importance.png
│ └── rules.txt
│
├── report/ # Rapport de projet
│ └── rapport_modelisation.md
│
└── app_streamlit.py # Interface utilisateur Streamlit (optionnelle)

---

## Étapes réalisées
1. **Collecte et prétraitement des données**  
   - Nettoyage (gestion des valeurs manquantes, remplacement des zéros invalides)  
   - Encodage des variables catégorielles  
   - Normalisation si nécessaire  
   - Création du dataframe final

2. **Analyse exploratoire (EDA)**  
   - Statistiques descriptives  
   - Corrélations et distributions graphiques

3. **Modélisation**  
   - Séparation train/test  
   - Entraînement de l’arbre de décision (`DecisionTreeClassifier`)  
   - Optimisation des hyperparamètres (`max_depth`, `min_samples_leaf`, `criterion`)  
   - Évaluation (accuracy, F1-score, recall, matrice de confusion)  

4. **Visualisation**  
   - Arbre de décision exporté en PNG  
   - Analyse de l’importance des features  
   - Export des règles de décision  

5. **Interface utilisateur (optionnelle)**  
   - Formulaire pour entrer les données  
   - Bouton “Diagnostiquer”  
   - Affichage du résultat et de la probabilité

---

## Membres et responsabilités
| Membre | Rôle | Tâches principales | Livrables |
|--------|------|-----------------|-----------|
| Membre 1 | Responsable Data | Collecte et prétraitement, EDA | `dataset_clean.csv`, `Notebook EDA.ipynb`, graphiques |
| Membre 2 | Modélisation | Création et évaluation du modèle | `train_model.py`, `decision_tree.joblib`, rapport + matrice de confusion |
| Membre 3 | Visualisation & Interface | Visualisation de l’arbre et interface Streamlit | `arbre_diabete.png`, `app_streamlit.py`, schémas pipeline ML |

---

## Instructions pour exécuter
1. Cloner le dépôt :
```bash
git clone https://github.com/RAYHANFARAJ/diabetes-decision-tree.git
cd diabetes-decision-tree

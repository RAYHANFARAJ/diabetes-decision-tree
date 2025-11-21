\# Tâche 3 – Visualisation \& Interface (Membre 3)



Ce document décrit le travail réalisé pour la \*\*Tâche 3\*\* du projet \*Diabetes Decision Tree\*, correspondant au rôle \*\*Visualisation \& Interface\*\*.



\## 1️⃣ Objectif



\- Créer une \*\*interface utilisateur interactive\*\* avec Streamlit pour tester le modèle de prédiction du diabète.

\- Fournir des \*\*visualisations claires\*\* de l’arbre de décision et de l’importance des features.

\- Exporter les graphiques et schémas pour documenter le pipeline Machine Learning.



---



\## 2️⃣ Étapes réalisées



\### a) Visualisation



\- Export de l’arbre de décision en PNG (`evaluation/decision\_tree\_simplified.png`).

\- Analyse de l’importance des features (`evaluation/feature\_importance.png`).

\- Schématisation du pipeline ML pour la documentation.



\### b) Interface utilisateur avec Streamlit



\- Création de `app/app\_streamlit.py`.

\- Formulaire permettant d’\*\*entrer les données utilisateur\*\*.

\- Bouton \*\*“Diagnostiquer”\*\* pour lancer la prédiction.

\- Affichage du \*\*résultat et de la probabilité\*\* de diabète.

\- L’interface utilise le modèle pré-entraîné `model/decision\_tree.joblib`.



---



\## 3️⃣ Fichiers livrables



| Fichier | Description |

|---------|-------------|

| `evaluation/decision\_tree\_simplified.png` | Représentation graphique de l’arbre de décision |

| `evaluation/feature\_importance.png` | Importance des features pour le modèle |

| `app/app\_streamlit.py` | Application Streamlit pour tester le modèle |



---



\## 4️⃣ Instructions pour utiliser l’interface



1\. Installer les dépendances :



```bash

pip install -r requirements.txt




import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os


# -------------------------------------------------
#            CONFIGURATION DE LA PAGE
# -------------------------------------------------
st.set_page_config(
    page_title="Diagnostic Diabète",
    page_icon="🩺",
    layout="wide",
)

# ----------------- CUSTOM STYLE -------------------
st.markdown("""
<style>
.main-title { font-size: 32px; font-weight: bold; color: #0a89c2; margin-bottom: 20px; }
.section-title { font-size: 24px; font-weight: bold; color: #146494; margin-top: 25px; }
.card {
    background-color: #f5f9ff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px #dcdcdc;
    margin-bottom: 20px;
}
.sidebar .sidebar-content {
    background: linear-gradient(#d7ebff, #edf5ff);
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
#                 CHEMINS DES FICHIERS
# -------------------------------------------------
MODEL_PATH = "./model/decision_tree.joblib"
TREE_IMG = "./evaluation/decision_tree_visual.png"
FEATURE_IMG = "./evaluation/feature_importance.png"
RULES_PATH = "./evaluation/rules.txt"

model = joblib.load(MODEL_PATH)


# -------------------------------------------------
#                 MENU LATÉRAL
# -------------------------------------------------
menu = st.sidebar.radio(
    "☰ Menu",
    [
        "🏠 Accueil",
        "🧪 Diagnostic",
        "🌳 Arbre de Décision",
        "📊 Importance des Variables",
        "📘 Règles du Modèle"
    ]
)



## =================================================
#                 PAGE : ACCUEIL
# =================================================
if menu == "🏠 Accueil":

    st.markdown("<h1 style='text-align:center; color:#0a89c2;'> Bienvenue au SugarSense</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Découvrez notre application interactive basée sur un <b>arbre de décision</b> pour prédire le diabète avec précision.</p>", unsafe_allow_html=True)

    # ---------------- IMAGES EN LIGNE ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("./assets/diabete5.jpg", use_container_width=True)
        st.caption("Symptômes")

    with col2:
        st.image("./assets/diabete2.jpg", use_container_width=True)
        st.caption("Prévention")

    with col3:
        st.image("./assets/diabete6.jpg", use_container_width=True)
        st.caption("Diagnostic")

    # ---------------- VIDEO EXPLICATIVE ----------------
    # ---------------- TEXTE EXPLICATIF À LA PLACE DE LA VIDÉO ----------------
    st.markdown("<h2 style='color:#146494;'>💡 Le diabète : Comprendre et se protéger</h2>", unsafe_allow_html=True)

    st.markdown("""
    Le **diabète** est une maladie chronique caractérisée par un taux élevé de glucose dans le sang.  
    Il apparaît lorsque le corps ne produit pas assez d’**insuline** ou ne l’utilise pas correctement.

    ### 🔍 Les signes qui doivent alerter :
    - Soif excessive  
    - Fatigue inhabituelle  
    - Perte ou prise de poids rapide  
    - Envies fréquentes d'uriner  
    - Vision trouble  

    ---

    ### 🛡️ Comment se protéger du diabète ?  
    Voici des actions simples et efficaces :

    - 🥗 **Adopter une alimentation équilibrée** (moins de sucre, moins de fritures, plus de légumes)  
    - 🚶 **Bouger au moins 30 minutes par jour**  
    - ⚖️ **Maintenir un poids stable**  
    - 💧 **Boire suffisamment d’eau**  
    - 😴 **Bien dormir et gérer le stress**  
    - 🩺 **Faire un contrôle de glycémie régulièrement**, surtout s’il y a des antécédents familiaux  

    Un mode de vie sain permet de réduire jusqu’à **70%** le risque de développer un diabète de type 2.

    """, unsafe_allow_html=True)


# =================================================
#             PAGE : DIAGNOSTIC
# =================================================
elif menu == "🧪 Diagnostic":

    st.markdown("<p class='section-title'>🧪 Tester un Patient</p>", unsafe_allow_html=True)

    with st.form("form_predict"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        Pregnancies = col1.number_input("Grossesses", 0, 20)
        Glucose = col2.number_input("Glucose", 0.0, 250.0)
        BloodPressure = col3.number_input("Pression Artérielle", 0.0, 150.0)

        SkinThickness = col1.number_input("Épaisseur de Peau", 0.0, 100.0)
        Insulin = col2.number_input("Insuline", 0.0, 900.0)
        BMI = col3.number_input("IMC", 0.0, 70.0)

        DPF = col1.number_input("DPF (Hérédité)", 0.0, 3.0)
        Age = col2.number_input("Âge", 1, 120)

        st.markdown("</div>", unsafe_allow_html=True)

        submit = st.form_submit_button("🔍 Diagnostiquer")

    if submit:
        data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                 Insulin, BMI, DPF, Age]]

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][pred] * 100

        if pred == 1:
            st.error(f"🩺 Résultat : **DIABÉTIQUE** (Confiance : {prob:.2f}%)")
        else:
            st.success(f"🩺 Résultat : **NON DIABÉTIQUE** (Confiance : {prob:.2f}%)")



# =================================================
#          PAGE : ARBRE DE DÉCISION
# =================================================
elif menu == "🌳 Arbre de Décision":

    st.markdown("<p class='section-title'>🌳 Arbre de Décision</p>", unsafe_allow_html=True)

    if os.path.exists(TREE_IMG):
        st.image(TREE_IMG, use_container_width=True)
    else:
        st.warning("Image manquante.")



# =================================================
#         PAGE : IMPORTANCE DES FEATURES
# =================================================
elif menu == "📊 Importance des Variables":

    st.markdown("<p class='section-title'>📊 Importance des Variables</p>", unsafe_allow_html=True)

    if os.path.exists(FEATURE_IMG):
        st.image(FEATURE_IMG, width=650)
    else:
        st.warning("Image introuvable.")



# =================================================
#              PAGE : RÈGLES DU MODÈLE
# =================================================
elif menu == "📘 Règles du Modèle":

    st.markdown("<p class='section-title'>📘 Règles du modèle (export_text)</p>", unsafe_allow_html=True)

    if os.path.exists(RULES_PATH):
        with open(RULES_PATH, "r") as f:
            rules = f.read()
        st.code(rules, language="markdown")
    else:
        st.warning("rules.txt non trouvé.")

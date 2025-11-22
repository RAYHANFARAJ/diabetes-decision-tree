import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os

# ----------- CONFIG --------------
st.set_page_config(
    page_title="Diagnostic Diabète - Decision Tree",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------- LOAD MODEL ----------
MODEL_PATH = "model/decision_tree.joblib"
RULES_PATH = "evaluation/rules.txt"
TREE_IMG = "evaluation/decision_tree_visual.png"
FEATURE_IMG = "evaluation/feature_importance.png"

model = joblib.load(MODEL_PATH)

# ------------- UI STYLE ----------
st.markdown("""
<style>
.big-title { font-size: 30px; font-weight: bold; color: #0a89c2; }
.section-title { font-size: 22px; font-weight: bold; color: #1b82b1; margin-top: 20px; }
.card {
    padding: 20px;
    border-radius: 10px;
    background-color: #f5f9ff;
    box-shadow: 0px 0px 10px #e0e0e0;
}
</style>
""", unsafe_allow_html=True)


# ============================
#   📌 SIDE MENU
# ============================
menu = st.sidebar.radio(
    "📌 Menu",
    ["🏠 Accueil",
     "🧪 Test de Diagnostic",
     "🌳 Arbre de Décision",
     "📊 Importance des Variables",
     "📘 Règles du Modèle"]
)


# ============================
#   🏠 ACCUEIL
# ============================
if menu == "🏠 Accueil":
    st.markdown("<p class='big-title'>🩺 Système de Diagnostic du Diabète</p>", unsafe_allow_html=True)
    st.write("""
    Bienvenue dans l'application interactive basée sur **un arbre de décision**
    développé pour prédire si un patient est diabétique ou non.  
    """)

    st.image(TREE_IMG, caption="Arbre de Décision - Vue Globale", use_container_width=True)


# ============================
#   🧪 TEST DE DIAGNOSTIC
# ============================
elif menu == "🧪 Test de Diagnostic":
    st.markdown("<p class='section-title'>🧪 Tester un Patient</p>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        Pregnancies = col1.number_input("Grossesses", min_value=0, max_value=20, step=1)
        Glucose = col2.number_input("Glucose", min_value=0.0, max_value=250.0)
        BloodPressure = col3.number_input("Pression Artérielle", min_value=0.0, max_value=150.0)

        SkinThickness = col1.number_input("Épaisseur de Peau", min_value=0.0, max_value=100.0)
        Insulin = col2.number_input("Insuline", min_value=0.0, max_value=900.0)
        BMI = col3.number_input("IMC", min_value=0.0, max_value=70.0)

        DiabetesPedigreeFunction = col1.number_input("DPF (hérédité)", min_value=0.0, max_value=3.0)
        Age = col2.number_input("Âge", min_value=1, max_value=120)

        submitted = st.form_submit_button("🔍 Diagnostiquer")

    if submitted:
        input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                       Insulin, BMI, DiabetesPedigreeFunction, Age]]

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][pred] * 100

        st.success("🎉 Diagnostic effectué !")

        if pred == 1:
            st.error(f"🩺 **Résultat : DIABÉTIQUE** (Confiance : {prob:.2f}%)")
        else:
            st.success(f"🩺 **Résultat : NON DIABÉTIQUE** (Confiance : {prob:.2f}%)")



# ============================
#   🌳 VISUALISATION ARBRE
# ============================
elif menu == "🌳 Arbre de Décision":
    st.markdown("<p class='section-title'>🌳 Arbre de Décision Complet</p>", unsafe_allow_html=True)

    st.info("Voici la visualisation officielle générée avec sklearn & matplotlib.")

    if os.path.exists(TREE_IMG):
        st.image(TREE_IMG, use_container_width=True)
    else:
        st.warning("⚠ L'image 'decision_tree_visual.png' n'a pas été trouvée.")



# ============================
#   📊 IMPORTANCE DES FEATURES
# ============================
elif menu == "📊 Importance des Variables":
    st.markdown("<p class='section-title'>📊 Importance des Variables</p>", unsafe_allow_html=True)

    if os.path.exists(FEATURE_IMG):
        st.image(FEATURE_IMG, use_container_width=False, width=600)
    else:
        st.warning("⚠ L'image 'feature_importance.png' est manquante.")


# ============================
#   📘 RÈGLES DU MODÈLE
# ============================
elif menu == "📘 Règles du Modèle":
    st.markdown("<p class='section-title'>📘 Règles du modèle (export_text)</p>", unsafe_allow_html=True)

    if os.path.exists(RULES_PATH):
        with open(RULES_PATH, "r") as f:
            rules = f.read()
        st.code(rules, language="markdown")
    else:
        st.warning("⚠ rules.txt non trouvé !")

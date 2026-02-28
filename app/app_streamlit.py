import streamlit as st
import joblib
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
BASE_DIR = os.path.dirname(__file__)           # chemin vers app/
ROOT_DIR = os.path.join(BASE_DIR, "..")        # chemin vers projetbigdata

# Modèle
MODEL_PATH = os.path.join(ROOT_DIR, "model", "decision_tree.joblib")

# Images
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
DIABETE5_IMG = os.path.join(ASSETS_DIR, "diabete5.jpg")
DIABETE2_IMG = os.path.join(ASSETS_DIR, "diabete2.jpg")
DIABETE6_IMG = os.path.join(ASSETS_DIR, "diabete6.jpg")

# Évaluation
EVAL_DIR = os.path.join(ROOT_DIR, "evaluation")
TREE_IMG = os.path.join(EVAL_DIR, "decision_tree_visual.png")
FEATURE_IMG = os.path.join(EVAL_DIR, "feature_importance.png")
RULES_PATH = os.path.join(EVAL_DIR, "rules.txt")

# Charger le modèle avec vérification
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error(f"Modèle non trouvé à l'emplacement : {MODEL_PATH}")
    st.stop()  # arrête le script si le modèle est introuvable

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

# =================================================
#                 PAGE : ACCUEIL
# =================================================
if menu == "🏠 Accueil":

    st.markdown("<h1 style='text-align:center; color:#0a89c2;'> Bienvenue au SugarSense</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Découvrez notre application interactive basée sur un <b>arbre de décision</b> pour prédire le diabète avec précision.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if os.path.exists(DIABETE5_IMG):
            st.image(DIABETE2_IMG, use_container_width=True)
        else:
            st.warning("Image diabete5.jpg manquante")
        st.caption("Symptômes")

    with col2:
        if os.path.exists(DIABETE2_IMG):
            st.image(DIABETE5_IMG, use_container_width=True)
        else:
            st.warning("Image diabete2.jpg manquante")
        st.caption("Prévention")

    with col3:
        if os.path.exists(DIABETE6_IMG):
            st.image(DIABETE6_IMG, use_container_width=True)
        else:
            st.warning("Image diabete6.jpg manquante")
        st.caption("Diagnostic")

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

    ### 🛡️ Comment se protéger du diabète ?  
    Voici des actions simples et efficaces :

    - 🥗 **Adopter une alimentation équilibrée**  
    - 🚶 **Bouger au moins 30 minutes par jour**  
    - ⚖️ **Maintenir un poids stable**  
    - 💧 **Boire suffisamment d’eau**  
    - 😴 **Bien dormir et gérer le stress**  
    - 🩺 **Faire un contrôle de glycémie régulièrement**
    """, unsafe_allow_html=True)

# =================================================
#                 PAGE : DIAGNOSTIC
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
        data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][pred] * 100

        st.markdown("<hr>", unsafe_allow_html=True)  # ligne de séparation

        # Style attrayant selon le résultat
        if pred == 1:
            st.markdown(
                f"""
                <div style="
                    background-color:#ffcccc;
                    border-radius:12px;
                    padding:20px;
                    text-align:center;
                    box-shadow:0px 0px 10px #dcdcdc;
                ">
                    <h2 style="color:#b30000;">🩺 Attention ! Résultat : DIABÉTIQUE</h2>
                    <p style="font-size:18px;">Confiance du modèle : <b>{prob:.2f}%</b></p>
                    <p style="font-size:16px;">
                        Il est fortement conseillé de consulter un professionnel de santé pour un suivi adapté.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background-color:#ccffcc;
                    border-radius:12px;
                    padding:20px;
                    text-align:center;
                    box-shadow:0px 0px 10px #dcdcdc;
                ">
                    <h2 style="color:#006600;">🩺 Bonne nouvelle ! Résultat : NON DIABÉTIQUE</h2>
                    <p style="font-size:18px;">Confiance du modèle : <b>{prob:.2f}%</b></p>
                    <p style="font-size:16px;">
                        Continuez à maintenir un mode de vie sain pour prévenir le diabète.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )

# =================================================
#          PAGE : ARBRE DE DÉCISION
# =================================================
elif menu == "🌳 Arbre de Décision":

    st.markdown("<p class='section-title'>🌳 Arbre de Décision</p>", unsafe_allow_html=True)
    if os.path.exists(TREE_IMG):
        st.image(TREE_IMG, use_column_width=True)
    else:
        st.warning("Image de l'arbre manquante")

# =================================================
#         PAGE : IMPORTANCE DES FEATURES
# =================================================
elif menu == "📊 Importance des Variables":

    st.markdown("<p class='section-title'>📊 Importance des Variables</p>", unsafe_allow_html=True)
    if os.path.exists(FEATURE_IMG):
        st.image(FEATURE_IMG, width=650)
    else:
        st.warning("Image des features manquante")

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
        st.warning("rules.txt non trouvé")

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance():
    # Charger le modèle
    model = joblib.load('model/decision_tree.joblib')

    # Features du dataset Pima Indians
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Importance des features
    importance = model.feature_importances_

    # Créer un DataFrame pour mieux visualiser
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    # Graphique d'importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Importance des Variables dans la Prédiction du Diabète')
    plt.tight_layout()
    plt.savefig('evaluation/feature_importance.png', dpi=300)
    plt.show()

    # Afficher les valeurs numériques
    print("Importance des features :")
    for feature, imp in zip(features, importance):
        print(f"{feature}: {imp:.4f}")


if __name__ == "__main__":
    plot_feature_importance()
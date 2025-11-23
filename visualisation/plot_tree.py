import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
from sklearn import tree
import graphviz


def visualize_decision_tree():
    # Charger le modèle entraîné
    model = joblib.load('../model/decision_tree.joblib')
    # Visualisation avec matplotlib
    plt.figure(figsize=(20, 10))
    plot_tree(model,
              feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
              class_names=['Non Diabétique', 'Diabétique'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title("Arbre de Décision - Diagnostic du Diabète")
    plt.savefig('../evaluation/decision_tree_visual.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Version simplifiée pour le rapport
    plt.figure(figsize=(12, 6))
    plot_tree(model,
              feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
              class_names=['Non Diabétique', 'Diabétique'],
              filled=True,
              rounded=True,
              max_depth=3)  # Limiter la profondeur pour la lisibilité
    plt.title("Arbre de Décision (Profondeur limitée à 3)")
    plt.savefig('../evaluation/decision_tree_simplified.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    visualize_decision_tree()
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib

df = pd.read_csv(r"C:\ARBRE\projetbigdata\data\dataset_final.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, None],
    "min_samples_leaf": [1, 2, 5]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)
best = grid.best_estimator_

joblib.dump(best, r"C:\ARBRE\projetbigdata\model\decision_tree.joblib")

rules = export_text(best, feature_names=list(X.columns))
with open(r"C:\ARBRE\projetbigdata\evaluation\rules.txt", "w") as f:
    f.write(rules)

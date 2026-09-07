"""Train and save the final SVM artifacts used by app.py."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="diagnosis")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True, random_state=RANDOM_STATE))])
param_grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.01, 0.001], "svm__kernel": ["rbf", "linear"]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True)
grid.fit(X_train, y_train)
model = grid.best_estimator_
pred = model.predict(X_test)
prob_benign = model.predict_proba(X_test)[:, 1]

joblib.dump(model.named_steps["svm"], BASE_DIR / "model.pkl", compress=3)
joblib.dump(model.named_steps["scaler"], BASE_DIR / "scaler.pkl", compress=3)
joblib.dump(model, BASE_DIR / "pipeline.pkl", compress=3)

metrics = {
    "dataset_rows": len(X), "dataset_features": X.shape[1], "test_size": 0.20, "random_state": RANDOM_STATE,
    "best_params": grid.best_params_, "cv_roc_auc": grid.best_score_,
    "test_metrics": {
        "accuracy": accuracy_score(y_test, pred),
        "precision_benign": precision_score(y_test, pred, pos_label=1),
        "recall_benign": recall_score(y_test, pred, pos_label=1),
        "f1_benign": f1_score(y_test, pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, prob_benign),
    },
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}
(BASE_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))

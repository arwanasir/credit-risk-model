from os import name
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import pandas as pd
from pathlib import Path


DATA_PATH = Path("../data/processed/final_training_data.csv")


def train_models():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    mlflow.set_experiment("Credit_Risk_Model")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "f1": f1_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, probs),
            }

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} logged to MLflow")


# if name == "main":
   # train_models()

# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_modeling_data(data_path):
    """Load the final dataset for modeling."""
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=['is_high_risk', 'CustomerId'], errors='ignore')
    y = df['is_high_risk']

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """Train and tune Logistic Regression."""
    print("Training Logistic Regression...")

    model = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_random_forest(X_train, y_train):
    """Train and tune Random Forest."""
    print("Training Random Forest...")

    model = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    random_search = RandomizedSearchCV(
        model, param_dist, n_iter=5, cv=3, scoring='roc_auc', random_state=42)
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_


def train_gradient_boosting(X_train, y_train):
    """Train and tune Gradient Boosting."""
    print("Training Gradient Boosting...")

    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """Calculate performance metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    return metrics


def log_experiment(model, model_name, params, metrics, run_name="CreditRiskModel"):
    """Log experiment to MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, f"{model_name}_model")

        print(f"\n{model_name} logged to MLflow")
        print(f"Params: {params}")
        print(f"Metrics: {metrics}")


def main():
    # Start MLflow
    mlflow.set_experiment("Credit_Risk_Modeling")

    # Load data
    DATA_PATH = '../data/processed/final_training_data.csv'
    X, y = load_modeling_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Train models
    models = {}

    # 1. Logistic Regression
    lr_model, lr_params = train_logistic_regression(X_train, y_train)
    models['LogisticRegression'] = lr_model

    # 2. Random Forest
    rf_model, rf_params = train_random_forest(X_train, y_train)
    models['RandomForest'] = rf_model

    # 3. Gradient Boosting
    gb_model, gb_params = train_gradient_boosting(X_train, y_train)
    models['GradientBoosting'] = gb_model

    # Evaluate and log
    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        params = lr_params if name == 'LogisticRegression' else rf_params if name == 'RandomForest' else gb_params

        log_experiment(model, name, params, metrics, run_name=f"{name}_run")

        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model = model
            best_name = name

    # Register best model
    if best_model:
        print(f"\nBest model: {best_name} with ROC-AUC: {best_score:.4f}")

        # Save locally
        joblib.dump(best_model, '../models/best_model.pkl')
        print("Best model saved to models/best_model.pkl")

        # Register in MLflow
        mlflow.sklearn.log_model(
            best_model, "best_model", registered_model_name="CreditRisk_Prod")

    # Save test set for later
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('../data/processed/test_set.csv', index=False)
    print("\nTest set saved to data/processed/test_set.csv")

# if __name__ == '__main__':
 #   main()

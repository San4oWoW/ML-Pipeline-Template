import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn


def load_params(path=None):
    if path is None:
        # путь из корня проекта
        project_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(project_root, "params.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data():
    train_path = "data/processed/train.csv"
    df = pd.read_csv(train_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def build_model(params):
    model_type = params["train"]["model_type"]
    random_state = params["train"]["random_state"]

    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, random_state=random_state)

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=params["train"]["n_estimators"],
            random_state=random_state,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def main():
    params = load_params()
    X, y = load_data()
    model = build_model(params)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run():
        # параметры
        mlflow.log_param("model_type", params["train"]["model_type"])
        mlflow.log_param("random_state", params["train"]["random_state"])
        mlflow.log_param("n_estimators", params["train"]["n_estimators"])

        # обучение
        model.fit(X, y)

        # метрики
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        mlflow.log_metric("accuracy", acc)

        # сохранение модели
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)

        # логирование артефакта
        mlflow.log_artifact(model_path)

        print(f"Training completed. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

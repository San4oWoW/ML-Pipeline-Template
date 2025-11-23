import os
import json
import joblib
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score


def load_params(path=None):
    if path is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(project_root, "params.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data():
    project_root = os.path.dirname(os.path.dirname(__file__))
    test_path = os.path.join(project_root, "data", "processed", "test.csv")
    df = pd.read_csv(test_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def load_model():
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root, "models", "model.pkl")
    return joblib.load(model_path)


def main():
    X, y = load_data()
    model = load_model()

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    metrics = {"accuracy": acc}

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation completed. Accuracy on test: {acc:.4f}")


if __name__ == "__main__":
    main()

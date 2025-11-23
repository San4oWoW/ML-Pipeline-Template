import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_params(path=None):
    if path is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(project_root, "params.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def main():
    params = load_params()

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    # reproducibility
    np.random.seed(random_state)

    df = pd.read_csv("data/raw/iris.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    os.makedirs("data/processed", exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train

    test_df = X_test.copy()
    test_df["target"] = y_test

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Prepare stage done.")


if __name__ == "__main__":
    main()

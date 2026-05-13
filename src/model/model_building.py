import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import os
from src.logger import logger


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded from {file_path}")
    return df


def train_model(X_train, y_train):
    model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=1000
    )

    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved at {path}")


def main():

    train_data = load_data("./data/processed/train_bow.csv")

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    model = train_model(X_train, y_train)

    save_model(model, "models/model.pkl")


if __name__ == "__main__":
    main()
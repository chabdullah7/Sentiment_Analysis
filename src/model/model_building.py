import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from src.logger import logger


def load_data(file_path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(file_path)

        logger.info(f"Data loaded from {file_path}")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LogisticRegression:

    try:

        model = LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=1000
        )

        model.fit(X_train, y_train)

        logger.info("Model training completed")

        return model

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def save_model(model, file_path: str):

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {file_path}")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():

    try:

        # =========================
        # LOAD TRAIN DATA
        # =========================
        train_data = load_data(
            "./data/processed/train_bow.csv"
        )

        # =========================
        # SPLIT FEATURES/LABELS
        # =========================
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # =========================
        # TRAIN MODEL
        # =========================
        model = train_model(
            X_train,
            y_train
        )

        # =========================
        # SAVE MODEL
        # =========================
        save_model(
            model,
            "models/model.pkl"
        )

        print("Model training successful")

    except Exception as e:

        logger.error(
            f"Model building pipeline failed: {e}"
        )

        print(f"Error: {e}")


if __name__ == "__main__":
    main()
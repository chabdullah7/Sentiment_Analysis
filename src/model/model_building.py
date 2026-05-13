import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import logging
from src.logger import logger
import os


def load_data(file_path: str) -> pd.DataFrame:
    """Load processed BOW data from CSV."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train Logistic Regression model (clean + no sklearn warnings)
    """
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


def save_model(model, file_path: str) -> None:
    """Save trained model to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logger.info(f"Model saved to {file_path}")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    try:
        # Load data
        train_data = load_data("./data/processed/train_bow.csv")

        # Split features and labels
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train model
        model = train_model(X_train, y_train)

        # Save model
        save_model(model, "models/model.pkl")

    except Exception as e:
        logger.error(f"Model building pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import os
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging

nltk.download('wordnet')
nltk.download('stopwords')


def preprocess_dataframe(df, col='text'):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        if pd.isna(text):
            return ""

        text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
        text = ''.join([char for char in text if not char.isdigit()])
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = text.replace('؛', "")
        text = re.sub(r'\s+', ' ', text).strip()

        text = " ".join([word for word in text.split() if word not in stop_words])
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

        return text

    df[col] = df[col].apply(preprocess_text)
    df = df.dropna(subset=[col])

    return df


def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        logging.info('Data loaded properly')

        train_processed = preprocess_dataframe(train_data, 'review')
        test_processed = preprocess_dataframe(test_data, 'review')

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        # ONLY ONE LOG HERE (correct place)
        logging.info('Data pre-processing completed')
        logging.info('Processed data saved successfully')

    except Exception as e:
        logging.error(f'Failed preprocessing: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
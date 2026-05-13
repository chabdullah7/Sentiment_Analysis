import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================================================
# DOWNLOADS (run once)
# =========================================================

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# =========================================================
# GLOBALS (load once, fast execution)
# =========================================================

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
DIGIT_PATTERN = re.compile(r"\d+")
PUNCT_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")
SPACE_PATTERN = re.compile(r"\s+")


# =========================================================
# MAIN PREPROCESSING FUNCTION
# =========================================================

def preprocess_text(text: str) -> str:
    """
    Clean text:
    1. lowercase
    2. remove URLs
    3. remove numbers
    4. remove stopwords
    5. remove punctuation
    6. lemmatize
    """

    if not isinstance(text, str):
        return ""

    # lowercase + remove urls
    text = text.lower()
    text = URL_PATTERN.sub("", text)

    # tokenize
    words = text.split()

    cleaned_words = []

    for word in words:

        # remove numbers
        word = DIGIT_PATTERN.sub("", word)

        # skip stopwords or empty words
        if not word or word in STOP_WORDS:
            continue

        # lemmatize
        word = LEMMATIZER.lemmatize(word)

        cleaned_words.append(word)

    text = " ".join(cleaned_words)

    # punctuation cleanup
    text = PUNCT_PATTERN.sub(" ", text)
    text = SPACE_PATTERN.sub(" ", text).strip()

    return text


# =========================================================
# DATA CLEANING UTILITY
# =========================================================

def remove_small_sentences(df, column="text", min_words=3):
    """
    Remove rows where text is too small.
    """

    return (
        df[df[column].astype(str).apply(lambda x: len(x.split()) >= min_words)]
        .reset_index(drop=True)
    )
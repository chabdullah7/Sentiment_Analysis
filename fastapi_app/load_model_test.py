import pickle
from pathlib import Path

# =========================================================
# VECTORIZER PATH
# =========================================================

pkl_file_path = Path("../models/vectorizer.pkl")

# =========================================================
# CHECK + LOAD
# =========================================================

if pkl_file_path.exists():

    print(f"File found: {pkl_file_path}")

    try:

        with open(pkl_file_path, "rb") as f:
            vectorizer = pickle.load(f)

        print("Vectorizer loaded successfully.")

    except Exception as e:

        print(f"Error loading vectorizer: {e}")

else:

    print(f"File not found: {pkl_file_path}")
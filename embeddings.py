import os
import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_DIR = "cache"
MODEL_NAME = "all-MiniLM-L6-v2"

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def load_model():
    ensure_cache_dir()
    return SentenceTransformer(MODEL_NAME)

def encode_list(model, texts, cache_path):
    ensure_cache_dir()

    # If cached file exists, load it
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    print(f"Encoding {len(texts)} items...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save(cache_path, embeddings)
    print(f"Saved embeddings to {cache_path}")

    return embeddings

def prepare_embeddings_for_df(df, text_column, cache_name):
    ensure_cache_dir()

    model = load_model()

    texts = df[text_column].fillna("").astype(str).tolist()

    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npy")

    embeddings = encode_list(model, texts, cache_path)

    return model, embeddings

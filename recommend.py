# recommend.py
import os
import pandas as pd
import numpy as np
from sentence_transformers import util
from embeddings import prepare_embeddings_for_df, load_model

DATA_DIR = "data"


def load_data():
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    books_path = os.path.join(DATA_DIR, "books.csv")

    # ----------------------------------------------------
    # LOAD MOVIES
    # ----------------------------------------------------
    movies = pd.read_csv(movies_path)

    if "title" not in movies.columns or "overview" not in movies.columns:
        raise ValueError("movies.csv must contain 'title' and 'overview' columns")

    movies.dropna(subset=["overview"], inplace=True)

    # ----------------------------------------------------
    # LOAD BOOKS (skip malformed lines)
    # ----------------------------------------------------
    books = pd.read_csv(
        books_path,
        on_bad_lines="skip",  # skip dirty rows
        engine="python"       # more robust CSV parser
    )

    if "title" not in books.columns or "authors" not in books.columns:
        raise ValueError("books.csv must contain 'title' and 'authors' columns")

    # Create synthetic description
    books["description"] = (
        books["title"].fillna("") +
        " by " +
        books["authors"].fillna("")
    )

    books.dropna(subset=["description"], inplace=True)

    return movies, books


class Recommender:
    def __init__(self):
        self.movies_df, self.books_df = load_data()
        self.model = load_model()

        # Create embeddings
        _, self.movie_embeddings = prepare_embeddings_for_df(
            self.movies_df, "overview", "movies_embeddings"
        )
        _, self.book_embeddings = prepare_embeddings_for_df(
            self.books_df, "description", "books_embeddings"
        )

    # --------------------------------------------------------
    # RECOMMENDATIONS BASED ON TITLE
    # --------------------------------------------------------
    def recommend_by_title(self, title, mode="movie", top_k=5):
        if mode == "movie":
            df = self.movies_df
            embeddings = self.movie_embeddings
            text_col = "overview"
        else:
            df = self.books_df
            embeddings = self.book_embeddings
            text_col = "description"

        match = df[df["title"].str.lower() == title.lower()]
        if match.empty:
            raise ValueError(f"'{title}' not found in {mode} dataset")

        idx = match.index[0]
        query_vec = embeddings[idx]
        scores = util.cos_sim(query_vec, embeddings)[0]

        # Get top matches
        top = np.argpartition(-scores, range(top_k + 1))[:top_k + 1]
        top = top[np.argsort(-scores[top])]

        results = []
        for i in top:
            i = int(i)  # FIX: Ensure Python int
            if i == idx:
                continue
            results.append({
                "title": df.iloc[i]["title"],
                "overview": df.iloc[i][text_col],
                "score": float(scores[i])
            })
            if len(results) == top_k:
                break

        return results

    # --------------------------------------------------------
    # RECOMMENDATIONS BASED ON FREE TEXT QUERY
    # --------------------------------------------------------
    def recommend_by_text(self, query, mode="movie", top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True)[0]

        if mode == "movie":
            df = self.movies_df
            embeddings = self.movie_embeddings
            text_col = "overview"
        else:
            df = self.books_df
            embeddings = self.book_embeddings
            text_col = "description"

        scores = util.cos_sim(query_vec, embeddings)[0]

        top = np.argpartition(-scores, range(top_k))[:top_k]
        top = top[np.argsort(-scores[top])]

        results = []
        for i in top:
            i = int(i)  # FIX: Ensure Python int
            results.append({
                "title": df.iloc[i]["title"],
                "overview": df.iloc[i][text_col],
                "score": float(scores[i])
            })

        return results

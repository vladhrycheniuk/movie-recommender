from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import random

app = FastAPI(title="Movie Recommender API")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для локальної розробки
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Завантаження даних =====
movies_path = "data/raw/ml-100k/u.item"
ratings_path = "data/raw/ml-100k/u.data"

movies_df = pd.read_csv(
    movies_path,
    sep="|",
    header=None,
    encoding="latin-1",
    usecols=[0, 1],
    names=["movie_id", "title"],
)

ratings_df = pd.read_csv(
    ratings_path,
    sep="\t",
    header=None,
    names=["user_id", "movie_id", "rating", "timestamp"],
)

# ===== ROUTES =====
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/search")
def search_movies(query: str = None):
    if not query or query.strip() == "":
        return []
    results = movies_df[movies_df["title"].str.contains(query, case=False, na=False)]
    return results.head(10).to_dict(orient="records")


@app.get("/recommend/{movie_id}")
def recommend_movies(movie_id: int, top_n: int = 5):
    # Простий приклад: топ N фільмів із найвищим середнім рейтингом
    movie_ratings = ratings_df.groupby("movie_id")["rating"].mean()
    top_movies = movie_ratings.sort_values(ascending=False).head(top_n).index
    results = movies_df[movies_df["movie_id"].isin(top_movies)]
    return results.to_dict(orient="records")


@app.get("/popular")
def popular_movies():
    """Повертає 10 випадкових фільмів для головної сторінки"""
    sample = movies_df.sample(10)
    return sample.to_dict(orient="records")

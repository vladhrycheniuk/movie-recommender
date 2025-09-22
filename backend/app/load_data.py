# backend/app/load_data.py
import pandas as pd
from pathlib import Path

# шлях: D:\movie-recommender\data\raw\ml-100k
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ml-100k"

# Жанри в MovieLens 100k (u.item містить 19 genre-flag стовпців). Подробиці — README ml-100k. :contentReference[oaicite:1]{index=1}
GENRE_NAMES = [
    "unknown","Action","Adventure","Animation","Childrens","Comedy","Crime","Documentary",
    "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
    "Thriller","War","Western"
]

def load_movies():
    movies_path = DATA_PATH / "u.item"
    if not movies_path.exists():
        raise FileNotFoundError(f"Movie file not found: {movies_path}. Розмісти u.item тут.")
    # Коли читаємо, даємо імена для всіх колонок (movie_id, title, release_date, video_release_date, imdb_url, <genres...>)
    col_names = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_NAMES
    movies = pd.read_csv(movies_path, sep="|", encoding="latin-1", header=None, names=col_names, index_col=False)

    # Збираємо жанри як pipe-separated string
    def _make_genres(row):
        g = [name for name in GENRE_NAMES if int(row.get(name, 0)) == 1]
        return "|".join(g)

    movies["genres"] = movies.apply(_make_genres, axis=1)

    # Очистка title (видалити рік в дужках якщо є)
    movies["title"] = movies["title"].astype(str).str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

    # Додатково витягуємо рік з release_date якщо є
    movies["year"] = movies["release_date"].astype(str).str.extract(r"(\d{4})")
    # Перетворимо movie_id в int
    movies["movie_id"] = movies["movie_id"].astype(int)

    # Залишаємо корисні колонки
    movies = movies[["movie_id", "title", "year", "genres", "release_date", "imdb_url"]]
    return movies

def load_ratings():
    ratings_path = DATA_PATH / "u.data"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}. Розмісти u.data тут.")
    ratings = pd.read_csv(ratings_path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"], index_col=False)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)
    return ratings

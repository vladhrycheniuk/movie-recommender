from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
from app.load_data import load_movies, load_ratings

# --- FastAPI ---
app = FastAPI(title="Movie Recommender API")

# Дозволяємо запити з будь-якого джерела (для локального фронтенду)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можна замінити на ["http://127.0.0.1:5500"] для Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Завантаження даних ---
movies_df = load_movies()
ratings_df = load_ratings()

# Середній рейтинг кожного фільму
mean_ratings = ratings_df.groupby('movie_id')['rating'].mean().reset_index()
movies_df = movies_df.merge(mean_ratings, on='movie_id', how='left')
movies_df['rating'] = movies_df['rating'].fillna(0)

# Переконайся, що є поле genres (якщо його немає, додаємо пусті рядки)
if 'genres' not in movies_df.columns:
    movies_df['genres'] = ''

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/search")
def search(q: Optional[str] = None, min_rating: float = 0.0, limit: int = 10):
    results = movies_df.copy()
    if q:
        ql = q.lower()
        results = results[results['title'].str.lower().str.contains(ql)]
    results = results[results['rating'] >= min_rating]
    results = results.sort_values(by='rating', ascending=False)
    return results.head(limit).to_dict(orient='records')

@app.get("/recommend")
def recommend(movie_id: Optional[int] = None, top_n: int = 10):
    results = movies_df.copy()

    if movie_id:
        if movie_id not in results['movie_id'].values:
            return {"error": "Movie ID not found"}
        # Витягуємо жанри обраного фільму
        movie_genres = results.loc[results['movie_id'] == movie_id, 'genres'].values[0].split('|')
        # Обчислюємо overlap жанрів
        results['genre_overlap'] = results['genres'].apply(lambda g: len(set(g.split('|')) & set(movie_genres)))
        results = results[results['genre_overlap'] > 0]
        results = results.sort_values(by=['genre_overlap', 'rating'], ascending=[False, False])
    else:
        # якщо movie_id не передано, повертаємо топ за рейтингом
        results = results.sort_values(by='rating', ascending=False)

    return results.head(top_n).to_dict(orient='records')

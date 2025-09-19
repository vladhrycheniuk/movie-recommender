from fastapi import FastAPI
from typing import Optional
import pandas as pd
from app.load_data import load_movies, load_ratings

app = FastAPI(title="Movie Recommender API")

# завантаження даних при старті
movies_df = load_movies()
ratings_df = load_ratings()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/search")
def search(q: Optional[str] = None, min_rating: float = 0.0, limit: int = 10):
    results = movies_df.copy()

    if q:
        ql = q.lower()
        results = results[results['title'].str.lower().str.contains(ql)]

    mean_ratings = ratings_df.groupby('movie_id')['rating'].mean().reset_index()
    results = results.merge(mean_ratings, on='movie_id', how='left')
    results['rating'] = results['rating'].fillna(0)

    results = results[results['rating'] >= min_rating]
    results = results.sort_values(by='rating', ascending=False)

    return results.head(limit).to_dict(orient='records')

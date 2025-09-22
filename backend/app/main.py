# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
from app.load_data import load_movies, load_ratings

app = FastAPI(title="Movie Recommender API")

# Allow local frontend (change allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для локальної розробки; у продакшн вкажи конкретні origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- load data on startup ---
movies_df = load_movies()
ratings_df = load_ratings()

# Compute mean rating and rating count
rating_stats = ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index().rename(columns={"mean":"rating", "count":"rating_count"})
movies_df = movies_df.merge(rating_stats, on="movie_id", how="left")
movies_df["rating"] = movies_df["rating"].fillna(0)
movies_df["rating_count"] = movies_df["rating_count"].fillna(0).astype(int)

# Precompute set of all genres
_all_genres = set()
for gs in movies_df["genres"].astype(str).values:
    if gs:
        _all_genres.update([g for g in gs.split("|") if g])
_all_genres = sorted([g for g in _all_genres if g])

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/genres")
def genres():
    """Повертає список жанрів (для dropdown на фронтенді)."""
    return _all_genres

@app.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    rec = movies_df[movies_df["movie_id"] == movie_id]
    if rec.empty:
        return {"error": "Movie ID not found"}
    return rec.iloc[0].to_dict()

@app.get("/search")
def search(
    q: Optional[str] = None,
    min_rating: float = 0.0,
    genre: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    sort_by: Optional[str] = "rating",
    limit: int = 50
):
    results = movies_df.copy()

    if q:
        ql = q.lower()
        results = results[results["title"].str.lower().str.contains(ql, na=False)]

    if genre:
        results = results[results["genres"].str.contains(genre, na=False)]

    if year_from:
        results = results[pd.to_numeric(results["year"], errors="coerce") >= int(year_from)]
    if year_to:
        results = results[pd.to_numeric(results["year"], errors="coerce") <= int(year_to)]

    # min_rating on scale of 1-5 (MovieLens)
    if min_rating and min_rating > 0:
        results = results[results["rating"] >= float(min_rating)]

    if sort_by == "rating":
        results = results.sort_values(by=["rating", "rating_count"], ascending=[False, False])
    elif sort_by == "year":
        results = results.sort_values(by="year", ascending=False)
    else:
        results = results.sort_values(by=["rating", "rating_count"], ascending=[False, False])

    return results.head(limit).to_dict(orient="records")

@app.get("/recommend")
def recommend(movie_id: Optional[int] = None, top_n: int = 10):
    results = movies_df.copy()
    if movie_id:
        if movie_id not in results["movie_id"].values:
            return {"error": "Movie ID not found"}
        base_genres = set(str(results.loc[results["movie_id"] == movie_id, "genres"].values[0]).split("|"))
        def overlap_score(genres_str):
            gs = set([g for g in str(genres_str).split("|") if g])
            # overlap count
            return len(base_genres & gs)
        results["genre_overlap"] = results["genres"].apply(overlap_score)
        # remove the movie itself
        results = results[results["movie_id"] != movie_id]
        results = results[results["genre_overlap"] > 0]
        # sort by overlap, then rating_count, then rating
        results = results.sort_values(by=["genre_overlap", "rating_count", "rating"], ascending=[False, False, False])
    else:
        results = results.sort_values(by=["rating", "rating_count"], ascending=[False, False])
    return results.head(top_n).to_dict(orient="records")

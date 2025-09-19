import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "ml-100k"

def load_movies():
    movies_path = DATA_PATH / "u.item"
    movies = pd.read_csv(
        movies_path,
        sep='|',
        encoding='latin-1',
        header=None,
        usecols=[0,1,2],
        names=['movie_id','title','year']
    )
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')[0]
    movies['title'] = movies['title'].str.replace(r'\(\d{4}\)','').str.strip()
    return movies

def load_ratings():
    ratings_path = DATA_PATH / "u.data"
    ratings = pd.read_csv(
        ratings_path,
        sep='\t',
        names=['user_id','movie_id','rating','timestamp']
    )
    return ratings

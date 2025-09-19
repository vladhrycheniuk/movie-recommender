# Movie Recommender

Навчальний проєкт — система рекомендацій фільмів (IMDb/MovieLens + ML).

## Функціонал
- Пошук і фільтрація фільмів
- Content-based рекомендації (схожі фільми)
- Collaborative filtering (ALS/SVD)
- Гібридні рекомендації
- Веб-застосунок (FastAPI + React)

## Структура
- `backend/` — FastAPI сервер, ML-моделі
- `frontend/` — React UI
- `data/` — сирі та оброблені дані
- `notebooks/` — дослідження, EDA, прототипи
- `infra/` — Docker, деплой

## Як запустити (на першому етапі)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

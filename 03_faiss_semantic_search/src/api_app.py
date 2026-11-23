from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.semantic_search import predict_masked_tokens, search_similar_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class TokenPredictionRequest(BaseModel):
    sentence: str
    top_n: int = 5


api_app = FastAPI(title="FAISS Semantic Search & Transformers")
api_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def handle_index_get() -> FileResponse:
    """Повертає головну сторінку index.html."""
    return FileResponse(STATIC_DIR / "index.html")


def handle_search_post(payload: SearchRequest) -> dict:
    """Виконує семантичний пошук у FAISS та повертає токени і знайдені документи."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Поле 'query' не може бути порожнім.")
    return search_similar_documents(query_text=payload.query, top_k=payload.top_k)


def handle_predict_tokens_post(payload: TokenPredictionRequest) -> dict:
    """Передбачає ймовірні токени для позиції [MASK] через Hugging Face Transformers."""
    if not payload.sentence.strip():
        raise HTTPException(status_code=400, detail="Поле 'sentence' не може бути порожнім.")
    return predict_masked_tokens(masked_sentence=payload.sentence, top_n=payload.top_n)


api_app.add_api_route("/", handle_index_get, methods=["GET"])
api_app.add_api_route("/api/search", handle_search_post, methods=["POST"])
api_app.add_api_route("/api/predict-tokens", handle_predict_tokens_post, methods=["POST"])
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.nlp_pipeline import run_nlp_task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

# Pydantic-модель використовується лише як схема валідації вхідного JSON запиту FastAPI
class ProcessRequestPayload(BaseModel):
    task: str = "auto_route"
    text: str
    context: str = ""


api_app = FastAPI(title="Ukrainian Text Processing Pipeline")
api_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def serve_index_page() -> FileResponse:
    """Повертає головну HTML-сторінку веб-інтерфейсу."""
    return FileResponse(STATIC_DIR / "index.html")


def process_text_endpoint(payload: ProcessRequestPayload) -> dict:
    """Обробляє запит користувача та повертає результат обраної NLP-задачі."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Поле 'text' не може бути порожнім.")

    try:
        return run_nlp_task(task_name=payload.task, text=payload.text, context=payload.context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


api_app.add_api_route("/", serve_index_page, methods=["GET"])
api_app.add_api_route("/api/process", process_text_endpoint, methods=["POST"])
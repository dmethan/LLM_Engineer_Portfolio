import json
import re
from pathlib import Path
import joblib
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.dataset_prep import clean_input_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Кеш завантажених моделей у пам'яті процесу, щоб не завантажувати їх при кожному запиті
PIPELINE_CACHE = {
    "device": None,
    "baseline_bundle": None,
    "cls_tokenizer": None,
    "cls_model": None,
    "gen_tokenizer": None,
    "gen_model": None,
}


def get_runtime_device() -> torch.device:
    """Повертає GPU (CUDA), якщо доступний, або CPU."""
    if PIPELINE_CACHE["device"] is None:
        PIPELINE_CACHE["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PIPELINE_CACHE["device"]


def load_relevance_classifier() -> tuple[str, object, object]:
    """Завантажує збережений DistilBERT для класифікації релевантності або запасний Baseline."""
    distilbert_dir = SAVED_MODELS_DIR / "distilbert_relevance"
    baseline_path = SAVED_MODELS_DIR / "baseline_model.joblib"

    if distilbert_dir.exists():
        if PIPELINE_CACHE["cls_model"] is None:
            device = get_runtime_device()
            PIPELINE_CACHE["cls_tokenizer"] = AutoTokenizer.from_pretrained(distilbert_dir)
            model = AutoModelForSequenceClassification.from_pretrained(distilbert_dir)
            model.to(device)
            model.eval()
            PIPELINE_CACHE["cls_model"] = model
        return "transformer_distilbert", PIPELINE_CACHE["cls_tokenizer"], PIPELINE_CACHE["cls_model"]

    if baseline_path.exists():
        if PIPELINE_CACHE["baseline_bundle"] is None:
            PIPELINE_CACHE["baseline_bundle"] = joblib.load(baseline_path)
        return "baseline_tfidf_logreg", PIPELINE_CACHE["baseline_bundle"], None

    raise FileNotFoundError(
        "Моделі класифікації не знайдено. Спочатку запустіть: python -m src.train_classifiers"
    )


def load_text_generation_llm() -> tuple[object, object, torch.device]:
    """Завантажує мовну модель для генерації наративу, JSON-подій, routing та QA."""
    device = get_runtime_device()
    if PIPELINE_CACHE["gen_model"] is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        PIPELINE_CACHE["gen_tokenizer"] = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        PIPELINE_CACHE["gen_model"] = model

    return PIPELINE_CACHE["gen_tokenizer"], PIPELINE_CACHE["gen_model"], device


def generate_llm_response(prompt_text: str, max_new_tokens: int = 256) -> str:
    """Виконує генерацію відповіді мовною моделлю за заданим промптом."""
    tokenizer, model, device = load_text_generation_llm()
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([chat_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_only = output_ids[0][model_inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated_only, skip_special_tokens=True).strip()


def predict_relevance(text: str) -> dict:
    """Задача 1: Бінарна класифікація релевантності тексту заданій тематиці (IT / інциденти / релізи)."""
    cleaned = clean_input_text(text)
    if not cleaned:
        return {"is_relevant": False, "label": 0, "confidence": 0.0, "model_used": "none"}

    backend_name, primary_obj, secondary_obj = load_relevance_classifier()

    if backend_name == "transformer_distilbert":
        tokenizer = primary_obj
        model = secondary_obj
        device = get_runtime_device()

        encoded = tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            predicted_label = int(torch.argmax(probabilities).item())
            confidence = round(float(probabilities[predicted_label].item()), 4)

        return {
            "is_relevant": predicted_label == 1,
            "label": predicted_label,
            "confidence": confidence,
            "model_used": "distilbert-base-multilingual-cased",
        }

    # Запасний варіант: Baseline TF-IDF + LogisticRegression
    vectorizer = primary_obj["vectorizer"]
    classifier = primary_obj["model"]
    features = vectorizer.transform([cleaned])
    predicted_label = int(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    confidence = round(float(probabilities[predicted_label]), 4)

    return {
        "is_relevant": predicted_label == 1,
        "label": predicted_label,
        "confidence": confidence,
        "model_used": "baseline_tfidf_logreg",
    }


def extract_event_json(text: str) -> dict:
    """Задача 2: Класифікація події за категоріями та формування валідного JSON українською мовою."""
    cleaned = clean_input_text(text)
    prompt = (
        "Проаналізуй вхідне повідомлення (може бути українською або російською, з помилками та скороченнями). "
        "Поверни ВИКЛЮЧНО валідний JSON-об'єкт без пояснень та без ```json.\n"
        "Поля JSON:\n"
        "- 'category': одне з ['інцидент', 'кібератака', 'реліз', 'інше']\n"
        "- 'severity': одне з ['низька', 'середня', 'висока']\n"
        "- 'summary_uk': короткий опис події українською мовою\n\n"
        "Приклад:\n"
        "Вхід: 'БД легла, 500 ошибка на сервере'\n"
        'Вихід: {"category": "інцидент", "severity": "висока", "summary_uk": "Збій бази даних та поява помилки 500 на сервері."}\n\n'
        f"Вхід: '{cleaned}'\n"
        "Вихід: "
    )

    raw_response = generate_llm_response(prompt, max_new_tokens=200)
    without_fences = re.sub(r"```json|```", "", raw_response).strip()
    start_idx = without_fences.find("{")
    end_idx = without_fences.rfind("}")

    if start_idx != -1 and end_idx != -1:
        json_candidate = without_fences[start_idx : end_idx + 1]
        try:
            parsed_object = json.loads(json_candidate)
            if isinstance(parsed_object, dict):
                return {
                    "is_valid_json": True,
                    "event_data": parsed_object,
                    "raw_model_output": raw_response,
                }
        except json.JSONDecodeError:
            pass

    return {
        "is_valid_json": False,
        "event_data": {
            "category": "інше",
            "severity": "середня",
            "summary_uk": cleaned,
        },
        "raw_model_output": raw_response,
    }


def generate_narrative_uk(text: str) -> dict:
    """Задача 3: Генерує короткий зв'язний наратив українською мовою на основі вхідного тексту."""
    cleaned = clean_input_text(text)
    prompt = (
        "Напиши короткий, зрозумілий та грамотний наратив (2-3 речення) українською мовою "
        "на основі наведеного нижче тексту. Навіть якщо вхідний текст написаний російською, з помилками або сленгом, "
        "підсумковий текст має бути виключно українською мовою без вигадування нових фактів.\n\n"
        f"Вхідний текст: {cleaned}\n"
        "Короткий наратив українською:"
    )

    narrative_text = generate_llm_response(prompt, max_new_tokens=220)
    return {"narrative_uk": narrative_text}


def answer_from_context(question: str, context: str) -> dict:
    """Задача 5: Відповідає на запитання користувача українською мовою виключно на основі контексту."""
    clean_q = clean_input_text(question)
    clean_ctx = clean_input_text(context)

    if not clean_ctx:
        return {
            "answer_uk": "Будь ласка, надайте текст контексту, на основі якого потрібно дати відповідь.",
        }

    prompt = (
        "Відповідай на запитання користувача українською мовою ВИКЛЮЧНО на основі наданого контексту. "
        "Якщо у контексті немає прямої відповіді, чесно напиши: "
        "'У наданому контексті немає достатньої інформації для відповіді.' Не вигадуй фактів.\n\n"
        f"Контекст:\n{clean_ctx}\n\n"
        f"Запитання: {clean_q}\n"
        "Відповідь українською:"
    )

    answer_text = generate_llm_response(prompt, max_new_tokens=250)
    return {"answer_uk": answer_text}


def detect_route_task(text: str, context: str = "") -> str:
    """Визначає сценарій обробки запиту користувача (routing)."""
    if context and context.strip():
        return "qa"

    lower_text = text.lower()
    if any(word in lower_text for word in ["json", "категор", "класифік", "структур"]):
        return "event_json"
    if any(word in lower_text for word in ["переказ", "наратив", "коротко", "резюме", "опиши"]):
        return "narrative"
    return "relevance"


def run_nlp_task(task_name: str, text: str, context: str = "") -> dict:
    """Задача 4 (Routing + єдина точка входу): запускає обрану задачу або автоматично обирає її."""
    selected_task = task_name.strip().lower()
    routed_automatically = False

    if selected_task == "auto_route":
        selected_task = detect_route_task(text, context)
        routed_automatically = True

    if selected_task == "relevance":
        result_payload = predict_relevance(text)
    elif selected_task == "event_json":
        result_payload = extract_event_json(text)
    elif selected_task == "narrative":
        result_payload = generate_narrative_uk(text)
    elif selected_task == "qa":
        result_payload = answer_from_context(question=text, context=context)
    else:
        raise ValueError(f"Непідтримуваний тип задачі: {selected_task}")

    return {
        "executed_task": selected_task,
        "routed_automatically": routed_automatically,
        "result": result_payload,
    }
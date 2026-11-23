import json
from pathlib import Path
import faiss
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from src.build_index import (
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
    build_and_save_faiss_index,
    compute_mean_pooled_embeddings,
    get_compute_device,
)

MASKED_LM_MODEL_NAME = "distilbert-base-uncased"

SEARCH_CACHE = {
    "faiss_index": None,
    "documents": None,
    "embed_tokenizer": None,
    "embed_model": None,
    "mlm_tokenizer": None,
    "mlm_model": None,
}


def load_search_resources() -> tuple:
    """Завантажує індекс FAISS, документи та модель ембедингів (з кешуванням у пам'яті)."""
    documents_path = DATA_DIR / "documents.json"
    index_path = FAISS_INDEX_DIR / "documents.index"

    if not documents_path.exists() or not index_path.exists():
        build_and_save_faiss_index(sample_count=400, force_rebuild=False)

    if SEARCH_CACHE["faiss_index"] is None:
        SEARCH_CACHE["faiss_index"] = faiss.read_index(str(index_path))

    if SEARCH_CACHE["documents"] is None:
        SEARCH_CACHE["documents"] = json.loads(documents_path.read_text(encoding="utf-8"))

    if SEARCH_CACHE["embed_model"] is None:
        device = get_compute_device()
        SEARCH_CACHE["embed_tokenizer"] = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
        embed_model.eval()
        SEARCH_CACHE["embed_model"] = embed_model

    return (
        SEARCH_CACHE["faiss_index"],
        SEARCH_CACHE["documents"],
        SEARCH_CACHE["embed_tokenizer"],
        SEARCH_CACHE["embed_model"],
    )


def inspect_tokenization(text: str) -> dict:
    """Повертає токени та їхні числові ідентифікатори (input_ids) для наочного перегляду."""
    _, _, tokenizer, _ = load_search_resources()
    encoded = tokenizer(text, truncation=True, max_length=64)
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "tokens_count": len(tokens),
        "tokens": tokens,
        "input_ids": token_ids,
    }


def search_similar_documents(query_text: str, top_k: int = 5) -> dict:
    """Виконує семантичний пошук у FAISS за запитом користувача."""
    clean_query = query_text.strip()
    if not clean_query:
        return {
            "query": "",
            "tokenization": {"tokens_count": 0, "tokens": [], "input_ids": []},
            "embedding_dim": 0,
            "results": [],
        }

    faiss_index, documents, embed_tokenizer, embed_model = load_search_resources()
    device = get_compute_device()

    query_vector = compute_mean_pooled_embeddings(
        texts=[clean_query],
        tokenizer=embed_tokenizer,
        model=embed_model,
        device=device,
        batch_size=1,
    )

    safe_top_k = max(1, min(int(top_k), len(documents)))
    similarity_scores, matched_indices = faiss_index.search(query_vector, safe_top_k)

    search_results = []
    for rank_idx, (doc_idx, score) in enumerate(
        zip(matched_indices[0], similarity_scores[0]),
        start=1,
    ):
        if int(doc_idx) < 0:
            continue
        doc_item = documents[int(doc_idx)]
        search_results.append(
            {
                "rank": rank_idx,
                "doc_id": doc_item["doc_id"],
                "category": doc_item["category"],
                "score": round(float(score), 4),
                "text": doc_item["text"],
            }
        )

    token_info = inspect_tokenization(clean_query)

    return {
        "query": clean_query,
        "model_name": EMBEDDING_MODEL_NAME,
        "embedding_dim": int(query_vector.shape[1]),
        "tokenization": token_info,
        "results": search_results,
    }


def predict_masked_tokens(masked_sentence: str, top_n: int = 5) -> dict:
    """
    Демонструє передбачення токенів через Masked Language Model (AutoModelForMaskedLM).
    Якщо у тексті немає токена [MASK], додає його в кінець речення.
    """
    clean_sentence = masked_sentence.strip()
    if not clean_sentence:
        return {"sentence": "", "predictions": []}

    if SEARCH_CACHE["mlm_model"] is None:
        device = get_compute_device()
        SEARCH_CACHE["mlm_tokenizer"] = AutoTokenizer.from_pretrained(MASKED_LM_MODEL_NAME)
        mlm_model = AutoModelForMaskedLM.from_pretrained(MASKED_LM_MODEL_NAME).to(device)
        mlm_model.eval()
        SEARCH_CACHE["mlm_model"] = mlm_model

    mlm_tokenizer = SEARCH_CACHE["mlm_tokenizer"]
    mlm_model = SEARCH_CACHE["mlm_model"]
    device = get_compute_device()

    sentence_with_mask = (
        clean_sentence if "[MASK]" in clean_sentence else f"{clean_sentence} [MASK]."
    )
    encoded_inputs = mlm_tokenizer(sentence_with_mask, return_tensors="pt").to(device)
    mask_token_id = mlm_tokenizer.mask_token_id
    input_id_list = encoded_inputs["input_ids"][0].tolist()

    if mask_token_id not in input_id_list:
        return {
            "sentence": sentence_with_mask,
            "model_name": MASKED_LM_MODEL_NAME,
            "predictions": [],
        }

    mask_position = input_id_list.index(mask_token_id)

    with torch.no_grad():
        logits = mlm_model(**encoded_inputs).logits
        mask_token_logits = logits[0, mask_position, :]
        probabilities = torch.softmax(mask_token_logits, dim=-1)
        top_probs, top_ids = torch.topk(probabilities, k=max(1, min(int(top_n), 10)))

    predictions = []
    for token_id, prob_value in zip(top_ids.tolist(), top_probs.tolist()):
        predicted_token = mlm_tokenizer.decode([token_id]).strip()
        completed_text = sentence_with_mask.replace("[MASK]", predicted_token, 1)
        predictions.append(
            {
                "token": predicted_token,
                "probability": round(float(prob_value), 4),
                "completed_text": completed_text,
            }
        )

    return {
        "sentence": sentence_with_mask,
        "model_name": MASKED_LM_MODEL_NAME,
        "predictions": predictions,
    }
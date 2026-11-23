import json
from pathlib import Path
import faiss
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CATEGORY_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def get_compute_device() -> torch.device:
    """Повертає NVIDIA CUDA GPU, якщо доступний, інакше CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_mean_pooled_embeddings(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Пропускає список текстів через токенізатор та модель Hugging Face Transformers:
    1) токенізація (input_ids, attention_mask),
    2) прямий прохід через трансформер (last_hidden_state),
    3) mean pooling з урахуванням attention_mask,
    4) L2-нормалізація векторів для косинусної подібності у FAISS.
    """
    model.eval()
    all_vectors = []

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx : start_idx + batch_size]

        encoded_batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            model_outputs = model(**encoded_batch)
            token_embeddings = model_outputs.last_hidden_state

        # Розширюємо attention_mask до розмірності [batch_size, seq_len, hidden_size]
        attention_mask_expanded = (
            encoded_batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        summed_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
        token_counts = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = summed_embeddings / token_counts

        # Нормалізуємо вектор по довжині (L2 norm), щоб скалярний добуток дорівнював cosine similarity
        normalized_vectors = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        all_vectors.append(normalized_vectors.cpu().numpy().astype(np.float32))

    return np.vstack(all_vectors)


def load_ag_news_subset(sample_count: int = 400) -> list[dict]:
    """Завантажує відкритий датасет ag_news та формує список документів із метаданими."""
    raw_dataset = load_dataset("ag_news", split="train")
    subset = raw_dataset.select(range(min(sample_count, len(raw_dataset))))

    documents = []
    for doc_index, item in enumerate(subset):
        label_id = int(item["label"])
        documents.append(
            {
                "doc_id": doc_index,
                "category": CATEGORY_NAMES.get(label_id, "Other"),
                "text": item["text"].strip(),
            }
        )

    return documents


def build_and_save_faiss_index(sample_count: int = 400, force_rebuild: bool = False) -> dict:
    """Формує датасет, обчислює ембединги через Hugging Face і зберігає індекс FAISS."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    documents_json_path = DATA_DIR / "documents.json"
    faiss_index_path = FAISS_INDEX_DIR / "documents.index"

    if not force_rebuild and documents_json_path.exists() and faiss_index_path.exists():
        existing_docs = json.loads(documents_json_path.read_text(encoding="utf-8"))
        return {
            "status": "loaded_existing",
            "documents_count": len(existing_docs),
            "model_name": EMBEDDING_MODEL_NAME,
            "index_path": str(faiss_index_path),
        }

    documents = load_ag_news_subset(sample_count=sample_count)
    texts_to_embed = [doc["text"] for doc in documents]

    device = get_compute_device()
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)

    embeddings_matrix = compute_mean_pooled_embeddings(
        texts=texts_to_embed,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=32,
    )

    vector_dimension = embeddings_matrix.shape[1]
    faiss_index = faiss.IndexFlatIP(vector_dimension)
    faiss_index.add(embeddings_matrix)

    faiss.write_index(faiss_index, str(faiss_index_path))
    documents_json_path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "built_new",
        "documents_count": len(documents),
        "vector_dimension": int(vector_dimension),
        "device": str(device),
        "model_name": EMBEDDING_MODEL_NAME,
        "index_path": str(faiss_index_path),
    }


if __name__ == "__main__":
    summary = build_and_save_faiss_index(sample_count=400, force_rebuild=True)
    print("Результат побудови індексу FAISS:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
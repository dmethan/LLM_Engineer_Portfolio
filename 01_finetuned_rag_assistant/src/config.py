import os
from pathlib import Path
from dotenv import load_dotenv

# Корінь першого проєкту: .../01_finetuned_rag_assistant
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Корінь спільного репозиторію, де лежить .env
REPO_ROOT = PROJECT_ROOT.parent


def load_project_config() -> dict:
    """Завантажує налаштування з кореневого .env та формує шляхи всередині 01_finetuned_rag_assistant."""
    root_env_path = REPO_ROOT / ".env"
    if root_env_path.exists():
        load_dotenv(dotenv_path=root_env_path)
    else:
        load_dotenv()

    hf_username = os.getenv("HF_USERNAME", "dmethan").strip()
    finetuned_repo_name = os.getenv("FINETUNED_REPO_NAME", "qwen2.5-1.5b-ds-rnd-lab-lora").strip()

    config = {
        "app_title": os.getenv("APP_TITLE", "Data Science R&D Laboratory — AI Assistant").strip(),
        "hf_token": os.getenv("HF_TOKEN", "").strip(),
        "hf_username": hf_username,
        "finetuned_repo_id": f"{hf_username}/{finetuned_repo_name}",
        "base_llm_model_name": os.getenv("BASE_LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct").strip(),
        "embedding_model_name": os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base").strip(),
        "knowledge_base_dir": PROJECT_ROOT / "knowledge_base",
        "dataset_output_path": PROJECT_ROOT / "dataset" / "train_dataset.jsonl",
        "chroma_db_dir": PROJECT_ROOT / "chroma_db",
        "chroma_collection_name": os.getenv("CHROMA_COLLECTION_NAME", "ds_rnd_lab_collection").strip(),
        "local_adapter_dir": PROJECT_ROOT / "lora_adapter_output",
        "chunk_size": int(os.getenv("CHUNK_SIZE", "800")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "150")),
        "top_k": int(os.getenv("TOP_K", "4")),
        "min_similarity_score": float(os.getenv("MIN_SIMILARITY_SCORE", "0.35")),
        "server_host": os.getenv("SERVER_HOST", "0.0.0.0").strip(),
        "server_port": int(os.getenv("SERVER_PORT", "7860")),
    }

    config["knowledge_base_dir"].mkdir(parents=True, exist_ok=True)
    config["dataset_output_path"].parent.mkdir(parents=True, exist_ok=True)
    config["chroma_db_dir"].mkdir(parents=True, exist_ok=True)
    config["local_adapter_dir"].mkdir(parents=True, exist_ok=True)

    return config
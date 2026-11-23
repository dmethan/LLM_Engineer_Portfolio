# LLM Engineering Portfolio

Навчально-практичний репозиторій із розробки рішень на базі Large Language Models (LLM), Fine-Tuning (LoRA), Retrieval-Augmented Generation (RAG), семантичного пошуку з **FAISS** та NLP-пайплайнів на **PyTorch** і **Hugging Face Transformers**.

Усі проєкти побудовано у **процедурному стилі** (звичайні функції `def`, без зайвих абстракцій) із зрозумілою архітектурою для демонстрації на технічній співбесіді.

---

## 📦 Проєкти в репозиторії

| Папка | Проєкт | Ключові технології та задачі | Інтерфейс |
|---|---|---|---|
| `01_finetuned_rag_assistant/` | **Data Science R&D Laboratory AI Assistant** | LoRA Fine-Tuning (`Qwen/Qwen2.5-1.5B-Instruct` на Hugging Face Hub), RAG (`ChromaDB` + `intfloat/multilingual-e5-base`), оцінка якості | `Gradio` |
| `02_ukrainian_nlp_pipeline/` | **Ukrainian Text Processing Pipeline** | Порівняння `TF-IDF + LogReg` vs `DistilBERT`, 3 варіанти промптів (`zero_shot`, `strict_schema`, `few_shot`), конвеєр із 5 задач (релевантність, JSON-події, наратив, routing, QA) | `FastAPI` + HTML/CSS/JS |
| `03_faiss_semantic_search/` | **FAISS Semantic Search & Transformers** | Токенізація (`input_ids`, `attention_mask`), mean pooling ембединги (`all-MiniLM-L6-v2`), індекс `FAISS` (`IndexFlatIP`), передбачення токенів (`distilbert-base-uncased`) | `FastAPI` + HTML/CSS/JS |

---

## 📁 Структура репозиторію

```text
LLM_Engineering_Portfolio/
├── .env.example                 # Шаблон змінних середовища (копіюється у локальний .env)
├── .gitignore                   # Виключення віртуальних середовищ, секретів, кешів та важких артефактів
├── requirements.txt             # Спільні залежності Python
├── README.md                    # Загальний опис портфоліо
├── 01_finetuned_rag_assistant/  # Проєкт 1: Fine-Tuned RAG Assistant
├── 02_ukrainian_nlp_pipeline/   # Проєкт 2: Ukrainian Text Processing Pipeline
└── 03_faiss_semantic_search/    # Проєкт 3: FAISS Semantic Search & Transformers
```

---

## 🚀 Швидкий старт

### 1. Створення віртуального середовища

Створіть та активуйте віртуальне середовище `.venv` у корені репозиторію.

### 2. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 3. Налаштування змінних середовища

Скопіюйте `.env.example` у `.env` та заповніть необхідні змінні.

Зокрема, `HF_TOKEN` потрібен для роботи з Hugging Face Hub у першому проєкті.

### 4. Документація проєктів

- [Проєкт 1: Fine-Tuned RAG Assistant](./01_finetuned_rag_assistant/README.md)
- [Проєкт 2: Ukrainian Text Processing Pipeline](./02_ukrainian_nlp_pipeline/README.md)
- [Проєкт 3: FAISS Semantic Search & Transformers](./03_faiss_semantic_search/README.md)

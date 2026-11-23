# 🤖 Data Science R&D Laboratory — Fine-Tuned RAG AI Assistant

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-dmethan%2Fqwen2.5--1.5b--ds--rnd--lab--lora-yellow?style=for-the-badge)](https://huggingface.co/dmethan/qwen2.5-1.5b-ds-rnd-lab-lora)
[![Base LLM](https://img.shields.io/badge/LLM-Qwen2.5--1.5B--Instruct-blueviolet?style=for-the-badge&logo=huggingface)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-PEFT%20LoRA-ff69b4?style=for-the-badge)](https://huggingface.co/docs/peft)
[![RAG Pipeline](https://img.shields.io/badge/RAG-ChromaDB%20%2B%20E5--Base-00b4d8?style=for-the-badge)](https://www.trychroma.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%205-f97316?style=for-the-badge&logo=gradio)](https://www.gradio.app/)

AI-асистент для пошуку та пояснення інформації про веб-сервіс **Data Science Research and Development Laboratory (Vulcan Core)** на базі донавченої через **LoRA** моделі та процедурного **RAG** (`ChromaDB` + `intfloat/multilingual-e5-base`).

![Main Banner](images/project_llm_data_science/banner_llm_assistant.png)

## Демонстрація режимів роботи

### 1. Чат з асистентом та перевірка джерел
![Chat RAG Mode](01_finetuned_rag_assistant/images/project_llm_data_science/mode_1_chat_rag.gif)
![Chat Sources and Guardrails](images/project_llm_data_science/mode_1_chat_sources.png)

### 2. Управління базою знань (ChromaDB)
![Knowledge Base Reindex](01_finetuned_rag_assistant/images/project_llm_data_science/mode_2_knowledge_base.gif)
![ChromaDB Index Status](01_finetuned_rag_assistant/images/project_llm_data_science/mode_2_chroma_stats.png)

### 3. Оцінка якості (Evaluation)
![RAG Evaluation Benchmark](01_finetuned_rag_assistant/images/project_llm_data_science/mode_3_evaluation.gif)
![Evaluation Report Metrics](01_finetuned_rag_assistant/images/project_llm_data_science/mode_3_eval_metrics.png)

### Картка моделі на Hugging Face Hub
![Hugging Face Model Card](01_finetuned_rag_assistant/images/project_llm_data_science/mode_1_chat_sources.png)

## Запуск першого проєкту

Перейдіть у директорію `01_finetuned_rag_assistant` та запустіть потрібний модуль (за умови активованого `.venv` з кореня репозиторію):

```powershell
      cd 01_finetuned_rag_assistant
      
      python -m src.train_and_push
      python -m src.app
```

Після запуску інтерфейс буде доступний за адресою http://127.0.0.1:7860.


---

### Як запускати в PyCharm із кореня репозиторію

Щоб Python у PyCharm бачив пакет `src` усередині `01_finetuned_rag_assistant` без необхідності постійно змінювати робочу папку, у PowerShell можна виконати:

```powershell
$env:PYTHONPATH="01_finetuned_rag_assistant"
python -m src.app
import time
import gradio as gr

from src.config import load_project_config
from src.rag_pipeline import ask_assistant, index_knowledge_base, retrieve_relevant_chunks

EVALUATION_BENCHMARK = [
    {
        "question": "Які три основні режими роботи реалізовані у веб-сервісі Vulcan Core?",
        "expected_source": "architecture_reference.md",
        "expected_keywords": ["basic_mode", "professional_mode", "manual_mode"],
    },
    {
        "question": "Який Docker Hub образ використовується для запуску проєкту та на якому порту він працює?",
        "expected_source": "README.md",
        "expected_keywords": ["dmetan/data_science_rnd_lab", "8000"],
    },
    {
        "question": "У чому полягає дефект BUG-09 у ручному режимі та як його виправити?",
        "expected_source": "architecture_reference.md",
        "expected_keywords": ["Data Leakage", "train_test_split", "X_train"],
    },
    {
        "question": "Як у TEST_PROF_ALGRS реалізовано метод .predict() для трансдуктивних алгоритмів кластеризації?",
        "expected_source": "architecture_reference.md",
        "expected_keywords": ["KNeighborsClassifier", "OPTICS", "SpectralClustering"],
    },
    {
        "question": "Яка базова модель SQL бази даних PostgreSQL налаштована у проєкті?",
        "expected_source": None,
        "expected_keywords": ["немає інформації"],
    },
]


def handle_user_question(question: str) -> tuple[str, str, str, str]:
    """Обробник натискання кнопки 'Запитати' у веб-інтерфейсі Gradio."""
    result = ask_assistant(question)
    status_info = f"Час відповіді: **{result['latency_seconds']} сек** | Активна модель: `{result['model_source']}`"
    return (
        result["answer"],
        result["sources_markdown"],
        status_info,
        result["chunks_markdown"],
    )


def handle_reindex_button() -> str:
    """Переіндексовує базу знань із папки knowledge_base."""
    return index_knowledge_base(force_reindex=True)


def run_rag_evaluation() -> str:
    """Запускає тестування системи на контрольних запитаннях і формує звіт з метриками."""
    total_questions = len(EVALUATION_BENCHMARK)
    retrieval_hits = 0
    answer_hits = 0
    total_latency = 0.0
    report_lines = ["# Звіт оцінювання якості (RAG + Fine-Tuned LLM)\n"]

    for idx, item in enumerate(EVALUATION_BENCHMARK, start=1):
        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_keywords"]

        start_time = time.perf_counter()
        chunks = retrieve_relevant_chunks(question)
        response = ask_assistant(question)
        elapsed = round(time.perf_counter() - start_time, 2)
        total_latency += elapsed

        retrieved_sources = [c["source_filename"] for c in chunks]
        retrieval_ok = (len(chunks) == 0) if expected_source is None else (expected_source in retrieved_sources)

        if retrieval_ok:
            retrieval_hits += 1

        answer_lower = response["answer"].lower()
        keywords_matched = all(kw.lower() in answer_lower for kw in expected_keywords)
        if keywords_matched:
            answer_hits += 1

        report_lines.append(
            f"### {idx}. {question}\n"
            f"- **Retrieval (Recall@K):** {'✅ Влучно' if retrieval_ok else '❌ Промах'} "
            f"(очікувалось: `{expected_source}`, знайдено: `{retrieved_sources}`)\n"
            f"- **Точність фактів у відповіді:** {'✅ Так' if keywords_matched else '⚠️ Частково'}\n"
            f"- **Час виконання:** {elapsed} сек\n"
            f"- **Відповідь моделі:** {response['answer']}\n"
        )

    recall_score = round((retrieval_hits / total_questions) * 100, 1)
    accuracy_score = round((answer_hits / total_questions) * 100, 1)
    avg_latency = round(total_latency / total_questions, 2)

    summary_header = (
        f"## Підсумкові метрики\n"
        f"- **Recall@K (точність пошуку джерела):** {recall_score}% ({retrieval_hits}/{total_questions})\n"
        f"- **Keyword Fact Accuracy (точність фактів):** {accuracy_score}% ({answer_hits}/{total_questions})\n"
        f"- **Середній час відповіді (Latency):** {avg_latency} сек\n\n---\n\n"
    )

    return summary_header + "\n".join(report_lines)


def build_gradio_interface() -> gr.Blocks:
    """Створює простий та зрозумілий веб-інтерфейс Gradio із 3 вкладками."""
    config = load_project_config()

    with gr.Blocks(title=config["app_title"], theme=gr.themes.Soft()) as demo_app:
        gr.Markdown(f"# 🤖 {config['app_title']}")
        gr.Markdown(
            "AI-асистент на базі **донавчаної через LoRA моделі (Hugging Face Hub)** "
            "та **процедурного RAG (ChromaDB + multilingual-e5-base)**."
        )

        with gr.Tab("💬 Чат з асистентом"):
            question_input = gr.Textbox(
                label="Ваше запитання по проєкту Data Science R&D Laboratory",
                placeholder="Наприклад: Які моделі підтримує професійний режим або в чому суть BUG-09?",
                lines=2,
            )
            ask_button = gr.Button("Отримати відповідь", variant="primary")
            answer_output = gr.Markdown(label="Відповідь асистента")
            status_output = gr.Markdown()
            sources_output = gr.Markdown(label="Використані джерела")

            with gr.Accordion("🔍 Переглянути знайдені фрагменти документації (Retrieved Chunks)", open=False):
                chunks_output = gr.Markdown()

            ask_button.click(
                fn=handle_user_question,
                inputs=[question_input],
                outputs=[answer_output, sources_output, status_output, chunks_output],
            )

        with gr.Tab("📚 База знань"):
            reindex_button = gr.Button("🔄 Переіндексувати базу знань", variant="secondary")
            reindex_status = gr.Textbox(label="Статус індексації", lines=4)
            reindex_button.click(fn=handle_reindex_button, inputs=[], outputs=[reindex_status])

        with gr.Tab("📊 Оцінка якості (Evaluation)"):
            eval_button = gr.Button("▶️ Запустити тестування RAG", variant="primary")
            eval_report_output = gr.Markdown()
            eval_button.click(fn=run_rag_evaluation, inputs=[], outputs=[eval_report_output])

    return demo_app


if __name__ == "__main__":
    project_config = load_project_config()
    print(index_knowledge_base(force_reindex=False))
    interface = build_gradio_interface()
    interface.launch(server_name=project_config["server_host"], server_port=project_config["server_port"], share=True)
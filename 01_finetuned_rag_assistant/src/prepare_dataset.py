import json
import re
from pathlib import Path
from src.config import load_project_config

SYSTEM_INSTRUCTION = (
    "Ти — технічний AI-асистент проєкту Data Science Research and Development Laboratory (Vulcan Core). "
    "Відповідай українською мовою чітко, технічно грамотно та виключно на основі наданого контексту документації."
)


def extract_sections_from_markdown(markdown_path: Path) -> list[dict]:
    """Зчитує Markdown-файл і розбиває його на пари (заголовок_розділу, текст_розділу)."""
    raw_text = markdown_path.read_text(encoding="utf-8")
    cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    parts = re.split(r"(^#{2,3}\s+.+$)", cleaned_text, flags=re.MULTILINE)
    sections = []
    current_heading = "Загальний опис проєкту"
    current_content = []

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if stripped.startswith("##"):
            if current_content:
                sections.append(
                    {
                        "source": markdown_path.name,
                        "heading": current_heading,
                        "content": "\n\n".join(current_content),
                    }
                )
                current_content = []
            current_heading = stripped.lstrip("#").strip()
        else:
            current_content.append(stripped)

    if current_content:
        sections.append(
            {
                "source": markdown_path.name,
                "heading": current_heading,
                "content": "\n\n".join(current_content),
            }
        )

    return sections


def build_training_examples(sections: list[dict]) -> list[dict]:
    """Формує навчальні приклади формату ChatML (system / user / assistant) для RAG-поведінки."""
    training_records = []

    for section in sections:
        source_name = section["source"]
        heading = section["heading"]
        content = section["content"]

        if len(content) < 40:
            continue

        trimmed_content = content[:1800]
        user_message = (
            f"Контекст із файлу {source_name} (розділ '{heading}'):\n"
            f"<context>\n{trimmed_content}\n</context>\n\n"
            f"Питання: Поясни деталі розділу «{heading}» у системі Data Science R&D Laboratory."
        )
        assistant_message = (
            f"Згідно з документацією (`{source_name}`, розділ **«{heading}»**):\n\n{trimmed_content}"
        )

        training_records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]
            }
        )

    return training_records


def generate_jsonl_dataset() -> Path:
    """Головна функція підготовки датасету з усіх .md файлів папки knowledge_base."""
    config = load_project_config()
    markdown_files = sorted(config["knowledge_base_dir"].glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(f"У папці {config['knowledge_base_dir']} не знайдено жодного .md файлу!")

    all_sections = []
    for md_file in markdown_files:
        file_sections = extract_sections_from_markdown(md_file)
        all_sections.extend(file_sections)
        print(f"Оброблено файл {md_file.name}: знайдено {len(file_sections)} розділів.")

    dataset_examples = build_training_examples(all_sections)
    output_path = config["dataset_output_path"]

    with output_path.open("w", encoding="utf-8") as jsonl_file:
        for example in dataset_examples:
            jsonl_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Датасет збережено у {output_path} (усього прикладів: {len(dataset_examples)}).")
    return output_path


if __name__ == "__main__":
    generate_jsonl_dataset()
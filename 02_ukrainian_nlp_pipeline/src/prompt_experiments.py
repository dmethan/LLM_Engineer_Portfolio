import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SAMPLE_EVENT_TEXTS = [
    "Сервак прод впав через витік памяті в контейнері докер, терміново рестартимо!",
    "Выкатили обнову бэкенда, теперь API отвечает в два раза быстрее",
    "Утечка БД клиентов в сеть, служба безопасности уже расследует инцидент",
]


def load_generation_llm() -> tuple:
    """Завантажує мовну модель для генеративних та структурованих задач."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model, device


def build_prompt_variant(variant_name: str, input_text: str) -> str:
    """Формує один із трьох варіантів промпту для порівняння якості JSON-відповіді."""
    if variant_name == "zero_shot":
        return (
            f"Визнач категорію події (інцидент, кібератака, реліз, інше) та поверни JSON з ключами "
            f"category, severity, summary_uk українською мовою для тексту:\n{input_text}"
        )

    if variant_name == "strict_schema":
        return (
            "Ти — аналітик технічних інцидентів. Проаналізуй текст (може бути українською або російською, з помилками). "
            "Поверни ВИКЛЮЧНО валідний JSON-об'єкт без пояснень і без markdown-блоків ```json.\n"
            "Дозволені значення category: 'інцидент', 'кібератака', 'реліз', 'інше'.\n"
            "Дозволені значення severity: 'низька', 'середня', 'висока'.\n"
            "Ключ 'summary_uk': короткий опис події українською (1 речення).\n"
            f"Текст: {input_text}"
        )

    if variant_name == "few_shot":
        return (
            "Витягни інформацію про подію у строгому форматі JSON (усі текстові значення — українською мовою).\n\n"
            "Приклад 1:\n"
            "Вхід: 'БД легла, 500 помилка на сайті'\n"
            'Вихід: {"category": "інцидент", "severity": "висока", "summary_uk": "Відмова бази даних та помилка 500 на сайті."}\n\n'
            "Приклад 2:\n"
            "Вхід: 'Выкатили новую версию сервиса авторизации'\n"
            'Вихід: {"category": "реліз", "severity": "низька", "summary_uk": "Випущено нову версію сервісу авторизації."}\n\n'
            f"Вхід: '{input_text}'\n"
            "Вихід: "
        )

    raise ValueError(f"Невідомий варіант промпту: {variant_name}")


def extract_and_validate_json(raw_output: str) -> tuple[bool, dict | None]:
    """Перевіряє, чи містить відповідь моделі валідний JSON."""
    cleaned = re.sub(r"```json|```", "", raw_output).strip()
    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    if start_brace == -1 or end_brace == -1:
        return False, None

    candidate_json = cleaned[start_brace : end_brace + 1]
    try:
        parsed_data = json.loads(candidate_json)
        return isinstance(parsed_data, dict), parsed_data
    except json.JSONDecodeError:
        return False, None


def run_prompt_comparison() -> None:
    """Тестує 3 варіанти промптів на контрольних текстах та підраховує частку валідних JSON."""
    tokenizer, model, device = load_generation_llm()
    variants = ["zero_shot", "strict_schema", "few_shot"]
    stats = {name: {"valid_count": 0, "total": len(SAMPLE_EVENT_TEXTS)} for name in variants}

    for variant in variants:
        print(f"\n================ ВАРІАНТ ПРОМПТУ: {variant} ================")
        for sample_text in SAMPLE_EVENT_TEXTS:
            prompt = build_prompt_variant(variant, sample_text)
            messages = [{"role": "user", "content": prompt}]
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([chat_text], return_tensors="pt").to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[0][inputs.input_ids.shape[1] :]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            is_valid, parsed_json = extract_and_validate_json(response_text)

            if is_valid:
                stats[variant]["valid_count"] += 1

            print(f"Текст: {sample_text}")
            print(f"Валідний JSON: {'ТАК' if is_valid else 'НІ'} | Відповідь: {parsed_json if is_valid else response_text}")

    print("\n================ ПІДСУМОК ПОРІВНЯННЯ ПРОМПТІВ ================")
    best_variant = None
    best_rate = -1.0
    for variant, info in stats.items():
        rate = info["valid_count"] / max(info["total"], 1)
        print(f"- {variant}: валідних JSON {info['valid_count']}/{info['total']} ({rate * 100:.1f}%)")
        if rate > best_rate:
            best_rate = rate
            best_variant = variant

    print(f"Рекомендований промпт за результатами тесту: {best_variant}")


if __name__ == "__main__":
    run_prompt_comparison()
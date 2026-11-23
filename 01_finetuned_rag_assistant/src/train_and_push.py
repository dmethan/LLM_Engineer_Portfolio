import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.config import load_project_config
from src.prepare_dataset import generate_jsonl_dataset


def run_lora_finetuning_and_publish() -> None:
    """Донавчає базову LLM методом LoRA на документації проєкту та публікує адаптер на Hugging Face Hub."""
    config = load_project_config()

    if not config["hf_token"] or config["hf_token"] == "hf_your_write_token_here":
        raise ValueError("Вкажіть дійсний HF_TOKEN (з правами Write) у файлі .env перед запуском навчання!")

    login(token=config["hf_token"])
    dataset_path = generate_jsonl_dataset()
    train_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    use_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"Завантаження базової моделі {config['base_llm_model_name']}...")

    tokenizer = AutoTokenizer.from_pretrained(config["base_llm_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["base_llm_model_name"],
        torch_dtype=compute_dtype,
        device_map="auto" if use_cuda else None,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_arguments = SFTConfig(
        output_dir=str(config["local_adapter_dir"]),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=2,
        save_strategy="no",
        max_seq_length=1024,
        bf16=use_cuda,
        fp16=False,
        use_cpu=not use_cuda,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_arguments,
    )

    trainer.train()

    adapter_dir = config["local_adapter_dir"]
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    repo_id = config["finetuned_repo_id"]
    trainer.model.push_to_hub(repo_id, token=config["hf_token"])
    tokenizer.push_to_hub(repo_id, token=config["hf_token"])
    print(f"Модель опублікована: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    run_lora_finetuning_and_publish()
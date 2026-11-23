import time
from pathlib import Path
import joblib
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset_prep import DATA_DIRECTORY, prepare_train_val_test_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
TRANSFORMER_BASE_MODEL = "distilbert-base-multilingual-cased"


def compute_binary_metrics(true_labels: list[int], predicted_labels: list[int]) -> dict:
    """Обчислює Accuracy, Precision, Recall та F1-score для бінарної класифікації."""
    return {
        "accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 4),
        "precision": round(float(precision_score(true_labels, predicted_labels, zero_division=0)), 4),
        "recall": round(float(recall_score(true_labels, predicted_labels, zero_division=0)), 4),
        "f1_score": round(float(f1_score(true_labels, predicted_labels, zero_division=0)), 4),
    }


def train_baseline_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Навчає простий baseline на TF-IDF ознаках та логістичній регресії."""
    start_timestamp = time.perf_counter()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    x_train_features = vectorizer.fit_transform(train_df["cleaned_text"])
    x_test_features = vectorizer.transform(test_df["cleaned_text"])

    logistic_model = LogisticRegression(max_iter=500, random_state=42)
    logistic_model.fit(x_train_features, train_df["relevance_label"])

    test_preds = logistic_model.predict(x_test_features)
    metrics = compute_binary_metrics(test_df["relevance_label"].tolist(), test_preds.tolist())
    metrics["elapsed_sec"] = round(time.perf_counter() - start_timestamp, 3)

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "model": logistic_model},
        SAVED_MODELS_DIR / "baseline_model.joblib",
    )
    return metrics


def build_dataloader(texts: list[str], labels: list[int], tokenizer, batch_size: int = 8, shuffle: bool = False) -> DataLoader:
    """Токенізує список текстів та формує PyTorch DataLoader."""
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    tensor_dataset = TensorDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_transformer_split(model, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    """Оцінює модель на переданому DataLoader (validation або test)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    mean_loss = total_loss / max(len(loader), 1)
    return mean_loss, all_targets, all_preds


def train_transformer_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_epochs: int = 4,
) -> dict:
    """Виконує fine-tuning pretrained Transformer (DistilBERT) та зберігає навчену модель."""
    start_timestamp = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nНавчання Transformer ({TRANSFORMER_BASE_MODEL}) на пристрої: {device}...")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_BASE_MODEL, num_labels=2)
    model.to(device)

    train_loader = build_dataloader(
        train_df["cleaned_text"].tolist(), train_df["relevance_label"].tolist(), tokenizer, batch_size=8, shuffle=True
    )
    val_loader = build_dataloader(
        val_df["cleaned_text"].tolist(), val_df["relevance_label"].tolist(), tokenizer, batch_size=8, shuffle=False
    )
    test_loader = build_dataloader(
        test_df["cleaned_text"].tolist(), test_df["relevance_label"].tolist(), tokenizer, batch_size=8, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    for epoch_num in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_targets, val_preds = evaluate_transformer_split(model, val_loader, device)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        print(
            f"Епоха {epoch_num}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

    _, test_targets, test_preds = evaluate_transformer_split(model, test_loader, device)
    metrics = compute_binary_metrics(test_targets, test_preds)
    metrics["elapsed_sec"] = round(time.perf_counter() - start_timestamp, 2)

    output_model_dir = SAVED_MODELS_DIR / "distilbert_relevance"
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Модель збережено у {output_model_dir}")
    return metrics


def run_all_classification_experiments() -> None:
    """Готує дані, навчає Baseline та Transformer і виводить порівняльну таблицю."""
    train_path = DATA_DIRECTORY / "train.csv"
    val_path = DATA_DIRECTORY / "val.csv"
    test_path = DATA_DIRECTORY / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        train_df, val_df, test_df = prepare_train_val_test_splits()
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

    print("--- 1. Baseline (TF-IDF + LogisticRegression) ---")
    baseline_metrics = train_baseline_classifier(train_df, test_df)

    print("\n--- 2. Transformer Fine-Tuning (DistilBERT) ---")
    transformer_metrics = train_transformer_classifier(train_df, val_df, test_df, num_epochs=4)

    comparison_df = pd.DataFrame(
        [
            {"Модель": "Baseline (TF-IDF + LogReg)", **baseline_metrics},
            {"Модель": "Transformer (DistilBERT)", **transformer_metrics},
        ]
    )
    print("\nПорівняння на тестовій вибірці:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    run_all_classification_experiments()
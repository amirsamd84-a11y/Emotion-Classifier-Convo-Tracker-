from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
HUB_REPO = "TrunkSam/support-emotion-classifier"

dataset = load_dataset("dair-ai/emotion")
label_names = dataset["train"].features["label"].names

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)


tokenized = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
label_names_map = {i: name for i, name in enumerate(label_names)}
model.config.id2label = label_names_map
model.config.label2id = {name: i for i, name in label_names_map.items()}

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_weighted": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()

    metrics = trainer.evaluate(tokenized["test"])
    print(metrics)

    model.push_to_hub(HUB_REPO)
    tokenizer.push_to_hub(HUB_REPO)

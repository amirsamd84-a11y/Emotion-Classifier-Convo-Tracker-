from datasets import load_dataset, Dataset, concatenate_datasets
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

# augment with hand-labeled negation/indirect examples the base dataset lacks
extra_examples = [
    {"text": "i bought this product and im not happy with it", "label": 3},
    {"text": "this is not what i expected at all", "label": 3},
    {"text": "im disappointed with this purchase", "label": 0},
    {"text": "i dont like this product", "label": 3},
    {"text": "this doesnt work the way i wanted", "label": 3},
    {"text": "im not satisfied with the service", "label": 3},
    {"text": "i cant say im impressed", "label": 0},
    {"text": "this is not okay", "label": 3},
    {"text": "i wasnt expecting it to break so soon", "label": 0},
    {"text": "im not happy with how this turned out", "label": 3},
    {"text": "this isnt good enough", "label": 3},
    {"text": "i regret buying this", "label": 0},
    {"text": "this is far from what i hoped for", "label": 0},
    {"text": "im not okay with this", "label": 3},
    {"text": "i dont think this is acceptable", "label": 3},
    {"text": "this doesnt feel right", "label": 4},
    {"text": "im not sure this was a good idea", "label": 4},
    {"text": "i wouldnt recommend this to anyone", "label": 3},
    {"text": "this isnt the quality i paid for", "label": 3},
    {"text": "im not thrilled about this outcome", "label": 0},
]

extra_dataset = Dataset.from_list(extra_examples)
extra_dataset = extra_dataset.cast(dataset["train"].features)
dataset["train"] = concatenate_datasets([dataset["train"], extra_dataset])

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
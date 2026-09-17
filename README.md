# Support Ticket Emotion Classifier

A fine-tuned DistilBERT model that classifies short text into one of six emotions — sadness, joy, love, anger, fear, surprise. Built to help triage customer support messages by urgency: an angry or fearful message can be flagged for priority handling instead of sitting in a generic queue.

Model on the Hub: [TrunkSam/support-emotion-classifier](https://huggingface.co/TrunkSam/support-emotion-classifier)

## Why this approach

Off-the-shelf sentiment models only give positive/negative/neutral, which isn't granular enough for triage — an angry customer and a confused one both read as "negative" but need different responses. Fine-tuning on a 6-class emotion dataset gives a more actionable signal at a similar computational cost.

## Dataset

[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — 20,000 short, English, Twitter-style sentences labeled with one of six emotions. Split: 16,000 train / 2,000 validation / 2,000 test.

Class distribution is imbalanced (joy and sadness dominate, love and surprise are underrepresented), which shows up directly in per-class results below.

| Emotion  | Train examples |
|----------|-----------------|
| Joy      | 5,362 |
| Sadness  | 4,666 |
| Anger    | 2,159 |
| Fear     | 1,937 |
| Love     | 1,304 |
| Surprise | 572   |

## Model

Base model: `distilbert-base-uncased`, fine-tuned with a classification head for 6 labels.

Training config:
- Learning rate: 2e-5
- Batch size: 32
- Epochs: 3
- Weight decay: 0.01
- Selection metric: weighted F1 (accuracy alone is misleading on imbalanced classes)

## Results

Evaluated on the held-out test set (2,000 examples, never seen during training):

| Emotion  | Precision | Recall | F1   | Support |
|----------|-----------|--------|------|---------|
| Sadness  | 0.96      | 0.96   | 0.96 | 581     |
| Joy      | 0.94      | 0.96   | 0.95 | 695     |
| Love     | 0.87      | 0.79   | 0.83 | 159     |
| Anger    | 0.95      | 0.90   | 0.92 | 275     |
| Fear     | 0.90      | 0.88   | 0.89 | 224     |
| Surprise | 0.70      | 0.83   | 0.76 | 66      |

**Overall accuracy: 93%**

The weakest class is surprise, which also has the fewest training examples (572) — the model performs worse exactly where the data is thinnest, which is the expected effect of class imbalance rather than a modeling flaw. Love shows a similar but smaller gap. Sadness, joy, and anger — the highest-volume classes — are the most reliable, which is also where triage matters most in a support context.

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="TrunkSam/support-emotion-classifier", top_k=None)
classifier("This is taking forever, I need help now")
```

```
[{'label': 'anger', 'score': 0.81}, {'label': 'sadness', 'score': 0.11}, ...]
```

## Running locally

```bash
pip install -r requirements.txt
python train.py   # retrains and pushes to the Hub
python app.py      # launches the Gradio demo
```

## Project structure

```
support-emotion-classifier/
├── README.md
├── train.py
├── app.py
├── requirements.txt
└── notebook.ipynb
```
## Conversation-level tracking

Beyond single-message classification, `conversation_tracker.py` tracks emotional trend across a multi-message conversation from the same customer, flagging whether they're escalating or de-escalating over time based on combined anger/sadness/fear scores. This is a simple first-vs-last comparison against a fixed threshold — a more robust version would fit a trend line across all messages rather than just comparing endpoints.

\`\`\`python
from conversation_tracker import ConversationTracker

tracker = ConversationTracker()
tracker.add_message("session_1", "Hi, I have a question about my order.")
tracker.add_message("session_1", "This is ridiculous, I need this resolved now.")

tracker.get_trend("session_1")  # "escalating"
\`\`\`

## Limitations

The dataset is short, informal English text (Twitter-style), so performance on longer or more formal writing (e.g. detailed support emails) isn't validated and would need separate testing before production use. The model also outputs a single dominant emotion per message; it doesn't handle mixed or ambiguous emotional tone within one message.

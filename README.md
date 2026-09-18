# Support Ticket Emotion Classifier

A fine-tuned DistilBERT model that classifies support messages into six emotions — sadness, joy, love, anger, fear, surprise — and tracks how a customer's tone shifts across a conversation, so agents can prioritize escalating threads. Includes a comparison against a local LLM (Ollama) on cases the fine-tuned model struggles with.

Model on the Hub: [TrunkSam/support-emotion-classifier](https://huggingface.co/TrunkSam/support-emotion-classifier)

## Why this approach

Off-the-shelf sentiment models only give positive/negative/neutral, which isn't granular enough for triage — an angry customer and a confused one both read as "negative" but need different responses. Fine-tuning on a 6-class emotion dataset gives a more actionable signal at a similar computational cost, and tracking the trend across a conversation (not just one message) catches customers getting worse over time, not just customers who are already upset.

## Dataset

[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — 20,000 short, English, Twitter-style sentences labeled with one of six emotions. Split: 16,000 train / 2,000 validation / 2,000 test.

Class distribution is imbalanced — joy and sadness dominate, love and surprise are underrepresented:

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

The weakest class is surprise, which also has the fewest training examples — performance drops exactly where the data is thinnest, the expected effect of class imbalance rather than a modeling flaw.

## Known limitation: negation

Manual testing surfaced a real failure mode: sentences using negation ("I bought this and I'm **not** happy with it") were misclassified as joy, since the base training data is almost entirely direct statements ("i feel humiliated") with very little negation. The model was picking up on surface vocabulary ("bought," "product") rather than the sentence's actual polarity.

Fix attempted: added 20 hand-labeled negation/indirect examples to the training set and retrained. This partially worked — sentences resembling the added examples ("not happy" patterns) were fixed, but the fix didn't generalize to negated words outside that set (e.g. "not afraid" still returned fear). With only 20 examples, the model appears to have learned specific negated phrases rather than the general rule that negation can flip emotional meaning. A production fix would need much broader, systematic coverage of negation across all six emotion words.

## Comparison: fine-tuned classifier vs. local LLM (Ollama)

To explore whether a general-purpose LLM handles negation better than a small fine-tuned classifier, `ollama_classifier.py` runs the same test cases through `llama3.2:3b` (via a local Ollama server) instead of DistilBERT.

| Sentence | Fine-tuned DistilBERT | LLM (llama3.2:3b, few-shot) |
|----------|------------------------|-------------------------------|
| "I'm not happy" | anger (0.79) ✓ | anger ✓ |
| "I'm not afraid" | fear (0.98) ✗ | joy (by convention, see note) |
| "I am afraid" | fear (0.99) ✓ | fear ✓ |
| "I bought this and I'm not happy with it" | anger (0.95) ✓ | anger ✓ |

Getting the LLM to handle negation reliably required **few-shot prompting** — showing worked examples directly in the prompt, including a negated-fear case — rather than relying on instructions alone. Zero-shot instructions ("pay attention to negation") produced inconsistent answers across runs; few-shot examples made the output consistent.

**Important caveat on "I'm not afraid":** this sentence doesn't clearly fit any of the six forced emotion categories — it's closer to neutral or confident than joy, sadness, or any other label. The few-shot example taught the model to consistently answer "joy" for this case, but that reflects an arbitrary convention chosen while prompting, not an objectively correct classification. This points to a real limitation of the task setup itself: a fixed 6-emotion label space can't represent neutral or ambiguous statements, no matter how capable the underlying model is.

**The real tradeoff this comparison shows:** the fine-tuned model is fast, consistent, and cheap to run, but its accuracy is capped by how well the training data covers a given pattern. The LLM approach required no retraining and, with the right prompting technique, generalized better — but is meaningfully slower per message (several seconds vs. milliseconds) and depends on a local LLM server being available, which limits how it can be deployed.

## Conversation-level tracking

Beyond single-message classification, `conversation_tracker.py` tracks emotional trend across a multi-message conversation from the same customer, flagging whether they're escalating or de-escalating. This compares the first message in a session against the most recent one — a simple heuristic, not a full trend line across every message, so a conversation that spikes and recovers mid-thread can still read as "stable."

```python
from conversation_tracker import ConversationTracker

tracker = ConversationTracker()
tracker.add_message("session_1", "Hi, I have a question about my order.")
tracker.add_message("session_1", "This is ridiculous, I need this resolved now.")

tracker.get_trend("session_1")  # "escalating"
```

Conversation history can be exported per session as a CSV via `tracker.export_session(session_id)` — surfaced in the app as a "Download conversation log" button.

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
python train.py   # retrains on the augmented dataset and pushes to the Hub
python app.py      # launches the chat demo with trend tracking and CSV export
```

To try the Ollama comparison, install [Ollama](https://ollama.com), run `ollama pull llama3.2:3b`, then:
```bash
python ollama_classifier.py
```

## Project structure

```
support-emotion-classifier/
├── README.md
├── train.py
├── app.py
├── conversation_tracker.py
├── ollama_classifier.py
├── requirements.txt
├── .gitignore
└── notebook.ipynb
```

## Limitations

The dataset is short, informal English text, so performance on longer or more formal writing (e.g. detailed support emails) isn't validated. The model outputs a single dominant emotion per message and doesn't handle mixed or ambiguous emotional tone within one message. Conversation memory is in-process only — session history isn't persisted to disk, so it resets when the app restarts. The trend calculation compares only the first and last message in a session, not the full trajectory in between. The fixed 6-emotion label space cannot represent neutral or ambiguous statements, a limitation of the task design rather than any specific model.

## Possible extensions

- Fit a trend line across the full conversation instead of comparing only the first and last message
- Persist session history to a database instead of in-memory only
- Expand negation training data systematically across all six emotions, rather than the current 20 hand-picked examples
- Add a "neutral" category to the label space to properly handle ambiguous statements like negated-fear sentences
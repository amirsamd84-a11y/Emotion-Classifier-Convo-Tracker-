from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

CHECKPOINT_PATH = r"G:\Projects\Emotion Classifier\Emotion_Classifier\results\checkpoint-1500"
HUB_REPO = "TrunkSam/support-emotion-classifier"

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in enumerate(label_names)}

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
model.config.id2label = id2label
model.config.label2id = label2id

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

login()

model.push_to_hub(HUB_REPO)
tokenizer.push_to_hub(HUB_REPO)

print("Done — labels fixed.")
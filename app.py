import gradio as gr
from transformers import pipeline

MODEL_REPO = "TrunkSam/support-emotion-classifier"

classifier = pipeline("text-classification", model=MODEL_REPO, top_k=None)


def predict_emotion(text):
    results = classifier(text)[0]
    return {r["label"]: r["score"] for r in results}


demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(placeholder="Paste a support message..."),
    outputs=gr.Label(num_top_classes=6),
    title="Support Ticket Emotion Triage",
    description="Classifies a support message into one of six emotions to help prioritize urgent or negative tickets.",
)

if __name__ == "__main__":
    demo.launch()

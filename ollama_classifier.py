import requests

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

FEW_SHOT_EXAMPLES = """Message: "I got the promotion!"
Answer: joy

Message: "I bought this and I'm not happy with it"
Answer: anger

Message: "I'm not afraid of you"
Answer: joy

Message: "I am afraid of the dark"
Answer: fear

Message: "This isn't what I expected"
Answer: sadness

Message: "I can't stop thinking about her"
Answer: love"""


def classify(text, model="llama3.2:3b"):
    prompt = f"""Classify the emotion in each message as exactly one word from: sadness, joy, love, anger, fear, surprise.
Pay close attention to negation words like "not", "isn't", "don't" — they reverse or change the emotion.

{FEW_SHOT_EXAMPLES}

Message: "{text}"
Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    raw = response.json()["response"].strip().lower()

    first_line = raw.splitlines()[0] if raw else ""
    for label in LABELS:
        if label in first_line:
            return label

    for label in LABELS:
        if label in raw:
            return label

    return "surprise"


if __name__ == "__main__":
    tests = [
        "I'm not happy",
        "I'm not afraid",
        "I am afraid",
        "I bought this and I'm not happy with it",
    ]

    for t in tests:
        print(f"{t!r} -> {classify(t)}")
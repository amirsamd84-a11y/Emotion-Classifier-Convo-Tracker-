from dataclasses import dataclass, field
from datetime import datetime
from transformers import pipeline


NEGATIVE_EMOTIONS = {"anger", "sadness", "fear"}


@dataclass
class MessageResult:
    text: str
    timestamp: datetime
    scores: dict
    top_emotion: str
    negative_score: float


@dataclass
class Session:
    session_id: str
    history: list = field(default_factory=list)


class ConversationTracker:
    def __init__(self, model_repo="TrunkSam/support-emotion-classifier"):
        self.classifier = pipeline("text-classification", model=model_repo, top_k=None)
        self.sessions = {}

    def _get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def add_message(self, session_id, text, timestamp=None):
        timestamp = timestamp or datetime.now()
        raw = self.classifier(text)[0]
        scores = {r["label"]: r["score"] for r in raw}
        top_emotion = max(scores, key=scores.get)
        negative_score = sum(scores[e] for e in NEGATIVE_EMOTIONS)

        result = MessageResult(
            text=text,
            timestamp=timestamp,
            scores=scores,
            top_emotion=top_emotion,
            negative_score=negative_score,
        )

        session = self._get_session(session_id)
        session.history.append(result)
        return result

    def get_trajectory(self, session_id):
        session = self._get_session(session_id)
        return [(r.timestamp, r.top_emotion, round(r.negative_score, 3)) for r in session.history]

    def get_trend(self, session_id):
        session = self._get_session(session_id)
        scores = [r.negative_score for r in session.history]

        if len(scores) < 2:
            return "not_enough_data"

        delta = scores[-1] - scores[0]

        if delta > 0.2:
            return "escalating"
        if delta < -0.2:
            return "de-escalating"
        return "stable"


if __name__ == "__main__":
    tracker = ConversationTracker()

    messages = [
        "Hi, I have a question about my order.",
        "It's been three days and I still haven't heard back.",
        "This is ridiculous, I need this resolved now.",
    ]

    for msg in messages:
        tracker.add_message("session_1", msg)

    for timestamp, emotion, negative_score in tracker.get_trajectory("session_1"):
        print(timestamp, emotion, negative_score)

    print("Trend:", tracker.get_trend("session_1"))

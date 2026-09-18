from dataclasses import dataclass, field
from datetime import datetime

from ollama_classifier import classify


NEGATIVE_EMOTIONS = {"anger", "sadness", "fear"}


@dataclass
class MessageResult:
    text: str
    timestamp: datetime
    top_emotion: str


@dataclass
class Session:
    session_id: str
    history: list = field(default_factory=list)


class ConversationTracker:
    def __init__(self, model="llama3.2:3b"):
        self.model = model
        self.sessions = {}

    def _get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def add_message(self, session_id, text, timestamp=None):
        timestamp = timestamp or datetime.now()
        top_emotion = classify(text, model=self.model)

        result = MessageResult(text=text, timestamp=timestamp, top_emotion=top_emotion)

        session = self._get_session(session_id)
        session.history.append(result)
        return result

    def get_trajectory(self, session_id):
        session = self._get_session(session_id)
        return [(r.timestamp, r.top_emotion) for r in session.history]

    def get_trend(self, session_id):
        session = self._get_session(session_id)
        negatives = [1 if r.top_emotion in NEGATIVE_EMOTIONS else 0 for r in session.history]

        if len(negatives) < 2:
            return "not_enough_data"

        delta = negatives[-1] - negatives[0]

        if delta > 0:
            return "escalating"
        if delta < 0:
            return "de-escalating"
        return "stable"

    def export_session(self, session_id):
        session = self._get_session(session_id)
        return [
            {
                "session_id": session_id,
                "timestamp": r.timestamp.isoformat(),
                "text": r.text,
                "top_emotion": r.top_emotion,
            }
            for r in session.history
        ]

    def export_all(self):
        rows = []
        for session_id in self.sessions:
            rows.extend(self.export_session(session_id))
        return rows


if __name__ == "__main__":
    tracker = ConversationTracker()

    messages = [
        "Hi, I have a question about my order.",
        "It's been three days and I still haven't heard back.",
        "This is ridiculous, I need this resolved now.",
    ]

    for msg in messages:
        tracker.add_message("session_1", msg)

    for timestamp, emotion in tracker.get_trajectory("session_1"):
        print(timestamp, emotion)

    print("Trend:", tracker.get_trend("session_1"))
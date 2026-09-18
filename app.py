import csv
import os
import uuid

import gradio as gr

from conversation_tracker import ConversationTracker

tracker = ConversationTracker()


def new_session():
    return str(uuid.uuid4())


def submit_message(text, session_id, chat_history):
    if not text.strip():
        return chat_history, "", session_id, "Trend: not_enough_data"

    result = tracker.add_message(session_id, text)
    trend = tracker.get_trend(session_id)

    bot_reply = f"**{result.top_emotion}**"

    chat_history = chat_history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": bot_reply},
    ]

    return chat_history, "", session_id, f"Trend: {trend}"


def reset_conversation():
    return [], "", new_session(), "Trend: not_enough_data"


def export_current_session(session_id):
    rows = tracker.export_session(session_id)
    if not rows:
        return None

    filepath = f"conversation_{session_id[:8]}.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return filepath


with gr.Blocks(title="Support Ticket Emotion Triage") as demo:
    gr.Markdown("# Support Ticket Emotion Triage")
    gr.Markdown(
        "Classifies each message in a conversation using a local LLM (llama3.2:3b via Ollama) "
        "and tracks whether the customer's tone is escalating over the thread."
    )

    session_id = gr.State(new_session())

    chatbot = gr.Chatbot(label="Conversation", height=400)
    trend_display = gr.Textbox(label="Conversation trend", value="Trend: not_enough_data", interactive=False)

    with gr.Row():
        message_box = gr.Textbox(placeholder="Type a support message...", scale=4, show_label=False)
        send_button = gr.Button("Send", scale=1)

    with gr.Row():
        clear_button = gr.Button("Start new conversation")
        export_button = gr.Button("Download conversation log (CSV)")

    export_file = gr.File(label="Exported CSV")

    send_button.click(
        fn=submit_message,
        inputs=[message_box, session_id, chatbot],
        outputs=[chatbot, message_box, session_id, trend_display],
    )
    message_box.submit(
        fn=submit_message,
        inputs=[message_box, session_id, chatbot],
        outputs=[chatbot, message_box, session_id, trend_display],
    )
    clear_button.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[chatbot, message_box, session_id, trend_display],
    )
    export_button.click(
        fn=export_current_session,
        inputs=[session_id],
        outputs=[export_file],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
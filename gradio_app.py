import argparse

import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_model(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def respond(prompt, tokenizer, model, device, max_new_tokens=96):
    input_text = f"question: {prompt}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.8,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Global variables to hold tokenizer and model
TOKENIZER = None
MODEL = None
DEVICE = None


def build_app(tokenizer, model, device):
    css = """
    :root {
        --bg-0: #03070d;
        --bg-1: #07121d;
        --panel: rgba(12, 24, 37, 0.88);
        --panel-2: rgba(18, 33, 49, 0.95);
        --text: #eaf4ff;
        --muted: #9bb0c6;
        --accent: #67f2c2;
        --accent-2: #ffb347;
        --border: rgba(103, 242, 194, 0.18);
    }

    body {
        background:
            radial-gradient(circle at 20% 0%, rgba(103, 242, 194, 0.12), transparent 28%),
            radial-gradient(circle at 80% 10%, rgba(255, 179, 71, 0.10), transparent 24%),
            linear-gradient(180deg, var(--bg-1), var(--bg-0));
        color: var(--text);
    }

    .gradio-container {
        background:
            radial-gradient(circle at top, rgba(103, 242, 194, 0.10), transparent 30%),
            radial-gradient(circle at 70% 15%, rgba(255, 179, 71, 0.10), transparent 22%),
            linear-gradient(180deg, var(--bg-1), var(--bg-0));
        color: var(--text);
        font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
    }

    #hero {
        max-width: 980px;
        margin: 0 auto 1rem auto;
        padding: 1.25rem 1.25rem 1rem 1.25rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(18, 33, 49, 0.92), rgba(8, 15, 24, 0.92));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }

    #hero h1 {
        margin: 0;
        color: var(--text);
        letter-spacing: 0.08em;
        font-size: 2rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    #hero .subtitle {
        margin-top: 0.45rem;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.4;
    }

    #hero .status {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(103, 242, 194, 0.28);
        background: rgba(103, 242, 194, 0.10);
        color: var(--accent);
        font-size: 0.85rem;
        letter-spacing: 0.04em;
    }

    .panel {
        max-width: 980px;
        margin: 0 auto;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--panel);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        overflow: hidden;
    }

    .panel-header {
        padding: 0.9rem 1rem;
        background: linear-gradient(90deg, rgba(103, 242, 194, 0.08), rgba(255, 179, 71, 0.08));
        border-bottom: 1px solid rgba(103, 242, 194, 0.14);
        color: var(--text);
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.82rem;
    }

    .gradio-container .gr-button {
        border-radius: 12px !important;
        border: 1px solid rgba(103, 242, 194, 0.22) !important;
        background: linear-gradient(180deg, rgba(103, 242, 194, 0.18), rgba(103, 242, 194, 0.08)) !important;
        color: var(--text) !important;
    }

    .gradio-container .gr-button:hover {
        border-color: rgba(255, 179, 71, 0.35) !important;
        background: linear-gradient(180deg, rgba(255, 179, 71, 0.18), rgba(103, 242, 194, 0.10)) !important;
    }

    .gradio-container textarea,
    .gradio-container input {
        background: rgba(6, 12, 20, 0.92) !important;
        color: var(--text) !important;
        border: 1px solid rgba(103, 242, 194, 0.18) !important;
        border-radius: 12px !important;
    }

    .gradio-container .prose,
    .gradio-container .markdown,
    .gradio-container .wrap {
        color: var(--text) !important;
    }
    """

    def chat(message, history):
        message = message.strip()
        if not message:
            return history

        reply = respond(message, tokenizer, model, device)
        # Gradio 6.x expects list of dicts with "role" and "content" keys
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history

    def clear_chat():
        return []

    with gr.Blocks(title="Rocky Mission Control") as app:
        gr.HTML(
            """
            <div id="hero">
              <h1>Rocky Chat</h1>
              <div class="subtitle">
                A Project Hail Mary themed chat interface for talking to Rocky.
                Built for fast demos, with a dark ship-console look and translation-link vibes.
              </div>
              <div class="status">Link stable. Translation active.</div>
            </div>
            """
        )

        gr.HTML('<div class="panel-header">Conversation Log</div>')
        chatbot = gr.Chatbot(
            label="Rocky",
            height=560,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type a message to Rocky...",
                label="Message",
                lines=2,
            )

        with gr.Row():
            send = gr.Button("Send")
            clear = gr.Button("Clear")

        gr.Examples(
            examples=[
                "Rocky, are you there?",
                "We need to fix the ship.",
                "How much time do we have?",
                "Can you help me think through this?",
                "The mission is in trouble.",
            ],
            inputs=msg,
            label="Try these",
        )

        send.click(
            chat,
            inputs=[msg, chatbot],
            outputs=[chatbot],
        ).then(
            lambda: "",
            outputs=[msg]
        )
        msg.submit(
            chat,
            inputs=[msg, chatbot],
            outputs=[chatbot],
        ).then(
            lambda: "",
            outputs=[msg]
        )
        clear.click(
            clear_chat,
            outputs=[chatbot],
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/rocky-dialogue-augmented-t5/checkpoint-epoch-5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading model from {args.model} on {device}...")
    tokenizer, model = load_model(args.model)
    model.to(device)

    css = """
    :root {
        --bg-0: #03070d;
        --bg-1: #07121d;
        --panel: rgba(12, 24, 37, 0.88);
        --panel-2: rgba(18, 33, 49, 0.95);
        --text: #eaf4ff;
        --muted: #9bb0c6;
        --accent: #67f2c2;
        --accent-2: #ffb347;
        --border: rgba(103, 242, 194, 0.18);
    }

    body {
        background:
            radial-gradient(circle at 20% 0%, rgba(103, 242, 194, 0.12), transparent 28%),
            radial-gradient(circle at 80% 10%, rgba(255, 179, 71, 0.10), transparent 24%),
            linear-gradient(180deg, var(--bg-1), var(--bg-0));
        color: var(--text);
    }

    .gradio-container {
        background:
            radial-gradient(circle at top, rgba(103, 242, 194, 0.10), transparent 30%),
            radial-gradient(circle at 70% 15%, rgba(255, 179, 71, 0.10), transparent 22%),
            linear-gradient(180deg, var(--bg-1), var(--bg-0));
        color: var(--text);
        font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
    }

    #hero {
        max-width: 980px;
        margin: 0 auto 1rem auto;
        padding: 1.25rem 1.25rem 1rem 1.25rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(18, 33, 49, 0.92), rgba(8, 15, 24, 0.92));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }

    #hero h1 {
        margin: 0;
        color: var(--text);
        letter-spacing: 0.08em;
        font-size: 2rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    #hero .subtitle {
        margin-top: 0.45rem;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.4;
    }

    #hero .status {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(103, 242, 194, 0.28);
        background: rgba(103, 242, 194, 0.10);
        color: var(--accent);
        font-size: 0.85rem;
        letter-spacing: 0.04em;
    }

    .panel {
        max-width: 980px;
        margin: 0 auto;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--panel);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        overflow: hidden;
    }

    .panel-header {
        padding: 0.9rem 1rem;
        background: linear-gradient(90deg, rgba(103, 242, 194, 0.08), rgba(255, 179, 71, 0.08));
        border-bottom: 1px solid rgba(103, 242, 194, 0.14);
        color: var(--text);
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.82rem;
    }

    .gradio-container .gr-button {
        border-radius: 12px !important;
        border: 1px solid rgba(103, 242, 194, 0.22) !important;
        background: linear-gradient(180deg, rgba(103, 242, 194, 0.18), rgba(103, 242, 194, 0.08)) !important;
        color: var(--text) !important;
    }

    .gradio-container .gr-button:hover {
        border-color: rgba(255, 179, 71, 0.35) !important;
        background: linear-gradient(180deg, rgba(255, 179, 71, 0.18), rgba(103, 242, 194, 0.10)) !important;
    }

    .gradio-container textarea,
    .gradio-container input {
        background: rgba(6, 12, 20, 0.92) !important;
        color: var(--text) !important;
        border: 1px solid rgba(103, 242, 194, 0.18) !important;
        border-radius: 12px !important;
    }

    .gradio-container .prose,
    .gradio-container .markdown,
    .gradio-container .wrap {
        color: var(--text) !important;
    }
    """
    app = build_app(tokenizer, model, device)
    app.launch(server_name=args.host, server_port=args.port, share=args.share, css=css)


if __name__ == "__main__":
    main()
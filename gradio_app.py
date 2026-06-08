import argparse

import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import inference

DEFAULT_MODEL = "weights/rocky-dialogue-augmented-t5/checkpoint-epoch-5"
DEFAULT_DEVICE = "cpu"
ICON_PATH = "gui/rocky_icon.png"
CSS_PATH = "gui/styles.css"

def load_css():
    try:
        with open(CSS_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def build_app(tokenizer, model, device):
    css = load_css()
    
    def chat(message, history):
        message = message.strip()
        if not message:
            return history

        reply = inference.respond(message, tokenizer, model, device)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history

    def clear_chat():
        return []

    with gr.Blocks(title="Rocky Mission Control", css=css) as app:
        with gr.Column(elem_classes='chat-shell'):
            gr.HTML(
                """
                <div id="hero">
                <h1>Rocky Chat</h1>
                <div class="subtitle">
                    Talk to Rocky.
                </div>
                </div>
                """
            )

            chatbot = gr.Chatbot(label='Rocky',
                                 scale=1,
                                 avatar_images=(None, ICON_PATH),
                                 elem_classes="chatbot")
            
            with gr.Row(scale=1):
                msg = gr.Textbox(
                    placeholder="Type a message to Rocky...",
                    label="Message",
                    lines=1,
                    scale=1,
                    elem_classes='input'
                )

            with gr.Row(scale=1):
                send = gr.Button("Send", scale=1, elem_classes='send-button')
                clear = gr.Button("Clear", scale=1, elem_classes='clear-button')

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

            send.click(chat, inputs=[msg, chatbot], outputs=[chatbot]).then(
                lambda: "", outputs=[msg]
            )
            msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot]).then(
                lambda: "", outputs=[msg]
            )
            clear.click(clear_chat, outputs=[chatbot])

    return app


# ============ Create app at module level for Gradio hot reloading ============
device = torch.device(DEFAULT_DEVICE)
print(f"Loading model from {DEFAULT_MODEL} on {device}...")
tokenizer, model = inference.load_model(DEFAULT_MODEL)
model.to(device)

app = build_app(tokenizer, model, device)
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading model from {args.model} on {device}...")
    tokenizer, model = inference.load_model(args.model)
    model.to(device)

    app_cli = build_app(tokenizer, model, device)
    app_cli.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
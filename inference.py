"""
Chat with Rocky.

Usage:
    python inference.py
    python inference.py --model rocky-t5/best
"""

import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def load_model(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model     = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def respond(prompt, tokenizer, model, device, max_length=128):
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
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=1.0,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/best")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model}...")
    tokenizer, model = load_model(args.model)
    model.to(device)
    print("Rocky is ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        response = respond(user_input, tokenizer, model, device)
        print(f"Rocky: {response}\n")


if __name__ == "__main__":
    main()
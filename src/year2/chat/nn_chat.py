"""
Phase 2.5 - Transformer Chat Interface
Connects the Transformer model to a terminal chat UI
"""

import numpy as np
from src.year2.chat.inference import InferenceEngine
from src.year2.chat.generation import greedy_decode, temperature_sample
from src.year2.chat.conversation import Conversation

CORPUS = """
hello how are you today i am fine thank you very much
the weather in cameroon is warm and sunny all year round
i am learning to build a language model from scratch in python
english and french are the two official languages of cameroon
bayangi and douala are indigenous cameroon languages spoken in the southwest
we are building an ai system that understands cameroon languages
language models learn patterns from large amounts of text data
building artificial intelligence requires mathematics and programming
cameroon is a beautiful country in central africa with many languages
the goal of this project is to build an ai that speaks cameroon languages
"""

LANGUAGES = ["english", "french", "bayangi", "douala"]


def build_engine():
    print("Loading Transformer model...")
    engine = InferenceEngine(
        vocab_size=150, d_model=32, num_heads=2,
        d_ff=64, num_layers=1
    )
    engine.build(CORPUS)
    print(f"Model ready. Vocab size: {engine.vocab.vocab_size}")
    return engine


def run_chat(test_inputs=None):
    print("=" * 50)
    print("  FNI-LLM Transformer Chat (Year 2)")
    print("  Language: English (Cameroon)")
    print("=" * 50)

    engine = build_engine()
    conv = Conversation(model_name="transformer-v0", language="english")

    print("\nType a prompt and the model will continue it.")
    print("Commands: 'temp 0.5', 'lang', 'history', 'clear', 'bye'\n")

    temperature = 1.0
    use_test = test_inputs is not None
    test_idx = 0

    while True:
        if use_test:
            if test_idx >= len(test_inputs):
                break
            user_input = test_inputs[test_idx]
            test_idx += 1
            print(f"You: {user_input}")
        else:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

        if not user_input:
            continue

        low = user_input.lower()

        if low in ("bye", "exit", "quit"):
            print("FNI-LLM: Goodbye! Keep building.")
            conv.add("user", user_input)
            conv.add("bot", "Goodbye! Keep building.")
            break

        if low == "history":
            conv.display()
            continue

        if low == "clear":
            conv.clear()
            print("FNI-LLM: Conversation cleared.")
            continue

        if low == "lang":
            print(f"FNI-LLM: Available languages: {LANGUAGES}")
            print(f"         Current: {conv.language}")
            print(f"         (More languages coming in Year 3)")
            continue

        if low.startswith("temp "):
            try:
                temperature = float(low.split()[1])
                print(f"FNI-LLM: Temperature set to {temperature}")
            except ValueError:
                print("FNI-LLM: Usage: temp 0.5")
            continue

        # Generate response using transformer
        context = conv.get_context() + " " + user_input
        if temperature == 1.0:
            response = greedy_decode(engine, context, max_new_tokens=8)
        else:
            response = temperature_sample(engine, context,
                                          max_new_tokens=8,
                                          temperature=temperature)

        # Show only the newly generated part
        input_words = context.strip().split()
        response_words = response.strip().split()
        new_words = response_words[len(input_words):]
        generated = " ".join(new_words) if new_words else response

        print(f"FNI-LLM: {generated}")
        print(f"         (temp={temperature} | model=untrained | "
              f"Year 3 training will improve this)")

        conv.add("user", user_input)
        conv.add("bot", generated)

    if conv.messages:
        path = conv.save()
        print(f"\n[Conversation saved to {path}]")

    return conv.messages


if __name__ == "__main__":
    run_chat()

"""
Phase 2.5 - Text Generation
Greedy decoding and temperature sampling from the Transformer
"""

import numpy as np
from src.year2.chat.inference import InferenceEngine


def softmax(x, temperature=1.0):
    x = x / temperature
    e = np.exp(x - np.max(x))
    return e / e.sum()


def greedy_decode(engine, prompt, max_new_tokens=10):
    """Always pick the most probable next token"""
    ids = engine.encode(prompt)
    for _ in range(max_new_tokens):
        token_ids = np.array([ids])
        logits = engine.model.forward(token_ids)
        next_id = int(logits[0, -1].argmax())
        ids.append(next_id)
        # Stop at END token
        if next_id == engine.vocab.word_to_id.get(engine.vocab.END, -1):
            break
    return engine.decode(ids)


def temperature_sample(engine, prompt, max_new_tokens=10, temperature=1.0):
    """
    Sample next token from probability distribution.
    temperature < 1.0 = more confident/repetitive
    temperature > 1.0 = more random/creative
    """
    ids = engine.encode(prompt)
    for _ in range(max_new_tokens):
        token_ids = np.array([ids])
        logits = engine.model.forward(token_ids)
        probs = softmax(logits[0, -1], temperature=temperature)
        next_id = int(np.random.choice(len(probs), p=probs))
        ids.append(next_id)
        if next_id == engine.vocab.word_to_id.get(engine.vocab.END, -1):
            break
    return engine.decode(ids)


if __name__ == "__main__":
    np.random.seed(42)

    CORPUS = """
    hello how are you today i am fine thank you
    the weather in cameroon is warm and sunny
    i am learning to build a language model from scratch
    english and french are official languages of cameroon
    bayangi and douala are indigenous cameroon languages
    we are building an ai that speaks cameroon languages
    language models learn from text data and generate new text
    """

    print("=== BUILDING ENGINE ===")
    engine = InferenceEngine(vocab_size=100, d_model=32,
                             num_heads=2, d_ff=64, num_layers=1)
    engine.build(CORPUS)
    print(engine)

    prompt = "hello how are"
    print(f"\nPrompt: '{prompt}'")

    print("\n=== GREEDY DECODING ===")
    for _ in range(3):
        out = greedy_decode(engine, prompt, max_new_tokens=5)
        print(f"  {out}")

    print("\n=== TEMPERATURE SAMPLING ===")
    for temp in [0.5, 1.0, 1.5]:
        out = temperature_sample(engine, prompt, max_new_tokens=5,
                                 temperature=temp)
        print(f"  temp={temp}: {out}")

    print("\nNote: outputs are random (model not trained yet)")
    print("After Year 3 training on Cameroon data, outputs will be meaningful.")
    print("\nDone: Generation complete.")

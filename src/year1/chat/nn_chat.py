"""
Phase 1.5 - Neural Network Chat
Loads the trained XOR network and lets user interact with it via terminal
"""

import numpy as np
from src.year1.neural_network.network import NeuralNetwork
from src.year1.chat.logger import save_conversation


def train_xor_model():
    """Train and return a fresh XOR model"""
    np.random.seed(42)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    nn = NeuralNetwork([2, 4, 1], ['relu', 'sigmoid'])
    nn.train(X, y, epochs=5000, learning_rate=0.1, verbose=False)
    return nn


def parse_input(user_input):
    """Parse two numbers from user input e.g. '0 1' or '1,0'"""
    parts = user_input.replace(",", " ").split()
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def run_nn_chat(test_inputs=None):
    print("=" * 45)
    print("  FNI-LLM Neural Network Chat (XOR Demo)  ")
    print("=" * 45)
    print("Loading and training XOR model...")
    nn = train_xor_model()
    print("Model ready!\n")
    print("Enter two numbers (0 or 1) to get XOR prediction.")
    print("Examples: '0 1'  '1 0'  '1 1'  '0 0'")
    print("Commands: 'help', 'accuracy', 'bye'\n")

    messages = []
    use_test = test_inputs is not None
    test_idx = 0

    # XOR ground truth for accuracy check
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_xor = np.array([[0],[1],[1],[0]], dtype=float)

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
            print("FNI-LLM: Goodbye!")
            messages.append({"role": "user", "text": user_input})
            messages.append({"role": "bot",  "text": "Goodbye!"})
            break

        if low == "help":
            msg = "Enter two numbers like '0 1'. Commands: accuracy, bye"
            print(f"FNI-LLM: {msg}")
            continue

        if low == "accuracy":
            preds = np.round(nn.predict(X_xor))
            acc = np.mean(preds == y_xor)
            msg = f"XOR accuracy: {acc:.0%} ({int(acc*4)}/4 correct)"
            print(f"FNI-LLM: {msg}")
            messages.append({"role": "user", "text": user_input})
            messages.append({"role": "bot",  "text": msg})
            continue

        parsed = parse_input(user_input)
        if parsed is None:
            msg = "Please enter exactly two numbers e.g. '0 1'"
            print(f"FNI-LLM: {msg}")
            continue

        x1, x2 = parsed
        inp = np.array([[x1, x2]])
        raw = nn.predict(inp)[0][0]
        result = round(raw)
        msg = (f"XOR({int(x1)}, {int(x2)}) = {result}  "
               f"(confidence: {raw:.4f})")
        print(f"FNI-LLM: {msg}")

        messages.append({"role": "user", "text": user_input})
        messages.append({"role": "bot",  "text": msg})

    if messages:
        path = save_conversation(messages)
        print(f"\n[Conversation saved to {path}]")

    return messages


if __name__ == "__main__":
    run_nn_chat()

"""
Phase 1.5 - Simple Terminal Chat
Pattern matching responses + conversation logging
"""

from src.year1.chat.logger import save_conversation, load_conversation, list_conversations

RESPONSES = {
    "hello":        "Hi! I am FNI-LLM. Ask me anything.",
    "hi":           "Hello! How can I help?",
    "how are you":  "I am doing great, training hard!",
    "what are you": "I am FNI-LLM, an AI being built from scratch.",
    "bye":          "Goodbye! See you next session.",
    "goodbye":      "Goodbye! Keep learning.",
    "help":         "Commands: hello, how are you, what are you, history, clear, bye",
    "history":      "__HISTORY__",
    "clear":        "__CLEAR__",
}


def get_response(user_input):
    key = user_input.lower().strip()
    for pattern, response in RESPONSES.items():
        if pattern in key:
            return response
    return "I don't understand that yet. I am still learning!"


def run_chat(test_inputs=None):
    """
    Run the chat interface.
    If test_inputs is provided, runs in test mode (no input() calls).
    """
    print("=" * 40)
    print("  FNI-LLM Simple Chat  ")
    print("  Type 'bye' to exit   ")
    print("=" * 40)

    messages = []
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

        if user_input.lower() in ("bye", "goodbye", "exit", "quit"):
            print("FNI-LLM: Goodbye! Keep learning.")
            messages.append({"role": "user", "text": user_input})
            messages.append({"role": "bot", "text": "Goodbye! Keep learning."})
            break

        if user_input.lower() == "history":
            logs = list_conversations()
            if logs:
                print(f"FNI-LLM: Found {len(logs)} saved conversation(s): {logs}")
            else:
                print("FNI-LLM: No saved conversations yet.")
            continue

        if user_input.lower() == "clear":
            messages = []
            print("FNI-LLM: Conversation cleared.")
            continue

        response = get_response(user_input)
        print(f"FNI-LLM: {response}")

        messages.append({"role": "user",  "text": user_input})
        messages.append({"role": "bot",   "text": response})

    if messages:
        path = save_conversation(messages)
        print(f"\n[Conversation saved to {path}]")

    return messages


if __name__ == "__main__":
    run_chat()

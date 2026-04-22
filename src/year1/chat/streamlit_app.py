"""
Phase 1.5 - Browser Chat UI (Streamlit)
Wraps simple_chat and nn_chat in a browser interface
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import streamlit as st
import numpy as np

from src.year1.chat.logger import save_conversation, list_conversations, load_conversation
from src.year1.chat.simple_chat import get_response

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FNI-LLM Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("FNI-LLM Chat Interface")
st.caption("Year 1 Preview — Pattern Matching + XOR Neural Network")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Chat Mode", ["Simple Chat", "Neural Network (XOR)"])
    st.divider()

    st.header("Conversation History")
    logs = list_conversations()
    if logs:
        selected = st.selectbox("Load past conversation", ["-- select --"] + logs)
        if selected != "-- select --":
            data = load_conversation(selected)
            st.json(data)
    else:
        st.info("No saved conversations yet.")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- LOAD NN MODEL ONCE ---
@st.cache_resource
def load_nn_model():
    from src.year1.neural_network.network import NeuralNetwork
    np.random.seed(42)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    nn = NeuralNetwork([2, 4, 1], ['relu', 'sigmoid'])
    nn.train(X, y, epochs=5000, learning_rate=0.1, verbose=False)
    return nn

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MODE DESCRIPTION ---
if mode == "Simple Chat":
    st.info("Type: hello, how are you, what are you, help")
else:
    st.info("Enter two numbers (0 or 1) e.g. `0 1` or `1 1` to get XOR prediction")
    nn = load_nn_model()

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- CHAT INPUT ---
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    if mode == "Simple Chat":
        response = get_response(user_input)

    else:
        # NN mode — parse two numbers
        parts = user_input.replace(",", " ").split()
        if user_input.lower() == "accuracy":
            X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
            y_xor = np.array([[0],[1],[1],[0]], dtype=float)
            preds = np.round(nn.predict(X_xor))
            acc = np.mean(preds == y_xor)
            response = f"XOR accuracy: {acc:.0%} (4/4 correct)"
        elif len(parts) == 2:
            try:
                x1, x2 = float(parts[0]), float(parts[1])
                raw = nn.predict(np.array([[x1, x2]]))[0][0]
                result = round(raw)
                response = f"XOR({int(x1)}, {int(x2)}) = {result}  (confidence: {raw:.4f})"
            except ValueError:
                response = "Please enter two numbers like: 0 1"
        else:
            response = "Please enter two numbers like: 0 1  |  Commands: accuracy"

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

    # Auto-save conversation
    if len(st.session_state.messages) >= 2:
        log_msgs = [{"role": m["role"], "text": m["content"]}
                    for m in st.session_state.messages]
        save_conversation(log_msgs)

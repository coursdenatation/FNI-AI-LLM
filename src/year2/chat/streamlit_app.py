"""
Phase 2.5 - Year 2 Browser Chat UI (Streamlit)
Transformer-powered chat with language selector and generation controls
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="FNI-LLM | Cameroon AI",
    page_icon="🇨🇲",
    layout="centered"
)

st.title("FNI-LLM Chat")
st.caption("Cameroon Language AI — Year 2 Preview (Transformer Architecture)")

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

LANGUAGES = {
    "English (Cameroon)": "english",
    "French (Cameroon)":  "french (coming Year 3)",
    "Bayangi":            "bayangi (coming Year 3)",
    "Douala":             "douala (coming Year 3)",
}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")

    language = st.selectbox("Language", list(LANGUAGES.keys()))
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                            help="Lower = focused, Higher = creative")
    max_tokens = st.slider("Max new tokens", 3, 20, 8)
    st.divider()

    st.header("About")
    st.info(
        "This is a Year 2 preview. The Transformer has random weights "
        "and has not been trained yet.\n\n"
        "After Year 3 training on Cameroon language data, "
        "responses will be meaningful."
    )

    st.header("Language Roadmap")
    for lang, status in LANGUAGES.items():
        icon = "✅" if "english" in status else "⏳"
        st.write(f"{icon} {lang}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- LOAD MODEL ONCE ---
@st.cache_resource
def load_engine():
    from src.year2.chat.inference import InferenceEngine
    engine = InferenceEngine(
        vocab_size=150, d_model=32, num_heads=2,
        d_ff=64, num_layers=1
    )
    engine.build(CORPUS)
    return engine

@st.cache_resource
def load_generation():
    from src.year2.chat.generation import greedy_decode, temperature_sample
    return greedy_decode, temperature_sample

engine = load_engine()
greedy_decode, temperature_sample = load_generation()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- STATUS BAR ---
col1, col2, col3 = st.columns(3)
col1.metric("Model", "Transformer")
col2.metric("Vocab Size", engine.vocab.vocab_size)
col3.metric("Status", "Untrained")

st.divider()

# --- LANGUAGE WARNING ---
if LANGUAGES[language] != "english":
    st.warning(f"{language} training data will be added in Year 3. "
               f"Currently running in English mode.")

# --- CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- INPUT ---
user_input = st.chat_input("Type a prompt and the model will continue it...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Build context from history
    history = " ".join([m["content"] for m in st.session_state.messages
                        if m["role"] == "user"][-3:])

    # Generate
    with st.spinner("Generating..."):
        if temperature == 1.0:
            response = greedy_decode(engine, history, max_new_tokens=max_tokens)
        else:
            response = temperature_sample(engine, history,
                                          max_new_tokens=max_tokens,
                                          temperature=temperature)

    # Show only new tokens
    input_words    = history.strip().split()
    response_words = response.strip().split()
    new_words = response_words[len(input_words):]
    generated = " ".join(new_words) if new_words else response

    st.session_state.messages.append({"role": "assistant", "content": generated})
    with st.chat_message("assistant"):
        st.write(generated)
        st.caption(f"temp={temperature} | tokens={max_tokens} | untrained model")

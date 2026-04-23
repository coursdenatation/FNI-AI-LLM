"""
FNI-LLM Unified Chat UI
Connects Year 1 (XOR NN + Simple Chat) and Year 2 (Transformer) in one interface
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="FNI-LLM | Cameroon AI",
    page_icon="🇨🇲",
    layout="wide"
)

# --- HEADER ---
st.title("🇨🇲 FNI-LLM — Cameroon Language AI")
st.caption("Built from scratch | Year 1 + Year 2 Preview")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://flagcdn.com/w80/cm.png", width=60)
    st.markdown("### Navigation")
    mode = st.radio("Select Mode", [
        "Transformer Chat (Year 2)",
        "XOR Neural Network (Year 1)",
        "Simple Pattern Chat (Year 1)",
        "Project Overview",
    ])

    st.divider()
    st.markdown("### Language Roadmap")
    langs = [
        ("English", "✅ Active"),
        ("French", "⏳ Year 3"),
        ("Bayangi", "⏳ Year 3"),
        ("Douala", "⏳ Year 3"),
        ("Others", "⏳ Year 3+"),
    ]
    for lang, status in langs:
        st.write(f"{status} {lang}")

    st.divider()
    st.markdown("### Model Stats")
    st.metric("Architecture", "Transformer")
    st.metric("Year 2 Params", "~11K (untrained)")
    st.metric("Target (Year 4)", "100M–1B")


# ============================================================
# MODE 1: TRANSFORMER CHAT
# ============================================================
if mode == "Transformer Chat (Year 2)":
    st.subheader("Transformer Chat")
    st.info(
        "The Transformer has the correct architecture but **random weights** — "
        "it has not been trained yet. Outputs will be random until Year 3 "
        "training on Cameroon language data."
    )

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

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                                help="Lower=focused, Higher=creative")
    with col2:
        max_tokens = st.slider("Max new tokens", 3, 20, 8)

    @st.cache_resource
    def load_transformer():
        from src.year2.chat.inference import InferenceEngine
        engine = InferenceEngine(
            vocab_size=150, d_model=64, num_heads=4,
            d_ff=256, num_layers=2
        )
        engine.build(CORPUS)
        return engine

    @st.cache_resource
    def load_gen_fns():
        from src.year2.chat.generation import greedy_decode, temperature_sample
        return greedy_decode, temperature_sample

    engine = load_transformer()
    greedy_decode, temperature_sample = load_gen_fns()

    col1, col2, col3 = st.columns(3)
    col1.metric("Vocab Size", engine.vocab.vocab_size)
    col2.metric("Parameters", f"~{engine.model.count_params():,}")
    col3.metric("Status", "Untrained")

    if "t2_messages" not in st.session_state:
        st.session_state.t2_messages = []

    if st.button("Clear Chat", key="clear_t2"):
        st.session_state.t2_messages = []
        st.rerun()

    for msg in st.session_state.t2_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Type a prompt — the model will continue it...")
    if prompt:
        st.session_state.t2_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        context = " ".join([m["content"] for m in st.session_state.t2_messages
                            if m["role"] == "user"][-3:])

        with st.spinner("Generating..."):
            response = temperature_sample(engine, context,
                                          max_new_tokens=max_tokens,
                                          temperature=temperature)

        input_words    = context.strip().split()
        response_words = response.strip().split()
        new_words = response_words[len(input_words):]
        generated = " ".join(new_words) if new_words else response

        st.session_state.t2_messages.append({"role": "assistant", "content": generated})
        with st.chat_message("assistant"):
            st.write(generated)
            st.caption(f"temp={temperature} | tokens={max_tokens} | untrained")


# ============================================================
# MODE 2: XOR NEURAL NETWORK
# ============================================================
elif mode == "XOR Neural Network (Year 1)":
    st.subheader("XOR Neural Network Chat")
    st.success(
        "This model IS trained. It learned the XOR function with 100% accuracy. "
        "Enter two numbers (0 or 1) to get a prediction."
    )

    @st.cache_resource
    def load_xor_model():
        from src.year1.neural_network.network import NeuralNetwork
        np.random.seed(42)
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([[0],[1],[1],[0]], dtype=float)
        nn = NeuralNetwork([2, 4, 1], ['relu', 'sigmoid'])
        nn.train(X, y, epochs=5000, learning_rate=0.1, verbose=False)
        return nn

    nn = load_xor_model()

    # XOR truth table
    st.markdown("#### XOR Truth Table")
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_xor = np.array([[0],[1],[1],[0]], dtype=float)
    preds = nn.predict(X_xor)

    import pandas as pd
    df = pd.DataFrame({
        "Input A": [0, 0, 1, 1],
        "Input B": [0, 1, 0, 1],
        "Expected": [0, 1, 1, 0],
        "Predicted": [round(float(p[0]), 4) for p in preds],
        "Correct": ["✅" if round(float(p[0])) == int(y[0])
                    else "❌" for p, y in zip(preds, y_xor)]
    })
    st.dataframe(df, use_container_width=True)
    acc = np.mean(np.round(preds) == y_xor)
    st.metric("Accuracy", f"{acc:.0%}")

    st.markdown("#### Try It Yourself")
    col1, col2 = st.columns(2)
    with col1:
        a = st.selectbox("Input A", [0, 1])
    with col2:
        b = st.selectbox("Input B", [0, 1])

    if st.button("Predict XOR"):
        inp = np.array([[float(a), float(b)]])
        raw = nn.predict(inp)[0][0]
        result = round(raw)
        st.success(f"XOR({a}, {b}) = **{result}** (confidence: {raw:.4f})")

    # Loss curve
    st.markdown("#### Training Loss Curve")
    if os.path.exists("docs/visualizations/xor_loss.png"):
        st.image("docs/visualizations/xor_loss.png")

    # Decision boundary
    st.markdown("#### Decision Boundary")
    if os.path.exists("docs/visualizations/xor_decision_boundary.png"):
        st.image("docs/visualizations/xor_decision_boundary.png")


# ============================================================
# MODE 3: SIMPLE PATTERN CHAT
# ============================================================
elif mode == "Simple Pattern Chat (Year 1)":
    st.subheader("Simple Pattern Matching Chat")
    st.info("Rule-based chat using pattern matching. No neural network involved.")

    from src.year1.chat.simple_chat import get_response
    from src.year1.chat.logger import list_conversations, load_conversation

    if "s1_messages" not in st.session_state:
        st.session_state.s1_messages = []

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Clear", key="clear_s1"):
            st.session_state.s1_messages = []
            st.rerun()

        logs = list_conversations()
        if logs:
            selected = st.selectbox("Load history", ["--"] + logs[-5:])
            if selected != "--":
                data = load_conversation(selected)
                st.json(data["messages"][:3])

    with col1:
        st.caption("Try: hello, how are you, what are you, help")
        for msg in st.session_state.s1_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Say something...")
        if user_input:
            st.session_state.s1_messages.append(
                {"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            response = get_response(user_input)
            st.session_state.s1_messages.append(
                {"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)


# ============================================================
# MODE 4: PROJECT OVERVIEW
# ============================================================
elif mode == "Project Overview":
    st.subheader("FNI-LLM Project Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What We've Built")
        st.markdown("""
        **Year 1 — Foundations**
        - Neural network from scratch (NumPy only)
        - XOR problem solved (100% accuracy)
        - Backpropagation implemented manually
        - Batch training with mini-batch GD
        - Terminal + browser chat UI

        **Year 2 — Language Models**
        - Character, Word and BPE tokenizers
        - Vocabulary management (save/load)
        - RNN, LSTM, GRU from scratch
        - Full Transformer architecture
        - Multi-head attention
        - Positional encoding
        - Transformer chat UI
        """)

    with col2:
        st.markdown("### What's Coming")
        st.markdown("""
        **Year 3 — Cameroon Language Data**
        - English (Cameroon) corpus
        - French (Cameroon) corpus
        - Bayangi language data
        - Douala language data
        - Data cleaning pipeline
        - Language-specific tokenizers

        **Year 4 — Training & Deployment**
        - Train on Colab GPU
        - 10M–100M parameter model
        - FastAPI backend
        - Public web deployment
        - Target: best AI for Cameroon languages
        """)

    st.divider()
    st.markdown("### Visualizations")
    viz_dir = "docs/visualizations"
    if os.path.exists(viz_dir):
        images = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        if images:
            cols = st.columns(3)
            for i, img in enumerate(images):
                with cols[i % 3]:
                    st.image(os.path.join(viz_dir, img),
                             caption=img.replace('_', ' ').replace('.png', ''),
                             use_container_width=True)

    st.divider()
    st.markdown("### GitHub")
    st.markdown("[github.com/coursdenatation/FNI-AI-LLM]"
                "(https://github.com/coursdenatation/FNI-AI-LLM)")

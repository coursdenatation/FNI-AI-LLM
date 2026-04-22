"""
Phase 0.5 - Environment Verification Tests
Verifies all required packages are installed and working
"""

import pytest


def test_numpy():
    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    print(f"NumPy {np.__version__} OK")


def test_pandas():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert len(df) == 2
    print(f"Pandas {pd.__version__} OK")


def test_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    plt.close()
    print(f"Matplotlib {matplotlib.__version__} OK")


def test_sklearn():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    assert model is not None
    import sklearn
    print(f"Scikit-learn {sklearn.__version__} OK")


def test_streamlit():
    import streamlit
    print(f"Streamlit {streamlit.__version__} OK")


def test_neural_network():
    import numpy as np
    import sys
    import os
    sys.path.insert(0, os.path.abspath('.'))
    from src.year1.neural_network.network import NeuralNetwork
    np.random.seed(42)
    nn = NeuralNetwork([2, 4, 1], ['relu', 'sigmoid'])
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    out = nn.forward(X)
    assert out.shape == (4, 1)
    print("NeuralNetwork forward pass OK")


def test_pytorch_note():
    """PyTorch is used in Google Colab only (no local disk space)."""
    print("PyTorch: installed in Colab - skipping local check")
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

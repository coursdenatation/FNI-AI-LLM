"""
Phase 1.2 - Calculus Basics
Covers: derivatives, partial derivatives, gradient descent, visualizations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import os

os.makedirs('docs/visualizations', exist_ok=True)

# --- NUMERICAL DERIVATIVE ---
# derivative of f at x = (f(x+h) - f(x-h)) / (2h)

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def partial_derivative(f, args, index, h=1e-5):
    """Derivative of f with respect to args[index], others held fixed"""
    args_plus  = list(args)
    args_minus = list(args)
    args_plus[index]  += h
    args_minus[index] -= h
    return (f(*args_plus) - f(*args_minus)) / (2 * h)

# Test functions
f1 = lambda x: x ** 2          # f'(x) = 2x
f2 = lambda x: x ** 3          # f'(x) = 3x^2
f3 = lambda x: np.sin(x)       # f'(x) = cos(x)

print("=== DERIVATIVES ===")
for x in [1.0, 2.0, 3.0]:
    print(f"f(x)=x^2  at x={x}: numerical={derivative(f1,x):.4f}, exact={2*x:.4f}")
    print(f"f(x)=x^3  at x={x}: numerical={derivative(f2,x):.4f}, exact={3*x**2:.4f}")
    print(f"f(x)=sin  at x={x}: numerical={derivative(f3,x):.4f}, exact={np.cos(x):.4f}")
    print()

# --- PARTIAL DERIVATIVES ---
print("=== PARTIAL DERIVATIVES ===")

# f(x, y) = x^2 + 3xy + y^2
# df/dx = 2x + 3y
# df/dy = 3x + 2y
f_xy = lambda x, y: x**2 + 3*x*y + y**2

x, y = 2.0, 3.0
df_dx = partial_derivative(f_xy, [x, y], index=0)
df_dy = partial_derivative(f_xy, [x, y], index=1)
print(f"f(x,y) = x^2 + 3xy + y^2 at ({x},{y})")
print(f"df/dx numerical={df_dx:.4f}, exact={2*x + 3*y:.4f}")
print(f"df/dy numerical={df_dy:.4f}, exact={3*x + 2*y:.4f}")

# MSE loss partial derivatives (used in neural networks)
# L(w) = (y_pred - y_true)^2 = (w*x - y)^2
# dL/dw = 2*(w*x - y)*x
print("\n=== MSE LOSS DERIVATIVE ===")
mse = lambda w, x_val, y_val: (w * x_val - y_val) ** 2
x_val, y_val = 2.0, 4.0
for w in [0.5, 1.0, 1.5, 2.0]:
    loss = mse(w, x_val, y_val)
    grad = partial_derivative(lambda w: mse(w, x_val, y_val), [w], index=0)
    exact_grad = 2 * (w * x_val - y_val) * x_val
    print(f"w={w:.1f}: loss={loss:.4f}, grad={grad:.4f}, exact={exact_grad:.4f}")

# --- GRADIENT DESCENT ---
print("\n=== GRADIENT DESCENT ===")

def gradient_descent(f, start, learning_rate=0.1, epochs=50):
    x = start
    history = [x]
    for _ in range(epochs):
        grad = derivative(f, x)
        x = x - learning_rate * grad
        history.append(x)
    return x, history

# Minimize f(x) = x^2 (minimum at x=0)
f_simple = lambda x: x ** 2
final, history = gradient_descent(f_simple, start=5.0, learning_rate=0.1, epochs=30)
print(f"Minimizing x^2 starting at x=5.0")
print(f"Final x={final:.6f} (should be ~0)")
print(f"Steps: {len(history)}")

# Minimize f(x) = (x-3)^2 (minimum at x=3)
f_shifted = lambda x: (x - 3) ** 2
final2, history2 = gradient_descent(f_shifted, start=0.0, learning_rate=0.1, epochs=50)
print(f"\nMinimizing (x-3)^2 starting at x=0.0")
print(f"Final x={final2:.6f} (should be ~3)")

# Learning rate comparison
print("\n=== LEARNING RATE EFFECT ===")
for lr in [0.01, 0.1, 0.5, 0.9]:
    final_lr, hist_lr = gradient_descent(f_simple, start=5.0, learning_rate=lr, epochs=50)
    print(f"lr={lr}: final x={final_lr:.6f} after {len(hist_lr)} steps")

# --- VISUALIZATIONS ---
x_vals = np.linspace(-3, 3, 200)

# Plot 1: Functions and their derivatives
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
funcs = [
    (lambda x: x**2,      lambda x: 2*x,      "x^2",   "2x"),
    (lambda x: x**3,      lambda x: 3*x**2,   "x^3",   "3x^2"),
    (lambda x: np.sin(x), lambda x: np.cos(x),"sin(x)","cos(x)"),
]
for ax, (f, df, fname, dfname) in zip(axes, funcs):
    ax.plot(x_vals, [f(x) for x in x_vals],  label=f"f={fname}")
    ax.plot(x_vals, [df(x) for x in x_vals], label=f"f'={dfname}", linestyle='--')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.set_title(f"f(x) = {fname}")
    ax.grid(True)
plt.tight_layout()
plt.savefig('docs/visualizations/derivatives.png', dpi=100)
plt.close()
print("\nSaved: docs/visualizations/derivatives.png")

# Plot 2: Gradient descent convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history)
axes[0].set_title("Gradient Descent: x^2 from x=5")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("x value")
axes[0].grid(True)

for lr in [0.01, 0.1, 0.3, 0.5]:
    _, h = gradient_descent(f_simple, start=5.0, learning_rate=lr, epochs=50)
    axes[1].plot([f_simple(x) for x in h], label=f"lr={lr}")
axes[1].set_title("Loss vs Steps (different LRs)")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('docs/visualizations/gradient_descent.png', dpi=100)
plt.close()
print("Saved: docs/visualizations/gradient_descent.png")


if __name__ == "__main__":
    print("\nDone: Calculus basics complete.")

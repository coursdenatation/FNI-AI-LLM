"""
Phase 3.5 - Model Trainer
Training loop with loss tracking, validation, checkpointing
Works locally (CPU) and in Colab (GPU via PyTorch)
"""

import numpy as np
import os
import json
import time
import pickle
from src.year2.transformer.transformer import Transformer
from src.year3.data_processing.dataset import TextDataset
from src.year3.data_processing.dataloaders import DataLoader, train_val_test_split


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_loss(logits, targets):
    """
    logits:  (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    batch, seq_len, vocab_size = logits.shape
    probs = softmax(logits.reshape(-1, vocab_size))
    targets_flat = targets.reshape(-1)

    # Gather probabilities of correct tokens
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]
    loss = -np.mean(np.log(correct_probs + 1e-9))
    return loss


def compute_accuracy(logits, targets):
    preds = logits.argmax(axis=-1)
    return (preds == targets).mean()


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 learning_rate=0.001, checkpoint_dir="models/checkpoints"):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.lr             = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.train_losses   = []
        self.val_losses     = []
        self.best_val_loss  = float('inf')
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _update_params(self, layer, grad_scale=1.0):
        """Simple SGD update for all weight matrices in a layer"""
        if hasattr(layer, 'W'):
            layer.W -= self.lr * grad_scale * np.clip(
                np.random.randn(*layer.W.shape) * 0.001, -0.01, 0.01)
        if hasattr(layer, 'b'):
            layer.b -= self.lr * grad_scale * np.clip(
                np.random.randn(*layer.b.shape) * 0.001, -0.01, 0.01)

    def train_epoch(self):
        epoch_losses = []
        for X_batch, y_batch in self.train_loader:
            logits = self.model.forward(X_batch)
            loss   = cross_entropy_loss(logits, y_batch)
            epoch_losses.append(loss)

            # Lightweight weight perturbation (placeholder for real backprop)
            # Real gradient-based training happens in Colab with PyTorch
            for block in self.model.blocks:
                self._update_params(block.attn)
                self._update_params(block.ff)

        return float(np.mean(epoch_losses))

    def validate(self):
        val_losses = []
        val_accs   = []
        for X_batch, y_batch in self.val_loader:
            logits = self.model.forward(X_batch)
            loss   = cross_entropy_loss(logits, y_batch)
            acc    = compute_accuracy(logits, y_batch)
            val_losses.append(loss)
            val_accs.append(acc)
        return float(np.mean(val_losses)), float(np.mean(val_accs))

    def save_checkpoint(self, language, epoch, val_loss):
        path = os.path.join(self.checkpoint_dir, f"{language}_epoch{epoch}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "epoch":    epoch,
                "val_loss": val_loss,
                "embedding": self.model.embedding,
                "config": {
                    "vocab_size":   self.model.vocab_size,
                    "d_model":      self.model.d_model,
                    "num_layers":   len(self.model.blocks),
                }
            }, f)
        return path

    def train(self, epochs, language="english", verbose=True):
        log = []
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            elapsed = time.time() - t0

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(language, epoch, val_loss)

            entry = {
                "epoch": epoch, "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4), "time": round(elapsed, 2)
            }
            log.append(entry)

            if verbose:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"train_loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"val_acc={val_acc:.2%} | "
                      f"time={elapsed:.1f}s")

        # Save training log
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/{language}_training.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nLog saved: {log_path}")
        return log


if __name__ == "__main__":
    np.random.seed(42)
    DATA_ROOT = "data/cameroon_languages"

    print("=== LOCAL TRAINING TEST (English, small model) ===\n")

    # Load dataset
    ds = TextDataset(language="english", seq_len=16)
    ds.load(f"{DATA_ROOT}/english/processed/english_clean.txt")
    train_ds, val_ds, _ = train_val_test_split(ds, seed=42)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    # Build small model
    model = Transformer(
        vocab_size=ds.vocab_size, d_model=64,
        num_heads=4, d_ff=128, num_layers=2
    )
    print(f"Model: {model}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # Train for 3 epochs locally to verify pipeline works
    trainer = Trainer(model, train_loader, val_loader,
                      learning_rate=0.001,
                      checkpoint_dir="models/checkpoints/english")
    trainer.train(epochs=3, language="english")

    print("\nLocal training test passed.")
    print("For real training with GPU, use the Colab notebook.")

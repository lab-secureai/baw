from __future__ import annotations

"""Compact numpy neural network with what BAW needs:

    - Two-hidden-layer MLP with ReLU
    - Adam optimizer
    - Full backprop for parameters (training / fine-tuning)
    - Gradient with respect to the INPUT (needed for PGD trigger crafting)

Nothing here is exotic; the value of the numpy implementation is that BAW
becomes reproducible with a single `pip install numpy` — no framework
version pinning, no CUDA, no wheel.
"""
import copy
import numpy as np


# ---------------------------------------------------------------------------
# Numerically stable softmax + cross-entropy
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y):
    p = softmax(logits)
    n = logits.shape[0]
    return -np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0)).mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP:
    """Two-hidden-layer ReLU MLP: (in → h → h → out).

    All state is in a flat parameter dict for easy cloning / distillation.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, rng=None):
        rng = rng or np.random.default_rng(0)
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.params = {
            "W1": rng.standard_normal((in_dim, hidden)).astype(np.float32) * np.sqrt(2 / in_dim),
            "b1": np.zeros(hidden, dtype=np.float32),
            "W2": rng.standard_normal((hidden, hidden)).astype(np.float32) * np.sqrt(2 / hidden),
            "b2": np.zeros(hidden, dtype=np.float32),
            "W3": rng.standard_normal((hidden, out_dim)).astype(np.float32) * np.sqrt(2 / hidden),
            "b3": np.zeros(out_dim, dtype=np.float32),
        }

    # ---- forward -----------------------------------------------------------
    def forward(self, X, cache: bool = False):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0)
        z3 = a2 @ p["W3"] + p["b3"]
        if not cache:
            return z3
        return z3, {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    # ---- backward w.r.t. parameters ---------------------------------------
    def backward_params(self, y, cached):
        p = self.params
        logits = cached["a2"] @ p["W3"] + p["b3"]
        probs = softmax(logits)
        n = y.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n

        grads = {}
        grads["W3"] = cached["a2"].T @ dlogits
        grads["b3"] = dlogits.sum(axis=0)

        da2 = dlogits @ p["W3"].T
        dz2 = da2 * (cached["z2"] > 0)

        grads["W2"] = cached["a1"].T @ dz2
        grads["b2"] = dz2.sum(axis=0)

        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (cached["z1"] > 0)

        grads["W1"] = cached["X"].T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return grads

    # ---- backward w.r.t. INPUT (for PGD) ----------------------------------
    def backward_input(self, X, y_target):
        """∂CE(f(X), y_target) / ∂X, used by PGD to perturb inputs."""
        _, cached = self.forward(X, cache=True)
        p = self.params
        logits = cached["a2"] @ p["W3"] + p["b3"]
        probs = softmax(logits)
        n = X.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(n), y_target] -= 1
        dlogits /= n

        da2 = dlogits @ p["W3"].T
        dz2 = da2 * (cached["z2"] > 0)
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (cached["z1"] > 0)
        dX = dz1 @ p["W1"].T
        return dX

    def clone(self) -> "MLP":
        c = MLP.__new__(MLP)
        c.in_dim, c.hidden, c.out_dim = self.in_dim, self.hidden, self.out_dim
        c.params = {k: v.copy() for k, v in self.params.items()}
        return c


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, params: dict, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train(model: MLP, X, y, epochs: int, lr: float, batch_size: int = 256,
          weight_decay: float = 1e-4, rng=None, verbose: bool = False):
    rng = rng or np.random.default_rng(0)
    opt = Adam(model.params, lr=lr, weight_decay=weight_decay)
    n = X.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            j = idx[i:i + batch_size]
            xb, yb = X[j], y[j]
            logits, cache = model.forward(xb, cache=True)
            losses.append(cross_entropy(logits, yb))
            grads = model.backward_params(yb, cache)
            opt.step(model.params, grads)
        if verbose:
            print(f"    epoch {ep+1:>2}: loss={np.mean(losses):.4f}")
    return model


def train_mixed(model: MLP, X_clean, y_clean, X_trig, y_trig, epochs, lr,
                trigger_weight=1.0, batch_size=256, weight_decay=1e-4,
                rng=None):
    """Fine-tune on clean + trigger, weighting the trigger loss.

    This is the watermark embedding routine. We interleave clean and trigger
    batches so the model does not "forget" the trigger set between epochs.
    """
    rng = rng or np.random.default_rng(0)
    opt = Adam(model.params, lr=lr, weight_decay=weight_decay)
    n_clean, n_trig = X_clean.shape[0], X_trig.shape[0]
    trig_bs = min(64, n_trig)

    for _ in range(epochs):
        idx = rng.permutation(n_clean)
        t_pos = 0
        t_order = rng.permutation(n_trig)
        for i in range(0, n_clean, batch_size):
            j = idx[i:i + batch_size]
            xb, yb = X_clean[j], y_clean[j]

            # Clean gradient
            logits, cache = model.forward(xb, cache=True)
            grads_c = model.backward_params(yb, cache)

            # Trigger gradient (mini-batch drawn cyclically)
            if t_pos + trig_bs > n_trig:
                t_order = rng.permutation(n_trig)
                t_pos = 0
            k = t_order[t_pos:t_pos + trig_bs]; t_pos += trig_bs
            xt, yt = X_trig[k], y_trig[k]
            logits_t, cache_t = model.forward(xt, cache=True)
            grads_t = model.backward_params(yt, cache_t)

            # Combined update
            grads = {k_: grads_c[k_] + trigger_weight * grads_t[k_]
                     for k_ in grads_c}
            opt.step(model.params, grads)
    return model


def accuracy(model: MLP, X, y):
    return (model.predict(X) == y).mean()


# ---------------------------------------------------------------------------
# PGD: minimize CE(f(x), target) subject to ‖x − x0‖_∞ ≤ ε
# ---------------------------------------------------------------------------
def pgd_towards_class(model: MLP, X0, target_class: int, eps: float,
                      alpha: float, steps: int, x_lo=None, x_hi=None):
    """Push each row of X0 toward `target_class` in model's decision.

    Returns X_adv of the same shape. Frozen model, standard L∞ PGD.
    """
    n = X0.shape[0]
    target = np.full(n, target_class, dtype=np.int64)
    X_adv = X0.copy()
    lo = (X0 - eps) if x_lo is None else np.maximum(X0 - eps, x_lo)
    hi = (X0 + eps) if x_hi is None else np.minimum(X0 + eps, x_hi)
    for _ in range(steps):
        g = model.backward_input(X_adv, target)
        # Minimize CE(target): step in the negative gradient direction
        X_adv = X_adv - alpha * np.sign(g)
        X_adv = np.clip(X_adv, lo, hi)
    return X_adv

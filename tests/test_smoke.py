import numpy as np

from baw.exploit import direct_exploit_count
from baw.nn import MLP, accuracy, train


def test_mlp_can_fit_tiny_binary_problem():
    rng = np.random.default_rng(7)
    x0 = rng.normal(-1.0, 0.2, size=(32, 4))
    x1 = rng.normal(+1.0, 0.2, size=(32, 4))
    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.array([0] * len(x0) + [1] * len(x1), dtype=np.int64)

    model = MLP(4, 16, 2, rng=np.random.default_rng(8))
    train(
        model,
        X,
        y,
        epochs=30,
        lr=1e-2,
        batch_size=16,
        weight_decay=0.0,
        rng=np.random.default_rng(9),
    )
    assert accuracy(model, X, y) > 0.95


def test_direct_exploit_count_uses_true_carrier_labels():
    class AlwaysBenign:
        def predict(self, X):
            return np.zeros(len(X), dtype=np.int64)

    X = np.zeros((4, 3), dtype=np.float32)
    y_true = np.array([0, 1, 1, 0], dtype=np.int64)
    assert direct_exploit_count(X, y_true, AlwaysBenign()) == 2

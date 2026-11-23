"""Refit scoring weights from labelled addresses, anchored to the expert defaults.

Plain logistic regression would discard everything known about features that
are rare in a small labelled sample. Instead this minimises

    NLL(y | X, w, b) + l2 * ||w - w_prior||^2

so weights only move away from the defaults where the labels give evidence.
Use ``l2`` around 0.5-5 for a few hundred labels; smaller values trust the data more.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .scoring import feature_names


def fit_weights(feats: pd.DataFrame, y: np.ndarray, cfg: dict, l2: float = 1.0) -> dict:
    names = feature_names(cfg)
    X = feats[[f"f_{n}" for n in names]].to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    w0 = np.array([float(cfg["scoring"]["weights"][n]) for n in names])
    b0 = float(cfg["scoring"]["intercept"])

    def objective(theta):
        b, w = theta[0], theta[1:]
        z = b + X @ w
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        diff = w - w0
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        grad = np.concatenate([[g.sum() + 2 * 0.01 * (b - b0)], X.T @ g + 2 * l2 * diff])
        return nll + l2 * diff @ diff + 0.01 * (b - b0) ** 2, grad

    res = minimize(objective, np.concatenate([[b0], w0]), jac=True, method="L-BFGS-B")
    b, w = res.x[0], res.x[1:]
    support = (X != 0).sum(axis=0)
    return {
        "intercept": round(float(b), 4),
        "weights": {n: round(float(v), 4) for n, v in zip(names, w)},
        "prior_weights": {n: float(v) for n, v in zip(names, w0)},
        "support": {n: int(s) for n, s in zip(names, support)},
        "l2": l2, "n_train": int(len(y)), "n_positive": int(y.sum()), "converged": bool(res.success),
    }


def save_weights(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")

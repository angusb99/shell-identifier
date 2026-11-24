"""Learning which addresses are dwellings from parcel dwelling counts and misplaced address labels.

Setting
-------
Each parcel has addresses i = 1..n with a hidden indicator z_i (private dwelling
or not) and a known count K = sum_i z_i. A labeller marked L addresses PD. The
count is right, but some PD labels may sit on the wrong address: a shell instead
of a unit, ``10`` instead of ``10B``.

Model
-----
* A classifier f(x_i) = P(z_i = 1 | features), shared by every parcel.
* A *misplacement* model for the labels. With A = addresses labelled PD and
  B = the rest, a configuration z has u PD labels on non-dwellings (zeros in A)
  and v dwellings without a PD label (ones in B), v - u = K - L. The number of
  misplacements m = min(u, v) has a prior P(m), and each of the C(|A|,u)C(|B|,v)
  ways to misplace is equally likely:

      P(z | labels, x)  ∝  prod_i f_i^z_i (1-f_i)^(1-z_i)  *  P(m) / (C(|A|,u) C(|B|,v))

  This fits "right count, wrong address" far better than independent per-address
  label noise, which over-penalises a single swap in a two- or three-address parcel.

EM
--
E-step: exact posterior marginals q_i via truncated Poisson-binomial
forward/backward recursions over A (counting zeros) and B (counting ones);
cost O(n * M) per parcel with M the largest misplacement count considered.
M-step: refit the classifier on soft targets q; re-estimate P(m) from the
posterior over m (Dirichlet-smoothed towards the initial prior).

The classifier is cross-fitted by parcel, so a parcel's f(x) never comes from a
model that saw its own labels. The final assignment is the exact MAP
configuration under the same model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.special import gammaln, logsumexp


def factorize(values) -> tuple[np.ndarray, list]:
    """Integer codes in order of first appearance, and the distinct values."""
    s = pl.Series(np.asarray(values).astype(str))
    uniques = s.unique(maintain_order=True)
    codes = s.replace_strict(uniques, pl.Series(np.arange(len(uniques), dtype=np.int64))).to_numpy()
    return codes, uniques.to_list()


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _log_comb(n: int, k: int) -> float:
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def _forward(lp_hit: np.ndarray, lp_miss: np.ndarray, cmax: int) -> np.ndarray:
    """fwd[j, c] = log P(c hits among the first j items), c <= cmax."""
    n = len(lp_hit)
    fwd = np.full((n + 1, cmax + 1), -np.inf)
    fwd[0, 0] = 0.0
    for j in range(n):
        fwd[j + 1, 0] = fwd[j, 0] + lp_miss[j]
        if cmax:
            fwd[j + 1, 1:] = np.logaddexp(fwd[j, 1:] + lp_miss[j], fwd[j, :-1] + lp_hit[j])
    return fwd


def _leave_one_out(lp_hit: np.ndarray, lp_miss: np.ndarray, cmax: int) -> np.ndarray:
    """out[i, c] = log P(c hits among all items except i), c <= cmax."""
    n = len(lp_hit)
    fwd = _forward(lp_hit, lp_miss, cmax)
    bwd = _forward(lp_hit[::-1], lp_miss[::-1], cmax)[::-1]   # bwd[j] covers items j..n-1
    out = np.full((n, cmax + 1), -np.inf)
    for c in range(cmax + 1):
        a = np.arange(c + 1)
        out[:, c] = logsumexp(fwd[:n, a] + bwd[1:, c - a], axis=1)
    return out


def conditional_marginals(logits: np.ndarray, k: int) -> np.ndarray:
    """P(z_i = 1 | sum z = k) for independent z_i ~ Bernoulli(sigmoid(logits_i)) (no labels)."""
    logits = np.asarray(logits, dtype=float)
    n = len(logits)
    if k <= 0:
        return np.zeros(n)
    if k >= n:
        return np.ones(n)
    lp1, lp0 = _log_sigmoid(logits), _log_sigmoid(-logits)
    total = _forward(lp1, lp0, k)[n, k]
    loo = _leave_one_out(lp1, lp0, k - 1)
    return np.clip(np.exp(lp1 + loo[:, k - 1] - total), 0.0, 1.0)


def _pairs(L: int, nB: int, k: int, M: int) -> list[tuple[int, int, int]]:
    """Feasible (u, v, m): u PD labels wrong, v unlabelled dwellings, m = min(u, v) <= M."""
    shift = k - L
    out = []
    for u in range(L + 1):
        v = u + shift
        if 0 <= v <= nB and min(u, v) <= M:
            out.append((u, v, min(u, v)))
    return out


def swap_posterior(logits: np.ndarray, labelled: np.ndarray, k: int, log_prior_m: np.ndarray):
    """Posterior dwelling marginals under the misplacement model.

    Returns ``(q, post_m)``: q_i = P(z_i = 1 | count, labels, f) and the posterior
    over the number of misplacements m = 0..M for this parcel.
    """
    logits = np.asarray(logits, dtype=float)
    n = len(logits)
    M = len(log_prior_m) - 1
    post_m = np.zeros(M + 1)
    A, B = np.flatnonzero(labelled), np.flatnonzero(~labelled)
    L, nB = len(A), len(B)
    if k <= 0 or k >= n:
        post_m[0] = 1.0
        return (np.zeros(n) if k <= 0 else np.ones(n)), post_m
    pairs = _pairs(L, nB, k, M)
    if not pairs:  # labels too far from the count to be explained by <= M misplacements
        return conditional_marginals(logits, k), post_m
    lp1, lp0 = _log_sigmoid(logits), _log_sigmoid(-logits)
    u_max = max(p[0] for p in pairs)
    v_max = max(p[1] for p in pairs)
    DA = _forward(lp0[A], lp1[A], u_max)[L]       # log P(u of the PD-labelled are not dwellings)
    DB = _forward(lp1[B], lp0[B], v_max)[nB]      # log P(v of the others are dwellings)
    lg = np.array([log_prior_m[m] - _log_comb(L, u) - _log_comb(nB, v) for u, v, m in pairs])
    terms = lg + np.array([DA[u] + DB[v] for u, v, _ in pairs])
    log_z = logsumexp(terms)
    w = np.exp(terms - log_z)
    for (_, _, m), wi in zip(pairs, w):
        post_m[m] += wi

    q = np.zeros(n)
    flip_a = np.zeros(L)
    if L and u_max >= 1:
        loo_a = _leave_one_out(lp0[A], lp1[A], u_max - 1)
        for (u, v, _), g in zip(pairs, lg):
            if u >= 1:
                flip_a += np.exp(g + lp0[A] + loo_a[:, u - 1] + DB[v] - log_z)
    q[A] = 1.0 - flip_a
    if nB and v_max >= 1:
        loo_b = _leave_one_out(lp1[B], lp0[B], v_max - 1)
        for (u, v, _), g in zip(pairs, lg):
            if v >= 1:
                q[B] += np.exp(g + DA[u] + lp1[B] + loo_b[:, v - 1] - log_z)
    return np.clip(q, 0.0, 1.0), post_m


def swap_map(logits: np.ndarray, labelled: np.ndarray, k: int, log_prior_m: np.ndarray) -> np.ndarray:
    """Most probable configuration: drop the u least dwelling-like PD labels, add the v most dwelling-like others."""
    logits = np.asarray(logits, dtype=float)
    n = len(logits)
    z = labelled.copy().astype(bool)
    if k <= 0 or k >= n:
        return np.full(n, k > 0)
    A, B = np.flatnonzero(labelled), np.flatnonzero(~labelled)
    pairs = _pairs(len(A), len(B), k, len(log_prior_m) - 1)
    if not pairs:
        z = np.zeros(n, dtype=bool)
        z[np.argsort(-logits, kind="stable")[:k]] = True
        return z
    a_order = A[np.argsort(logits[A], kind="stable")]           # least dwelling-like first
    b_order = B[np.argsort(-logits[B], kind="stable")]          # most dwelling-like first
    gain_a = np.concatenate([[0.0], np.cumsum(-logits[a_order])])  # log(1-p) - log p = -logit
    gain_b = np.concatenate([[0.0], np.cumsum(logits[b_order])])
    best, best_score = None, -np.inf
    for u, v, m in pairs:
        sc = log_prior_m[m] - _log_comb(len(A), u) - _log_comb(len(B), v) + gain_a[u] + gain_b[v]
        if sc > best_score + 1e-12:
            best, best_score = (u, v), sc
    u, v = best
    z = labelled.copy().astype(bool)
    z[a_order[:u]] = False
    z[b_order[:v]] = True
    return z


@dataclass
class EMSettings:
    folds: int = 5
    max_iter: int = 8
    tol: float = 0.0005                  # stop when fewer than this share of assignments change
    max_misplaced: int = 3               # largest number of misplaced PD labels per parcel considered
    misplacement_prior: tuple = (0.75, 0.18, 0.05, 0.02)  # initial P(m = 0, 1, 2, 3)
    prior_strength: float = 50.0         # Dirichlet pseudo-counts behind the prior
    estimate_prior: bool = True
    trivial_sample: float = 0.25         # share of single-address parcels used for training
    clf_params: dict = field(default_factory=lambda: dict(
        max_iter=150, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    ))
    random_state: int = 0


def fit_soft(X: np.ndarray, q: np.ndarray, params: dict, weight: np.ndarray | None = None):
    """Fit a classifier to soft targets: confident rows once, uncertain rows twice weighted q / 1-q."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    weight = np.ones(len(q)) if weight is None else weight
    hard = (q <= 0.01) | (q >= 0.99)
    soft = ~hard
    Xs = np.vstack([X[hard], X[soft], X[soft]])
    ys = np.concatenate([(q[hard] >= 0.5).astype(int), np.ones(soft.sum(), int), np.zeros(soft.sum(), int)])
    ws = np.concatenate([weight[hard], weight[soft] * q[soft], weight[soft] * (1.0 - q[soft])])
    if len(np.unique(ys)) < 2:  # degenerate: add a weightless counter-example
        Xs, ys, ws = np.vstack([Xs, Xs[:1]]), np.append(ys, 1 - ys[0]), np.append(ws, 1e-6)
    return HistGradientBoostingClassifier(**params).fit(Xs, ys, sample_weight=ws)


class CountConstrainedEM:
    """EM over hidden dwelling indicators with per-parcel counts and misplaced labels."""

    def __init__(self, settings: EMSettings | None = None):
        self.s = settings or EMSettings()
        prior = np.asarray(self.s.misplacement_prior[: self.s.max_misplaced + 1], dtype=float)
        self.prior0 = prior / prior.sum()
        self.log_prior_m = np.log(self.prior0)
        self.history: list[dict] = []
        self.model = None

    def _e_step(self, logits, groups, k, labelled):
        q = np.empty(len(logits))
        post = np.zeros(len(self.prior0))
        for idx, kk in zip(groups, k):
            q[idx], pm = swap_posterior(logits[idx], labelled[idx], int(kk), self.log_prior_m)
            if 0 < kk < len(idx) and labelled[idx].any() and (~labelled[idx]).any():
                post += pm
        return q, post

    def _map(self, logits, groups, k, labelled):
        z = np.zeros(len(logits), dtype=bool)
        for idx, kk in zip(groups, k):
            z[idx] = swap_map(logits[idx], labelled[idx], int(kk), self.log_prior_m)
        return z

    def fit(self, X: np.ndarray, parcel: np.ndarray, k_parcel: Mapping[str, int], y: np.ndarray, log=print):
        """Fit on labelled parcels.

        ``X`` rows are addresses, ``parcel`` their parcel ids, ``k_parcel`` the
        dwelling count per parcel id and ``y`` the address labels (1 PD, 0 out of
        scope, -1 unknown). Sets ``oof_prob``, ``posterior`` and ``assigned``.
        """
        from sklearn.model_selection import GroupKFold

        n = len(X)
        labelled = y == 1
        codes, uniques = factorize(parcel)
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
        groups = [order[bounds[g]:bounds[g + 1]] for g in range(len(uniques))]
        sizes = np.array([len(g) for g in groups])
        k = np.minimum(np.array([int(k_parcel.get(u, 0)) for u in uniques], dtype=int), sizes)

        # Single-address parcels have a fixed answer: subsample them for training, reweighted.
        rng = np.random.default_rng(self.s.random_state)
        trivial = sizes[codes] == 1
        keep = ~trivial | (rng.random(n) < self.s.trivial_sample)
        weight = np.where(trivial, 1.0 / self.s.trivial_sample, 1.0)
        splits = list(GroupKFold(n_splits=min(self.s.folds, max(2, len(uniques)))).split(X, groups=codes))

        base = float(np.clip(labelled.mean(), 0.01, 0.99))
        logits = np.full(n, _logit(np.array([base]))[0])
        q, _ = self._e_step(logits, groups, k, labelled)
        z = self._map(logits, groups, k, labelled)
        prob = np.full(n, base)
        for it in range(self.s.max_iter):
            for train, test in splits:
                tr = train[keep[train]]
                clf = fit_soft(X[tr], q[tr], self.s.clf_params, weight[tr])
                prob[test] = clf.predict_proba(X[test])[:, 1]
            logits = _logit(prob)
            q, post = self._e_step(logits, groups, k, labelled)
            if self.s.estimate_prior and post.sum() > 0:
                counts = post + self.s.prior_strength * self.prior0
                self.log_prior_m = np.log(counts / counts.sum())
            z_new = self._map(logits, groups, k, labelled)
            changed = int((z_new != z).sum())
            z = z_new
            moved = int((labelled & ~z).sum())
            pm = np.exp(self.log_prior_m)
            self.history.append({"iteration": it + 1, "assignments_changed": changed, "labels_moved": moved,
                                 "p_misplaced": round(float(1 - pm[0]), 4)})
            log(f"EM iteration {it + 1}: {changed} assignments changed, {moved} PD labels moved, "
                f"P(parcel has a misplaced label) = {1 - pm[0]:.3f}")
            if it > 0 and changed <= self.s.tol * n:
                break
        self.oof_prob, self.posterior, self.assigned = prob, q, z
        self.model = fit_soft(X[keep], q[keep], self.s.clf_params, weight[keep])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

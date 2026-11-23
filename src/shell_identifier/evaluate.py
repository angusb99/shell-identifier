"""Evaluation of scores against labelled truth (synthetic or hand-labelled)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) statistic, ties averaged."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": (tp + tn) / len(y)}


def join_truth(scores: pd.DataFrame, truth: pd.DataFrame, id_col: str = "address_pid",
               truth_id: str = "address_detail_pid", label: str = "is_shell") -> pd.DataFrame:
    t = truth.rename(columns={truth_id: id_col})
    m = scores.merge(t, on=id_col, how="inner", suffixes=("", "_truth"))
    m[label] = m[label].astype(int)
    return m


def evaluate(merged: pd.DataFrame, label: str = "is_shell") -> dict:
    """Metrics for every scored address and for the *contested* subset.

    The contested subset is what the tool is really for: addresses that have
    something nested, ranged or suffixed around them (``f_leaf_*`` both zero).
    Leaf addresses are trivially not shells and inflate overall accuracy.
    """
    y = merged[label].to_numpy()
    p = merged["shell_probability"].to_numpy()
    out = {
        "n": int(len(merged)), "n_shell": int(y.sum()), "auc": roc_auc(y, p),
        "at_0.5": binary_metrics(y, p, 0.5), "at_0.8": binary_metrics(y, p, 0.8),
    }
    contested = merged[(merged.get("f_leaf_subaddress", 0) == 0) & (merged.get("f_leaf_street_address", 0) == 0)]
    if len(contested):
        yc, pc = contested[label].to_numpy(), contested["shell_probability"].to_numpy()
        out["contested"] = {"n": int(len(contested)), "n_shell": int(yc.sum()), "auc": roc_auc(yc, pc),
                            "at_0.5": binary_metrics(yc, pc, 0.5)}
    out["class_by_truth"] = pd.crosstab(merged["shell_class"], merged[label]).to_dict()
    return out


def breakdown(merged: pd.DataFrame, by: str = "role", label: str = "is_shell") -> pd.DataFrame:
    """Accuracy and probability spread per truth group (e.g. role or scenario)."""
    m = merged.assign(_correct=(merged["shell_probability"] >= 0.5) == merged[label].astype(bool),
                      _uncertain=merged["shell_class"].eq("UNCERTAIN"))
    g = m.groupby(by)
    return pd.DataFrame({
        "n": g.size(),
        "truth_shell_rate": g[label].mean().round(3),
        "accuracy": g["_correct"].mean().round(3),
        "uncertain_rate": g["_uncertain"].mean().round(3),
        "p_median": g["shell_probability"].median().round(3),
        "p_min": g["shell_probability"].min().round(3),
        "p_max": g["shell_probability"].max().round(3),
    }).sort_values(["accuracy", "n"], ascending=[True, False])


def markdown_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (avoids a tabulate dependency)."""
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def report_markdown(metrics: dict, by_role: pd.DataFrame, errors: pd.DataFrame | None = None) -> str:
    def fmt(d):
        return (f"precision {d['precision']:.3f}, recall {d['recall']:.3f}, F1 {d['f1']:.3f}, "
                f"accuracy {d['accuracy']:.4f} (TP {d['tp']}, FP {d['fp']}, FN {d['fn']}, TN {d['tn']})")

    lines = [
        "# Shell identifier evaluation", "",
        f"* Addresses scored against truth: **{metrics['n']}** ({metrics['n_shell']} shells)",
        f"* ROC AUC: **{metrics['auc']:.4f}**",
        f"* At p >= 0.5: {fmt(metrics['at_0.5'])}",
        f"* At p >= 0.8: {fmt(metrics['at_0.8'])}",
    ]
    if "contested" in metrics:
        c = metrics["contested"]
        lines += ["", "## Contested addresses only",
                  "Addresses with something nested, ranged or suffixed around them (the cases that need judgement).", "",
                  f"* n = {c['n']} ({c['n_shell']} shells), ROC AUC {c['auc']:.4f}",
                  f"* At p >= 0.5: {fmt(c['at_0.5'])}"]
    lines += ["", "## By truth role", "", markdown_table(by_role)]
    if errors is not None and len(errors):
        cols = [c for c in ["address_label", "role", "scenario", "shell_probability", "reasons"] if c in errors]
        lines += ["", f"## Misclassified at 0.5 ({len(errors)})", ""]
        for r in errors[cols].itertuples(index=False):
            d = r._asdict()
            lines.append(f"* `{d.get('address_label')}`, {d.get('role')}, p={d['shell_probability']:.3f}: {d.get('reasons')}")
    return "\n".join(lines) + "\n"

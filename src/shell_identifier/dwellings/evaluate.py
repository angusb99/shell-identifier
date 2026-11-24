"""Evaluate dwelling assignments against ground truth (synthetic data or a hand-checked sample)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import markdown_table
from .roles import PRIMARY, SUB


def evaluate_dwellings(assign: pd.DataFrame, truth: pd.DataFrame, truth_id: str = "address_detail_pid") -> tuple[dict, str]:
    """Compare input labels and model assignments with true dwelling / shell status.

    ``truth`` needs ``is_dwelling`` (0/1) and optionally ``is_shell``, ``scenario``, ``site``.
    """
    t = truth.rename(columns={truth_id: "address_pid", "role": "truth_role"}).copy()
    t["is_dwelling"] = t["is_dwelling"].astype(int).astype(bool)
    m = assign.merge(t, on="address_pid", how="inner")
    m["assigned_dwelling"] = m["assigned_dwelling"].astype(str).str.lower().isin(["true", "1"])
    v = m[m["mode"] == "verified"].copy()
    label_pd = v["input_label"].astype(str).str.upper().isin(["PD", "1", "Y", "YES", "TRUE"])
    wrong = label_pd != v["is_dwelling"]
    fixed = wrong & (v["assigned_dwelling"] == v["is_dwelling"])
    broke = ~wrong & (v["assigned_dwelling"] != v["is_dwelling"])
    parcel_ok_labels = (label_pd == v["is_dwelling"]).groupby(v["parcel_pid"]).all()
    parcel_ok_model = (v["assigned_dwelling"] == v["is_dwelling"]).groupby(v["parcel_pid"]).all()

    metrics = {
        "addresses": int(len(v)),
        "true_dwellings": int(v["is_dwelling"].sum()),
        "label_accuracy": float((label_pd == v["is_dwelling"]).mean()),
        "model_accuracy": float((v["assigned_dwelling"] == v["is_dwelling"]).mean()),
        "wrong_labels": int(wrong.sum()),
        "wrong_labels_fixed": int(fixed.sum()),
        "correct_labels_broken": int(broke.sum()),
        "parcels_all_correct_labels": float(parcel_ok_labels.mean()),
        "parcels_all_correct_model": float(parcel_ok_model.mean()),
    }
    lines = ["# Dwelling model evaluation", ""]
    if len(v):
        lines += [
            f"* Verified addresses: {metrics['addresses']:,} ({metrics['true_dwellings']:,} true dwellings)",
            f"* Address accuracy: input labels **{metrics['label_accuracy']:.4f}** -> model "
            f"**{metrics['model_accuracy']:.4f}**",
            f"* Wrongly placed labels: {metrics['wrong_labels']:,}; fixed by the model: "
            f"**{metrics['wrong_labels_fixed']:,}** ({metrics['wrong_labels_fixed'] / max(1, metrics['wrong_labels']):.1%})"
            f"; correct labels the model broke: **{metrics['correct_labels_broken']:,}**",
            f"* Parcels with every address right: labels {metrics['parcels_all_correct_labels']:.4f} -> "
            f"model {metrics['parcels_all_correct_model']:.4f}",
        ]

    if "is_shell" in m:
        # Shells only count where the site actually has dwellings (a shell is a container of dwellings).
        m["is_shell"] = m["is_shell"].astype(int).astype(bool)
        group = m["site"] if "site" in m else m["parcel_pid"]
        has_dw = m["is_dwelling"].groupby(group).transform("any")
        true_shell = m["is_shell"] & has_dw
        pred_shell = m["role"].isin([PRIMARY, SUB])
        tp = int((true_shell & pred_shell).sum())
        precision = tp / max(1, int(pred_shell.sum()))
        recall = tp / max(1, int(true_shell.sum()))
        metrics.update(shell_precision=precision, shell_recall=recall, shells_true=int(true_shell.sum()),
                       shells_predicted=int(pred_shell.sum()))
        lines.append(f"* Shells (containers of dwellings): precision **{precision:.4f}**, recall **{recall:.4f}** "
                     f"({tp:,} of {int(true_shell.sum()):,} found, {int(pred_shell.sum()):,} predicted)")

    for mode, what in (("predicted", "no labels, no counts"), ("counted", "no labels, known parcel counts")):
        p = m[m["mode"] == mode]
        if len(p):
            ok = p["assigned_dwelling"] == p["is_dwelling"]
            true_k = p.groupby("parcel_pid")["is_dwelling"].sum()
            got_k = p.groupby("parcel_pid")["assigned_dwelling"].sum()
            metrics[f"{mode}_accuracy"] = float(ok.mean())
            metrics[f"{mode}_parcel_count_accuracy"] = float((true_k == got_k).mean())
            lines.append(f"* {mode.title()} parcels ({what}): address accuracy **{ok.mean():.4f}**, "
                         f"parcels with every address right {ok.groupby(p['parcel_pid']).all().mean():.4f}, "
                         f"dwelling count exactly right {(true_k == got_k).mean():.4f}")

    if "scenario" in v and len(v):
        by = v.assign(_label_ok=(label_pd == v["is_dwelling"]), _model_ok=(v["assigned_dwelling"] == v["is_dwelling"]),
                      _wrong=wrong, _fixed=fixed, _broke=broke).groupby("scenario")
        table = pd.DataFrame({
            "addresses": by.size(),
            "label_acc": by["_label_ok"].mean().round(4),
            "model_acc": by["_model_ok"].mean().round(4),
            "wrong_labels": by["_wrong"].sum(),
            "fixed": by["_fixed"].sum(),
            "broken": by["_broke"].sum(),
        }).sort_values("model_acc")
        lines += ["", "## By scenario", "", markdown_table(table)]
    return metrics, "\n".join(lines) + "\n"

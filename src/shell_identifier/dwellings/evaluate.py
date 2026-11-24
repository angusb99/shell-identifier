"""Evaluate dwelling assignments against ground truth (synthetic data or a hand-checked sample)."""

from __future__ import annotations

import polars as pl

from ..frames import c, markdown_table
from .roles import PRIMARY, SUB

TRUE_VALUES = ["TRUE", "1", "Y", "YES", "T", "1.0"]
PD_LABELS = ["PD", "1", "Y", "YES", "TRUE"]


def _as_bool(col: str) -> pl.Expr:
    return c(col).cast(pl.Utf8).fill_null("").str.to_uppercase().is_in(TRUE_VALUES)


def evaluate_dwellings(assign: pl.DataFrame, truth: pl.DataFrame, truth_id: str = "address_detail_pid"
                       ) -> tuple[dict, str]:
    """Compare input labels and model assignments with true dwelling / shell status.

    ``truth`` needs ``is_dwelling`` (0/1) and optionally ``is_shell``, ``scenario``, ``site``.
    """
    t = truth.rename({truth_id: "address_pid", **({"role": "truth_role"} if "role" in truth.columns else {})})
    t = t.with_columns(c("address_pid").cast(pl.Utf8), is_dwelling=_as_bool("is_dwelling"))
    m = assign.with_columns(c("address_pid").cast(pl.Utf8), c("parcel_pid").cast(pl.Utf8),
                            assigned_dwelling=_as_bool("assigned_dwelling")).join(t, on="address_pid", how="inner")
    v = m.filter(c("mode") == "verified").with_columns(
        _label_pd=c("input_label").cast(pl.Utf8).fill_null("").str.to_uppercase().is_in(PD_LABELS))
    v = v.with_columns(_wrong=c("_label_pd") != c("is_dwelling"),
                       _model_ok=c("assigned_dwelling") == c("is_dwelling"))
    v = v.with_columns(_fixed=c("_wrong") & c("_model_ok"), _broke=~c("_wrong") & ~c("_model_ok"))

    per_parcel = v.group_by("parcel_pid").agg((~c("_wrong")).all().alias("labels_ok"), c("_model_ok").all().alias("model_ok"))
    n = max(1, v.height)
    metrics = {
        "addresses": v.height,
        "true_dwellings": int(v["is_dwelling"].sum()),
        "label_accuracy": float((~v["_wrong"]).sum() / n),
        "model_accuracy": float(v["_model_ok"].sum() / n),
        "wrong_labels": int(v["_wrong"].sum()),
        "wrong_labels_fixed": int(v["_fixed"].sum()),
        "correct_labels_broken": int(v["_broke"].sum()),
        "parcels_all_correct_labels": float(per_parcel["labels_ok"].mean() or 0.0),
        "parcels_all_correct_model": float(per_parcel["model_ok"].mean() or 0.0),
    }
    lines = ["# Dwelling model evaluation", ""]
    if v.height:
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

    if "is_shell" in m.columns:
        # Shells only count where the site actually has dwellings (a shell is a container of dwellings).
        group = "site" if "site" in m.columns else "parcel_pid"
        m = m.with_columns(_true_shell=_as_bool("is_shell") & c("is_dwelling").any().over(group),
                           _pred_shell=c("role").is_in([PRIMARY, SUB]))
        tp = int((m["_true_shell"] & m["_pred_shell"]).sum())
        n_true, n_pred = int(m["_true_shell"].sum()), int(m["_pred_shell"].sum())
        precision, recall = tp / max(1, n_pred), tp / max(1, n_true)
        metrics.update(shell_precision=precision, shell_recall=recall, shells_true=n_true, shells_predicted=n_pred)
        lines.append(f"* Shells (containers of dwellings): precision **{precision:.4f}**, recall **{recall:.4f}** "
                     f"({tp:,} of {n_true:,} found, {n_pred:,} predicted)")

    for mode, what in (("predicted", "no labels, no counts"), ("counted", "no labels, known parcel counts")):
        p = m.filter(c("mode") == mode)
        if p.height:
            p = p.with_columns(_ok=c("assigned_dwelling") == c("is_dwelling"))
            g = p.group_by("parcel_pid").agg(c("_ok").all().alias("all_ok"),
                                              (c("is_dwelling").sum() == c("assigned_dwelling").sum()).alias("k_ok"))
            acc = float(p["_ok"].mean())
            metrics[f"{mode}_accuracy"] = acc
            metrics[f"{mode}_parcel_count_accuracy"] = float(g["k_ok"].mean())
            lines.append(f"* {mode.title()} parcels ({what}): address accuracy **{acc:.4f}**, "
                         f"parcels with every address right {g['all_ok'].mean():.4f}, "
                         f"dwelling count exactly right {g['k_ok'].mean():.4f}")

    if "scenario" in v.columns and v.height:
        table = v.group_by("scenario").agg(
            pl.len().alias("addresses"),
            (~c("_wrong")).mean().round(4).alias("label_acc"),
            c("_model_ok").mean().round(4).alias("model_acc"),
            c("_wrong").sum().alias("wrong_labels"),
            c("_fixed").sum().alias("fixed"),
            c("_broke").sum().alias("broken"),
        ).sort(["model_acc", "scenario"])
        lines += ["", "## By scenario", "", markdown_table(table)]
    return metrics, "\n".join(lines) + "\n"

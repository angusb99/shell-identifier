"""Transparent log-odds scoring of shell likelihood with per-feature explanations.

``logit = intercept + sum(weight_k * feature_k)`` and ``p = sigmoid(logit)``.
Weights start as expert defaults (``default_config.yaml``) and can be refitted
from labelled examples with :mod:`shell_identifier.train`. Because the model is
linear in readable features, every score decomposes into named contributions,
which become the ``reasons`` text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def feature_names(cfg: dict) -> list[str]:
    return list(cfg["scoring"]["weights"])


def _s(x) -> pd.Series:
    return x.round(0).astype("Int64").astype(str)


def _breakdown(df: pd.DataFrame) -> pd.Series:
    parts = []
    for cat in ("dwelling", "commercial", "level", "structure", "secondary", "ancillary"):
        n = df[f"n_desc_{cat}"].fillna(0).astype(int)
        parts.append(np.where(n > 0, n.astype(str) + " " + cat, ""))
    joined = pd.Series([", ".join(p for p in row if p) for row in zip(*parts)], index=df.index)
    return joined


def reason_texts(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Human-readable text for each feature, evaluated row-wise (used only where the feature fires)."""
    pct = (100 * df["f_stacked_children"]).round(0).astype(int).astype(str)
    return {
        "log_children": _s(df["n_desc"]) + " nested sub-address(es) (" + _breakdown(df) + ")",
        "single_child": pd.Series("only one nested sub-address", index=df.index),
        "only_secondary_children": pd.Series("nested sub-addresses are secondary dwellings only (REAR/COTTAGE)", index=df.index),
        "only_ancillary_children": pd.Series("nested sub-addresses are ancillary only (car space/storage)", index=df.index),
        "log_range_children": "number range covers " + _s(df["n_range_numbers"]) + " other number(s), "
                              + _s(df["n_range_desc"]) + " address(es)",
        "log_offparcel_children": _s(df["n_offparcel_desc"]) + " sub-address(es) of this street address on nearby parcels",
        "stacked_children": pct + "% of nested sub-addresses share its geocode",
        "suffix_base": "unsuffixed number with suffixed siblings (" + df["suffix_list"].astype(str) + ")",
        "suffix_multi_variant": pd.Series("two or more suffix variants: base number usually split", index=df.index),
        "suffix_single_variant": pd.Series("single suffix variant: base and suffix often both occupied", index=df.index),
        "strata_matches_siblings": "strata count " + _s(df["total_strata_count"]) + " = "
                                   + _s(df["n_other_dwellings"]) + " other dwelling(s)",
        "strata_exceeds_siblings": "strata count " + _s(df["total_strata_count"]) + " > "
                                   + _s(df["n_other_dwellings"]) + " other dwelling(s)",
        "building_surplus": _s(df["n_dwelling_buildings"]) + " building(s) for " + _s(df["n_other_dwellings"])
                            + " other dwelling(s)",
        "building_deficit": "only " + _s(df["n_dwelling_buildings"]) + " building(s) for "
                            + _s(df["n_other_dwellings"]) + " other dwelling(s)",
        "own_building": pd.Series("geocode in a building no other address uses", index=df.index),
        "outside_buildings": pd.Series("geocode outside every building while siblings are inside one", index=df.index),
        "multistorey": "parcel has a ~" + _s(df["max_levels"]) + "-storey building",
        "vertical_strata": "vertical strata scheme (" + _s(df["vertical_strata_count"]) + " lots)",
        "parcel_level_geocode": "parcel-level geocode (" + df["geocode_type"].astype(str)
                                + ") while siblings have building/unit geocodes",
        "gnaf_primary_linked": "G-NAF primary with " + _s(df["n_ps_secondaries"]) + " linked secondaries",
        "gnaf_primary_flag_only": pd.Series("G-NAF primary flag, no linked secondaries", index=df.index),
        "gnaf_secondary": pd.Series("G-NAF secondary address", index=df.index),
        "legal_common_property": "legal parcel is common property (" + df["legal_parcel_id"].astype(str) + ")",
        "legal_strata_lot": "legal parcel is a strata lot (" + df["legal_parcel_id"].astype(str) + ")",
        "vanity_name": pd.Series("carries a building/complex name", index=df.index),
        "leaf_subaddress": pd.Series("sub-address with nothing nested under it", index=df.index),
        "leaf_street_address": pd.Series("street address with no nested, ranged or suffixed addresses", index=df.index),
        "single_address_parcel": pd.Series("only address on the parcel", index=df.index),
        "has_parent": "nested under " + df.get("parent_label", df["parent_pid"]).astype(str),
        "alias": pd.Series("G-NAF alias address", index=df.index),
    }


def classify(p: pd.Series, bands: list) -> pd.Series:
    bands = sorted(((float(lo), str(name)) for lo, name in bands), reverse=True)
    out = pd.Series(bands[-1][1], index=p.index, dtype=object)
    for lo, name in reversed(bands):
        out[p >= lo] = name
    return out


def score(feats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ``shell_logit``, ``shell_probability``, ``shell_class`` and ``reasons``."""
    sc = cfg["scoring"]
    names = feature_names(cfg)
    X = feats[[f"f_{n}" for n in names]].to_numpy(dtype=float)
    w = np.array([float(sc["weights"][n]) for n in names])
    contrib = X * w
    out = feats.copy()
    out["shell_logit"] = float(sc["intercept"]) + contrib.sum(axis=1)
    out["shell_probability"] = 1.0 / (1.0 + np.exp(-out["shell_logit"]))
    out["shell_class"] = classify(out["shell_probability"], sc["classes"])

    # Reasons: top-k non-zero contributions per row, largest magnitude first.
    rows, cols = np.nonzero(contrib)
    if len(rows):
        texts = reason_texts(out)
        values = contrib[rows, cols]
        text = np.empty(len(rows), dtype=object)
        for j, name in enumerate(names):
            sel = cols == j
            if sel.any():
                text[sel] = texts[name].to_numpy()[rows[sel]]
        long = pd.DataFrame({"row": rows, "abs": np.abs(values), "txt": [f"{t} [{v:+.1f}]" for t, v in zip(text, values)]})
        long = long.sort_values(["row", "abs"], ascending=[True, False])
        long = long.groupby("row").head(int(sc.get("max_reasons", 4)))
        reasons = long.groupby("row")["txt"].agg("; ".join)
        out["reasons"] = pd.Series(reasons.reindex(np.arange(len(out))).fillna("").to_numpy(), index=out.index)
    else:
        out["reasons"] = ""
    return out

"""Transparent log-odds scoring of shell likelihood with per-feature explanations.

``logit = intercept + sum(weight_k * feature_k)`` and ``p = sigmoid(logit)``.
Weights start as expert defaults (``default_config.yaml``) and can be refitted
from labelled examples with :mod:`shell_identifier.train`. Because the model is
linear in readable features, every score decomposes into named contributions,
which become the ``reasons`` text.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .frames import c, int_text, join_nonempty

REASON_CATS = ("dwelling", "commercial", "level", "structure", "secondary", "ancillary")


def feature_names(cfg: dict) -> list[str]:
    return list(cfg["scoring"]["weights"])


def _breakdown() -> pl.Expr:
    return join_nonempty([pl.when(c(f"n_desc_{k}") > 0).then(int_text(c(f"n_desc_{k}")) + f" {k}")
                          .otherwise(pl.lit("")) for k in REASON_CATS], ", ")


def reason_texts() -> dict[str, pl.Expr]:
    """Human-readable text for each feature (evaluated only on rows where the feature contributes)."""
    lit = pl.lit
    return {
        "log_children": int_text(c("n_desc")) + " nested sub-address(es) (" + _breakdown() + ")",
        "single_child": lit("only one nested sub-address"),
        "only_secondary_children": lit("nested sub-addresses are secondary dwellings only (REAR/COTTAGE)"),
        "only_ancillary_children": lit("nested sub-addresses are ancillary only (car space/storage)"),
        "log_range_children": "number range covers " + int_text(c("n_range_numbers")) + " other number(s), "
                              + int_text(c("n_range_desc")) + " address(es)",
        "log_offparcel_children": int_text(c("n_offparcel_desc")) + " sub-address(es) of this street address on nearby parcels",
        "stacked_children": int_text(100 * c("f_stacked_children")) + "% of nested sub-addresses share its geocode",
        "suffix_base": "unsuffixed number with suffixed siblings (" + c("suffix_list") + ")",
        "suffix_multi_variant": lit("two or more suffix variants: base number usually split"),
        "suffix_single_variant": lit("single suffix variant: base and suffix often both occupied"),
        "strata_matches_siblings": "strata count " + int_text(c("total_strata_count")) + " = "
                                   + int_text(c("n_other_dwellings")) + " other dwelling(s)",
        "strata_exceeds_siblings": "strata count " + int_text(c("total_strata_count")) + " > "
                                   + int_text(c("n_other_dwellings")) + " other dwelling(s)",
        "building_surplus": int_text(c("n_dwelling_buildings")) + " building(s) for "
                            + int_text(c("n_other_dwellings")) + " other dwelling(s)",
        "building_deficit": "only " + int_text(c("n_dwelling_buildings")) + " building(s) for "
                            + int_text(c("n_other_dwellings")) + " other dwelling(s)",
        "own_building": lit("geocode in a building no other address uses"),
        "outside_buildings": lit("geocode outside every building while siblings are inside one"),
        "multistorey": "parcel has a ~" + int_text(c("max_levels")) + "-storey building",
        "vertical_strata": "vertical strata scheme (" + int_text(c("vertical_strata_count")) + " lots)",
        "parcel_level_geocode": "parcel-level geocode (" + c("geocode_type").cast(pl.Utf8)
                                + ") while siblings have building/unit geocodes",
        "gnaf_primary_linked": "G-NAF primary with " + int_text(c("n_ps_secondaries").cast(pl.Float64, strict=False))
                               + " linked secondaries",
        "gnaf_primary_flag_only": lit("G-NAF primary flag, no linked secondaries"),
        "gnaf_secondary": lit("G-NAF secondary address"),
        "legal_common_property": "legal parcel is common property (" + c("legal_parcel_id").cast(pl.Utf8) + ")",
        "legal_strata_lot": "legal parcel is a strata lot (" + c("legal_parcel_id").cast(pl.Utf8) + ")",
        "vanity_name": lit("carries a building/complex name"),
        "leaf_subaddress": lit("sub-address with nothing nested under it"),
        "leaf_street_address": lit("street address with no nested, ranged or suffixed addresses"),
        "single_address_parcel": lit("only address on the parcel"),
        "has_parent": "nested under " + c("parent_label"),
        "alias": lit("G-NAF alias address"),
    }


def classify(p: pl.Expr, bands: list) -> pl.Expr:
    bands = sorted(((float(lo), str(name)) for lo, name in bands), reverse=True)
    expr = pl.when(p >= bands[0][0]).then(pl.lit(bands[0][1]))
    for lo, name in bands[1:-1]:
        expr = expr.when(p >= lo).then(pl.lit(name))
    return expr.otherwise(pl.lit(bands[-1][1]))


def _reasons(out: pl.DataFrame, names: list[str], contrib: np.ndarray, k: int) -> pl.Series:
    rows, cols = np.nonzero(contrib)
    if len(rows) == 0:
        return pl.Series("reasons", [""] * out.height, dtype=pl.Utf8)
    vals = contrib[rows, cols]
    top = (pl.DataFrame({"row": rows, "col": cols, "val": vals, "abs": np.abs(vals)})
           .sort(["row", "abs"], descending=[False, True]).group_by("row", maintain_order=True).head(k))
    texts = reason_texts()
    frames = []
    for j, name in enumerate(names):
        sel = top.filter(c("col") == j)
        if sel.is_empty():
            continue
        expr = texts[name].fill_null("").alias("t")
        roots = list(dict.fromkeys(expr.meta.root_names()))
        sub = out.select(roots)[sel["row"].to_numpy()] if roots else sel.select("row")
        txt = sub.with_columns(expr)["t"]
        sign = pl.when(c("val") >= 0).then(pl.lit("+")).otherwise(pl.lit("-"))
        frames.append(sel.with_columns(t=txt).select(
            "row", "abs", txt=c("t") + " [" + sign + c("val").abs().round(1).cast(pl.Utf8) + "]"))
    joined = (pl.concat(frames).sort(["row", "abs"], descending=[False, True])
              .group_by("row", maintain_order=True).agg(c("txt").str.join("; ")))
    full = pl.DataFrame({"row": np.arange(out.height)}).join(joined, on="row", how="left", maintain_order="left")
    return full["txt"].fill_null("").alias("reasons")


def score(feats: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Add ``shell_logit``, ``shell_probability``, ``shell_class`` and ``reasons``."""
    sc = cfg["scoring"]
    names = feature_names(cfg)
    X = feats.select([c(f"f_{n}").cast(pl.Float64) for n in names]).to_numpy()
    w = np.array([float(sc["weights"][n]) for n in names])
    contrib = X * w
    logit = float(sc["intercept"]) + contrib.sum(axis=1)
    out = feats.with_columns(shell_logit=pl.Series(logit), shell_probability=pl.Series(1.0 / (1.0 + np.exp(-logit))))
    out = out.with_columns(shell_class=classify(c("shell_probability"), sc["classes"]))
    if "parent_label" not in out.columns:
        out = out.with_columns(parent_label=c("parent_pid"))
    return out.with_columns(_reasons(out, names, contrib, int(sc.get("max_reasons", 4))))

"""Context features describing each address relative to the other addresses on its parcel.

The *context set* of a parcel is its principal (non-alias) addresses. Every
address - alias or not - is described against that set:

* nested children    - same street + number, strictly more specific components
* range children     - a number range (1-15) that covers other numbers on the parcel
* off-parcel children- sub-addresses of the same street address on nearby parcels
* suffix siblings    - 100A / 100B next to an unsuffixed 100
* parents            - the most specific address it nests under
* building / strata balance, storeys, G-NAF primary/secondary links, legal parcel id

All work is groupby/merge based so it scales to state-sized extracts.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .authority import GEOCODE_TYPES
from .components import COMP_COLS, MASK_BITS, finalise_components

GROUP = ["parcel_pid", "street_key", "num_key"]
NODE_CATS = ["dwelling", "commercial", "ancillary", "secondary", "level", "structure"]


def _popcount(m: int) -> int:
    return bin(m).count("1")


def _comp_keys(mask: int) -> list[str]:
    return [c for c, bit in MASK_BITS.items() if mask & bit]


def _lookup(left: pd.DataFrame, table: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Left-join ``table`` (unique on ``keys``) onto ``left`` keeping ``left``'s index."""
    merged = left[keys].merge(table[keys + cols], on=keys, how="left")
    merged.index = left.index
    return merged[cols]


def _promote_vanity_names(a: pd.DataFrame) -> None:
    """Treat vanity names as building ids where one street number holds several named buildings.

    "THE ASH" and "THE BIRCH" under the same 1-15 SMITH ST are separate
    buildings; a single name across the site ("THE ARCADIA") is just a label.
    """
    named = a[a["in_ctx"] & (a["vanity_name"] != "") & ((a["comp_flat"] != "") | (a["comp_level"] != ""))]
    if named.empty:
        return
    n_names = named.groupby(GROUP)["vanity_name"].nunique()
    multi = n_names[n_names >= 2].reset_index()[GROUP]
    if multi.empty:
        return
    multi["_multi"] = True
    flag = _lookup(a, multi, GROUP, ["_multi"])["_multi"].fillna(False).astype(bool)
    promote = flag & (a["comp_structure"] == "") & (a["vanity_name"] != "")
    a.loc[promote, "comp_structure"] = "NAME " + a.loc[promote, "vanity_name"]


def _descendants(a: pd.DataFrame, ctx: pd.DataFrame) -> None:
    cols = ["n_desc", "eff_desc", "n_desc_buildings"] + [f"n_desc_{c}" for c in NODE_CATS]
    for c in cols + ["n_desc_stacked"]:
        a[c] = 0.0
    for m in range(8):
        if m & MASK_BITS["comp_flat"]:
            continue  # a flat is never a container
        targets = a.index[a["mask"] == m]
        sub = ctx[((ctx["mask"] & m) == m) & (ctx["mask"] != m)]
        if len(targets) == 0 or sub.empty:
            continue
        keys = GROUP + _comp_keys(m)
        spec = {"n_desc": ("address_pid", "size"), "eff_desc": ("child_w", "sum"),
                "n_desc_buildings": ("building_pid", "nunique")}
        spec.update({f"n_desc_{c}": (f"_is_{c}", "sum") for c in NODE_CATS})
        agg = sub.groupby(keys).agg(**spec).reset_index()
        vals = _lookup(a.loc[targets], agg, keys, cols).fillna(0)
        a.loc[targets, cols] = vals.to_numpy(dtype=float)
        stacked = sub.groupby(keys + ["xy_key"]).size().rename("n_desc_stacked").reset_index()
        st = _lookup(a.loc[targets], stacked, keys + ["xy_key"], ["n_desc_stacked"]).fillna(0)
        a.loc[targets, "n_desc_stacked"] = st["n_desc_stacked"].to_numpy(dtype=float)


def _ancestors(a: pd.DataFrame, ctx: pd.DataFrame) -> None:
    a["parent_pid"] = ""
    a["n_ancestors"] = 0
    for s in sorted((m for m in range(8) if not m & MASK_BITS["comp_flat"]), key=_popcount):
        anc = ctx[ctx["mask"] == s]
        targets = a.index[((a["mask"] & s) == s) & (a["mask"] != s)]
        if anc.empty or len(targets) == 0:
            continue
        keys = GROUP + _comp_keys(s)
        table = anc.groupby(keys).agg(_anc_pid=("address_pid", "first")).reset_index()
        found = _lookup(a.loc[targets], table, keys, ["_anc_pid"])["_anc_pid"].dropna()
        a.loc[found.index, "parent_pid"] = found  # later (more specific) masks overwrite
        a.loc[found.index, "n_ancestors"] += 1


def _ranges(a: pd.DataFrame) -> None:
    """Range containers (1-15 SMITH ST) and the range parent of each covered address."""
    a["n_range_desc"] = 0.0
    a["n_range_numbers"] = 0.0
    a["n_range_leaves"] = 0.0
    a["eff_range"] = 0.0
    a["range_parent_pid"] = ""
    rng = a[a["num_last"].notna() & (a["comp_flat"] == "") & (a["parcel_pid"] != "")]
    if rng.empty:
        return
    left_cols = ["_row", "address_pid", "parcel_pid", "street_key", "num_key", "_nf", "_nl", "in_ctx"] + COMP_COLS
    right_cols = ["_row", "address_pid", "parcel_pid", "street_key", "num_key", "_nf", "_nl", "in_ctx", "child_w",
                  "_leaf"] + COMP_COLS
    m = rng[left_cols].merge(a[right_cols], on=["parcel_pid", "street_key"], suffixes=("", "_c"))
    nl_c = m["_nl_c"].fillna(m["_nf_c"])
    inside = (m["num_key_c"] != m["num_key"]) & m["_nf_c"].notna() & (m["_nf_c"] >= m["_nf"]) & (nl_c <= m["_nl"])
    for comp in COMP_COLS:
        inside &= (m[comp] == "") | (m[comp] == m[comp + "_c"])
    m = m[inside]
    if m.empty:
        return
    desc = m[m["in_ctx_c"]]
    agg = desc.groupby("_row").agg(n_range_desc=("_row_c", "size"), n_range_numbers=("num_key_c", "nunique"),
                                   n_range_leaves=("_leaf", "sum"), eff_range=("child_w", "sum"))
    a.loc[agg.index, ["n_range_desc", "n_range_numbers", "n_range_leaves", "eff_range"]] = agg.to_numpy(dtype=float)
    par = m[m["in_ctx"]].assign(_width=lambda d: d["_nl"] - d["_nf"]).sort_values("_width")
    par = par.drop_duplicates("_row_c").set_index("_row_c")["address_pid"]
    a.loc[par.index, "range_parent_pid"] = par


def _offparcel(a: pd.DataFrame, radius: float) -> None:
    """Sub-addresses of the same street address that were geocoded onto other parcels nearby."""
    a["n_offparcel_desc"] = 0.0
    a["eff_offparcel"] = 0.0
    bare = a[(a["mask"] == 0) & (a["num_key"] != "") & (a["street_key"].str.len() > 2)]
    subs = a[~a["is_alias"] & (a["mask"] != 0) & (a["num_key"] != "")]
    if bare.empty or subs.empty:
        return
    m = bare[["_row", "street_key", "num_key", "parcel_pid", "_x", "_y"]].merge(
        subs[["street_key", "num_key", "parcel_pid", "_x", "_y", "child_w"]],
        on=["street_key", "num_key"], suffixes=("", "_c"))
    m = m[m["parcel_pid_c"] != m["parcel_pid"]]
    m = m[np.hypot(m["_x"] - m["_x_c"], m["_y"] - m["_y_c"]) <= radius]
    if m.empty:
        return
    agg = m.groupby("_row").agg(n_offparcel_desc=("child_w", "size"), eff_offparcel=("child_w", "sum"))
    a.loc[agg.index, ["n_offparcel_desc", "eff_offparcel"]] = agg.to_numpy()


def _suffix_siblings(a: pd.DataFrame, ctx: pd.DataFrame) -> None:
    a["n_suffix_variants"] = 0
    a["suffix_list"] = ""
    base = a.index[(a["mask"] == 0) & (a["num_suffix"] == "") & a["num_last"].isna() & (a["num_key"] != "")]
    var = ctx[ctx["num_suffix"] != ""]
    if len(base) == 0 or var.empty:
        return
    keys = ["parcel_pid", "street_key", "base_num_key"]
    agg = var.groupby(keys).agg(n_suffix_variants=("num_key", "nunique"),
                                suffix_list=("num_suffix", lambda s: ",".join(sorted(set(s))))).reset_index()
    vals = _lookup(a.loc[base], agg, keys, ["n_suffix_variants", "suffix_list"])
    a.loc[base, "n_suffix_variants"] = vals["n_suffix_variants"].fillna(0).astype(int).to_numpy()
    a.loc[base, "suffix_list"] = vals["suffix_list"].fillna("").to_numpy()


def _parcel_counts(a: pd.DataFrame, ctx: pd.DataFrame) -> None:
    """Counts over the parcel's context set, excluding the address itself."""
    self_ctx = a["in_ctx"].astype(int)
    a["n_parcel_others"] = a["parcel_pid"].map(ctx.groupby("parcel_pid").size()).fillna(0).astype(int) - self_ctx
    a.loc[a["parcel_pid"] == "", "n_parcel_others"] = 0

    # "Other dwellings" are the occupiable leaves that compete with this address: those sharing its
    # street and base number (100, 100A, UNIT 1 100) plus anything its number range covers.
    is_leaf = a["_leaf"] & (a["n_range_desc"] == 0)
    grp = ["parcel_pid", "street_key", "base_num_key"]
    leaves = a[a["in_ctx"] & is_leaf].groupby(grp).size().rename("_n").reset_index()
    same = _lookup(a, leaves, grp, ["_n"])["_n"].fillna(0)
    a["n_other_dwellings"] = (same - (a["in_ctx"] & is_leaf).astype(int) + a["n_range_leaves"]).astype(int)
    a["n_other_streets"] = a["parcel_pid"].map(ctx.groupby("parcel_pid")["street_key"].nunique()).fillna(1).astype(int) - 1

    # Buildings can straddle parcels, so building occupancy counts every principal address.
    principal = a[~a["is_alias"] & a["building_pid"].notna()]
    per_b = principal.groupby("building_pid").size()
    has_b = a["building_pid"].notna()
    a["n_same_building"] = 0.0
    a.loc[has_b, "n_same_building"] = (a.loc[has_b, "building_pid"].map(per_b).fillna(0)
                                       - (~a.loc[has_b, "is_alias"]).astype(int)).to_numpy(dtype=float)

    spatial = ctx[ctx["in_building_spatial"]].groupby("parcel_pid").size()
    a["n_others_in_buildings"] = (a["parcel_pid"].map(spatial).fillna(0).astype(int)
                                  - (a["in_ctx"] & a["in_building_spatial"]).astype(int))
    bu = ctx[ctx["geocode_cat"].isin(["building", "unit"])].groupby("parcel_pid").size()
    a["n_others_bu_geocode"] = (a["parcel_pid"].map(bu).fillna(0).astype(int)
                                - (a["in_ctx"] & a["geocode_cat"].isin(["building", "unit"])).astype(int))


def _regex_flag(series: pd.Series, patterns: list[str]) -> pd.Series:
    if not patterns:
        return pd.Series(False, index=series.index)
    rx = re.compile("|".join(f"(?:{p})" for p in patterns))
    return series.fillna("").astype(str).str.upper().map(lambda s: bool(rx.search(s)))


def compute_features(addr: pd.DataFrame, parcel_ctx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return ``addr`` with raw context columns and the model features (``f_*``).

    ``addr`` must hold the component columns from
    :func:`shell_identifier.components.build_components` plus ``parcel_pid``,
    ``building_pid``, ``in_building_spatial`` and metric ``_x`` / ``_y``.
    """
    fcfg, scfg = cfg["features"], cfg["spatial"]
    a = addr.copy().reset_index(drop=True)
    a["_row"] = a.index
    a["parcel_pid"] = a["parcel_pid"].fillna("").astype(str)
    a["is_alias"] = a["alias_principal"].fillna("").astype(str).str.upper().eq("A")
    a["in_ctx"] = ~a["is_alias"] & (a["parcel_pid"] != "")
    a["building_pid"] = a["building_pid"].where(a["building_pid"].notna() & (a["building_pid"].astype(str) != ""))
    a["in_building_spatial"] = a["in_building_spatial"].fillna(False).astype(bool)
    tol = float(scfg["stacked_tolerance_m"])
    a["xy_key"] = (np.round(a["_x"] / tol).astype("int64").astype(str) + "_"
                   + np.round(a["_y"] / tol).astype("int64").astype(str))
    a["_nf"] = a["num_first"].astype("float64")
    a["_nl"] = a["num_last"].astype("float64")
    a["geocode_cat"] = a["geocode_type"].fillna("").astype(str).str.upper().map(
        lambda c: GEOCODE_TYPES.get(c, ("", ""))[1])

    _promote_vanity_names(a)
    finalise_components(a)
    a["child_w"] = a["node_type"].map(fcfg["child_weights"]).fillna(1.0)
    for c in NODE_CATS:
        a[f"_is_{c}"] = (a["node_type"] == c).astype(int)
    ctx = a[a["in_ctx"]]

    _descendants(a, ctx)
    _ancestors(a, ctx)
    a["_leaf"] = (a["eff_desc"] == 0) & (a["node_type"] != "ancillary")
    _ranges(a)
    _offparcel(a, float(scfg["offparcel_radius_m"]))
    _suffix_siblings(a, ctx)
    _parcel_counts(a, ctx)

    pc = parcel_ctx.reindex(a["parcel_pid"])
    for col in ("total_strata_count", "horizontal_strata_count", "vertical_strata_count",
                "n_buildings", "n_dwelling_buildings", "max_levels", "parcel_area_m2"):
        a[col] = pc[col].to_numpy() if col in pc else np.nan
    for col in ("n_buildings", "n_dwelling_buildings"):
        a[col] = a[col].fillna(0).astype(int)

    a["parent_pid"] = np.where(a["parent_pid"] != "", a["parent_pid"], a["range_parent_pid"])
    ps = a["primary_secondary"].fillna("").astype(str).str.upper()
    n_ps = pd.to_numeric(a["n_ps_secondaries"], errors="coerce").fillna(0)
    legal = a["legal_parcel_id"] if "legal_parcel_id" in a else pd.Series("", index=a.index)

    # ---- model features ---------------------------------------------------
    eff = a["eff_desc"]
    a["suffix_base"] = (a["n_suffix_variants"] > 0) & (eff == 0)
    cand = (eff > 0) | (a["n_range_desc"] > 0) | a["suffix_base"] | (a["n_offparcel_desc"] > 0)
    a["is_candidate"] = cand
    n_other = a["n_other_dwellings"]
    small = cand & (n_other >= 1) & (n_other <= int(fcfg["small_group_max_dwellings"]))
    strata = a["total_strata_count"]
    strata_known = strata.notna() & (strata > 0)
    n_bld = a["n_dwelling_buildings"]
    real_children = a["n_desc"] - a["n_desc_secondary"] - a["n_desc_ancillary"]

    f = {
        "log_children": np.log1p(eff),
        "single_child": (eff > 0) & (eff <= 1.5) & (a["n_range_desc"] == 0),
        "only_secondary_children": (a["n_desc_secondary"] > 0) & (real_children == 0),
        "only_ancillary_children": (a["n_desc_ancillary"] > 0) & (a["n_desc"] == a["n_desc_ancillary"]),
        "log_range_children": np.log1p(a["eff_range"]),
        "log_offparcel_children": np.log1p(a["eff_offparcel"]),
        "stacked_children": np.where(a["n_desc"] > 0, a["n_desc_stacked"] / a["n_desc"].clip(lower=1), 0.0),
        "suffix_base": a["suffix_base"],
        "suffix_multi_variant": a["suffix_base"] & (a["n_suffix_variants"] >= 2),
        "suffix_single_variant": a["suffix_base"] & (a["n_suffix_variants"] == 1),
        "strata_matches_siblings": small & strata_known & (strata <= n_other),
        "strata_exceeds_siblings": small & strata_known & (strata >= n_other + 1),
        "building_surplus": small & (n_bld >= n_other + 1),
        "building_deficit": small & (n_bld >= 1) & (n_bld <= n_other),
        "own_building": cand & a["building_pid"].notna() & (a["n_same_building"] == 0),
        "outside_buildings": cand & ~a["in_building_spatial"] & (n_bld >= 1) & (a["n_others_in_buildings"] >= 1),
        "multistorey": cand & (a["max_levels"].fillna(0) >= 3),
        "vertical_strata": cand & (a["vertical_strata_count"].fillna(0) >= 2),
        "parcel_level_geocode": cand & (a["geocode_cat"] == "parcel") & (a["n_others_bu_geocode"] >= 1),
        "gnaf_primary_linked": n_ps >= 1,
        "gnaf_primary_flag_only": (ps == "P") & (n_ps == 0),
        "gnaf_secondary": ps == "S",
        "legal_common_property": _regex_flag(legal, fcfg["legal_common_property_patterns"]),
        "legal_strata_lot": _regex_flag(legal, fcfg["legal_strata_lot_patterns"]),
        "vanity_name": cand & ((a["vanity_name"] != "") | a["comp_structure"].str.startswith("NAME ")) & (a["comp_flat"] == ""),
        "leaf_subaddress": (a["mask"] != 0) & ~cand,
        "leaf_street_address": (a["mask"] == 0) & ~cand,
        "single_address_parcel": (a["n_parcel_others"] == 0) & (a["n_offparcel_desc"] == 0),
        "has_parent": a["parent_pid"] != "",
        "alias": a["is_alias"],
    }
    for name, values in f.items():
        a[f"f_{name}"] = np.asarray(values, dtype=float)
    return a.drop(columns=[c for c in a.columns if c.startswith("_is_")] + ["_row", "_leaf"])

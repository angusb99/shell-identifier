"""Context features describing each address relative to the other addresses on its parcel.

The *context set* of a parcel is its principal (non-alias) addresses. Every
address - alias or not - is described against that set:

* nested children    - same street + number, strictly more specific components
* range children     - a number range (1-15) that covers other numbers on the parcel
* off-parcel children- sub-addresses of the same street address on nearby parcels
* suffix siblings    - 100A / 100B next to an unsuffixed 100
* parents            - the most specific address it nests under
* building / strata balance, storeys, G-NAF primary/secondary links, legal parcel id

Everything is polars group-by / join work, so it scales to state-sized extracts.
"""

from __future__ import annotations

import polars as pl

from .authority import geocode_category, normalise_geocode_type
from .components import COMP_COLS, MASK_BITS, finalise_components
from .frames import c, ensure_columns, left_join, map_values

GROUP = ["parcel_pid", "street_key", "num_key"]
NODE_CATS = ["dwelling", "commercial", "ancillary", "secondary", "level", "structure"]
DESC_COLS = ["n_desc", "eff_desc", "n_desc_buildings", "n_desc_stacked"] + [f"n_desc_{k}" for k in NODE_CATS]
CONTAINER_MASKS = [m for m in range(8) if not m & MASK_BITS["comp_flat"]]   # a flat is never a container
F64 = pl.Float64


def _popcount(m: int) -> int:
    return bin(m).count("1")


def _comp_keys(mask: int) -> list[str]:
    return [col for col, bit in MASK_BITS.items() if mask & bit]


def _has(mask_bits: int) -> pl.Expr:
    """Address mask contains every bit of ``mask_bits`` and is strictly more specific."""
    return ((c("mask") & mask_bits) == mask_bits) & (c("mask") != mask_bits)


def _promote_vanity_names(a: pl.DataFrame) -> pl.DataFrame:
    """Treat vanity names as building ids where one street number holds several named buildings.

    "THE ASH" and "THE BIRCH" under the same 1-15 SMITH ST are separate
    buildings; a single name across the site ("THE ARCADIA") is just a label.
    """
    named = a.filter(c("in_ctx") & (c("vanity_name") != "") & ((c("comp_flat") != "") | (c("comp_level") != "")))
    if named.is_empty():
        return a
    multi = (named.group_by(GROUP).agg(c("vanity_name").n_unique().alias("_n"))
             .filter(c("_n") >= 2).select(GROUP).with_columns(_multi=pl.lit(True)))
    if multi.is_empty():
        return a
    a = left_join(a, multi, GROUP)
    promote = c("_multi").fill_null(False) & (c("comp_structure") == "") & (c("vanity_name") != "")
    return a.with_columns(
        comp_structure=pl.when(promote).then("NAME " + c("vanity_name")).otherwise(c("comp_structure"))
    ).drop("_multi")


def _descendants(a: pl.DataFrame, ctx: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for m in CONTAINER_MASKS:
        targets = a.filter(c("mask") == m)
        sub = ctx.filter(_has(m))
        if targets.is_empty() or sub.is_empty():
            continue
        keys = GROUP + _comp_keys(m)
        agg = sub.group_by(keys).agg(
            pl.len().alias("n_desc"),
            c("child_w").sum().alias("eff_desc"),
            c("building_pid").drop_nulls().n_unique().alias("n_desc_buildings"),
            *[c(f"_is_{k}").sum().alias(f"n_desc_{k}") for k in NODE_CATS],
        )
        stacked = sub.group_by(keys + ["xy_key"]).agg(pl.len().alias("n_desc_stacked"))
        t = (targets.select(["_row"] + keys + ["xy_key"])
             .join(agg, on=keys, how="left").join(stacked, on=keys + ["xy_key"], how="left"))
        parts.append(t.select(["_row"] + [c(x).cast(F64) for x in DESC_COLS]))
    if not parts:
        return a.with_columns([pl.lit(0.0).alias(x) for x in DESC_COLS])
    desc = pl.concat(parts)
    return left_join(a, desc, "_row").with_columns([c(x).fill_null(0.0) for x in DESC_COLS])


def _ancestors(a: pl.DataFrame, ctx: pl.DataFrame) -> pl.DataFrame:
    """Most specific context address each address nests under (``parent_pid``) and how many it nests under."""
    parts = []
    for order, s in enumerate(sorted(CONTAINER_MASKS, key=_popcount)):
        anc = ctx.filter(c("mask") == s)
        targets = a.filter(_has(s))
        if anc.is_empty() or targets.is_empty():
            continue
        keys = GROUP + _comp_keys(s)
        table = anc.group_by(keys).agg(c("address_pid").first().alias("_anc_pid"))
        parts.append(targets.select(["_row"] + keys).join(table, on=keys, how="inner")
                     .select("_row", "_anc_pid", pl.lit(order).alias("_ord")))
    if not parts:
        return a.with_columns(parent_pid=pl.lit(""), n_ancestors=pl.lit(0))
    found = (pl.concat(parts).sort("_ord").group_by("_row")
             .agg(c("_anc_pid").last().alias("parent_pid"), pl.len().cast(pl.Int64).alias("n_ancestors")))
    return left_join(a, found, "_row").with_columns(c("parent_pid").fill_null(""), c("n_ancestors").fill_null(0))


def _ranges(a: pl.DataFrame) -> pl.DataFrame:
    """Range containers (1-15 SMITH ST) and the range parent of each covered address."""
    cols = {"n_range_desc": 0.0, "n_range_numbers": 0.0, "n_range_leaves": 0.0, "eff_range": 0.0}
    rng = a.filter(c("num_last").is_not_null() & (c("comp_flat") == "") & (c("parcel_pid") != ""))
    if rng.is_empty():
        return a.with_columns([pl.lit(v).alias(k) for k, v in cols.items()] + [pl.lit("").alias("range_parent_pid")])
    shared = ["_row", "address_pid", "parcel_pid", "street_key", "num_key", "_nf", "_nl", "in_ctx"] + COMP_COLS
    m = rng.select(shared).join(a.select(shared + ["child_w", "_leaf"]), on=["parcel_pid", "street_key"],
                                how="inner", suffix="_c")
    nl_c = c("_nl_c").fill_null(c("_nf_c"))
    inside = ((c("num_key_c") != c("num_key")) & c("_nf_c").is_not_null() & (c("_nf_c") >= c("_nf"))
              & (nl_c <= c("_nl")))
    for comp in COMP_COLS:
        inside = inside & ((c(comp) == "") | (c(comp) == c(f"{comp}_c")))
    m = m.filter(inside)
    if m.is_empty():
        return a.with_columns([pl.lit(v).alias(k) for k, v in cols.items()] + [pl.lit("").alias("range_parent_pid")])
    desc = m.filter(c("in_ctx_c")).group_by("_row").agg(
        pl.len().cast(F64).alias("n_range_desc"),
        c("num_key_c").n_unique().cast(F64).alias("n_range_numbers"),
        c("_leaf").sum().cast(F64).alias("n_range_leaves"),
        c("child_w").sum().cast(F64).alias("eff_range"),
    )
    parent = (m.filter(c("in_ctx")).with_columns(_width=c("_nl") - c("_nf"))
              .sort("_width", maintain_order=True).unique("_row_c", keep="first", maintain_order=True)
              .select(c("_row_c").alias("_row"), c("address_pid").alias("range_parent_pid")))
    a = left_join(left_join(a, desc, "_row"), parent, "_row")
    return a.with_columns([c(k).fill_null(v) for k, v in cols.items()] + [c("range_parent_pid").fill_null("")])


def _offparcel(a: pl.DataFrame, radius: float) -> pl.DataFrame:
    """Sub-addresses of the same street address that were geocoded onto other parcels nearby."""
    bare = a.filter((c("mask") == 0) & (c("num_key") != "") & (c("street_key").str.len_chars() > 2))
    subs = a.filter(~c("is_alias") & (c("mask") != 0) & (c("num_key") != ""))
    agg = None
    if not bare.is_empty() and not subs.is_empty():
        m = (bare.select("_row", "street_key", "num_key", "parcel_pid", "_x", "_y")
             .join(subs.select("street_key", "num_key", "parcel_pid", "_x", "_y", "child_w"),
                   on=["street_key", "num_key"], how="inner", suffix="_c"))
        dist = ((c("_x") - c("_x_c")) ** 2 + (c("_y") - c("_y_c")) ** 2).sqrt()
        m = m.filter((c("parcel_pid_c") != c("parcel_pid")) & (dist <= radius))
        if not m.is_empty():
            agg = m.group_by("_row").agg(pl.len().cast(F64).alias("n_offparcel_desc"),
                                         c("child_w").sum().alias("eff_offparcel"))
    if agg is None:
        return a.with_columns(n_offparcel_desc=pl.lit(0.0), eff_offparcel=pl.lit(0.0))
    return left_join(a, agg, "_row").with_columns(c("n_offparcel_desc").fill_null(0.0),
                                                  c("eff_offparcel").fill_null(0.0))


def _suffix_siblings(a: pl.DataFrame, ctx: pl.DataFrame) -> pl.DataFrame:
    keys = ["parcel_pid", "street_key", "base_num_key"]
    base = (c("mask") == 0) & (c("num_suffix") == "") & c("num_last").is_null() & (c("num_key") != "")
    var = ctx.filter(c("num_suffix") != "")
    if var.is_empty():
        return a.with_columns(n_suffix_variants=pl.lit(0), suffix_list=pl.lit(""))
    agg = var.group_by(keys).agg(c("num_key").n_unique().cast(pl.Int64).alias("_nsv"),
                                 c("num_suffix").unique().sort().str.join(",").alias("_sl"))
    a = left_join(a, agg, keys)
    return a.with_columns(
        n_suffix_variants=pl.when(base).then(c("_nsv").fill_null(0)).otherwise(0),
        suffix_list=pl.when(base).then(c("_sl").fill_null("")).otherwise(pl.lit("")),
    ).drop("_nsv", "_sl")


def _parcel_counts(a: pl.DataFrame, ctx: pl.DataFrame) -> pl.DataFrame:
    """Counts over the parcel's context set, excluding the address itself."""
    bu = c("geocode_cat").is_in(["building", "unit"])
    per_parcel = ctx.group_by("parcel_pid").agg(
        pl.len().cast(pl.Int64).alias("_np"),
        c("street_key").n_unique().cast(pl.Int64).alias("_nstreets"),
        c("in_building_spatial").sum().cast(pl.Int64).alias("_nspatial"),
        bu.sum().cast(pl.Int64).alias("_nbu"),
    )
    # "Other dwellings" are the occupiable leaves that compete with this address: those sharing its
    # street and base number (100, 100A, UNIT 1 100) plus anything its number range covers.
    is_leaf = c("_leaf") & (c("n_range_desc") == 0)
    grp = ["parcel_pid", "street_key", "base_num_key"]
    leaves = a.filter(c("in_ctx") & is_leaf).group_by(grp).agg(pl.len().cast(pl.Int64).alias("_nsame"))
    # Buildings can straddle parcels, so building occupancy counts every principal address.
    per_b = (a.filter(~c("is_alias") & c("building_pid").is_not_null()).group_by("building_pid")
             .agg(pl.len().cast(pl.Int64).alias("_nb")))
    a = left_join(left_join(left_join(a, per_parcel, "parcel_pid"), leaves, grp), per_b, "building_pid")
    own = lambda e: (c("in_ctx") & e).cast(pl.Int64)  # noqa: E731
    return a.with_columns(
        n_parcel_others=pl.when(c("parcel_pid") == "").then(0).otherwise(c("_np").fill_null(0) - own(pl.lit(True))),
        n_other_dwellings=(c("_nsame").fill_null(0) - own(is_leaf) + c("n_range_leaves")).cast(pl.Int64),
        n_other_streets=c("_nstreets").fill_null(1) - 1,
        n_same_building=pl.when(c("building_pid").is_not_null())
        .then(c("_nb").fill_null(0) - (~c("is_alias")).cast(pl.Int64)).otherwise(0).cast(F64),
        n_others_in_buildings=c("_nspatial").fill_null(0) - own(c("in_building_spatial")),
        n_others_bu_geocode=c("_nbu").fill_null(0) - own(bu),
    ).drop("_np", "_nstreets", "_nspatial", "_nbu", "_nsame", "_nb")


def _regex_flag(col: str, patterns: list[str]) -> pl.Expr:
    if not patterns:
        return pl.lit(False)
    rx = "|".join(f"(?:{p})" for p in patterns)
    return c(col).cast(pl.Utf8).fill_null("").str.to_uppercase().str.contains(rx)


def compute_features(addr: pl.DataFrame, parcel_ctx: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Return ``addr`` with raw context columns and the model features (``f_*``).

    ``addr`` must hold the component columns from
    :func:`shell_identifier.components.build_components` plus ``parcel_pid``,
    ``building_pid``, ``in_building_spatial`` and metric ``_x`` / ``_y``.
    """
    fcfg, scfg = cfg["features"], cfg["spatial"]
    tol = float(scfg["stacked_tolerance_m"])
    a = ensure_columns(addr, {"alias_principal": "", "primary_secondary": "", "legal_parcel_id": "",
                              "geocode_type": "", "n_ps_secondaries": 0})
    a = a.with_row_index("_row").with_columns(
        parcel_pid=c("parcel_pid").cast(pl.Utf8).fill_null(""),
        is_alias=c("alias_principal").cast(pl.Utf8).fill_null("").str.to_uppercase() == "A",
        building_pid=pl.when(c("building_pid").cast(pl.Utf8) != "").then(c("building_pid").cast(pl.Utf8)),
        in_building_spatial=c("in_building_spatial").fill_null(False).cast(pl.Boolean),
        xy_key=pl.concat_str([(c("_x") / tol).round().cast(pl.Int64).cast(pl.Utf8), pl.lit("_"),
                              (c("_y") / tol).round().cast(pl.Int64).cast(pl.Utf8)]),
        _nf=c("num_first").cast(F64),
        _nl=c("num_last").cast(F64),
        geocode_type=c("geocode_type").cast(pl.Utf8).fill_null(""),
    )
    a = a.with_columns(in_ctx=~c("is_alias") & (c("parcel_pid") != ""),
                       geocode_type=map_values(a, "geocode_type", normalise_geocode_type))
    a = a.with_columns(geocode_cat=map_values(a, "geocode_type", geocode_category))

    a = finalise_components(_promote_vanity_names(a))
    weights = fcfg["child_weights"]
    a = a.with_columns(
        child_w=c("node_type").replace_strict(list(weights), [float(v) for v in weights.values()],
                                              default=1.0, return_dtype=F64),
        *[(c("node_type") == k).cast(pl.Int64).alias(f"_is_{k}") for k in NODE_CATS],
    )
    ctx = a.filter(c("in_ctx"))

    a = _descendants(a, ctx)
    a = _ancestors(a, ctx)
    a = a.with_columns(_leaf=(c("eff_desc") == 0) & (c("node_type") != "ancillary"))
    a = _ranges(a)
    a = _offparcel(a, float(scfg["offparcel_radius_m"]))
    a = _suffix_siblings(a, ctx)
    a = _parcel_counts(a, ctx)

    pcols = ["total_strata_count", "horizontal_strata_count", "vertical_strata_count",
             "n_buildings", "n_dwelling_buildings", "max_levels", "parcel_area_m2"]
    pc = ensure_columns(parcel_ctx, {k: None for k in pcols}).select(["parcel_pid"] + pcols)
    a = left_join(a.drop([k for k in pcols if k in a.columns]), pc.with_columns([c(k).cast(F64) for k in pcols]),
                  "parcel_pid")
    a = a.with_columns(
        c("n_buildings").fill_null(0).cast(pl.Int64),
        c("n_dwelling_buildings").fill_null(0).cast(pl.Int64),
        parent_pid=pl.when(c("parent_pid") != "").then(c("parent_pid")).otherwise(c("range_parent_pid")),
        _ps=c("primary_secondary").cast(pl.Utf8).fill_null("").str.to_uppercase(),
        _nps=c("n_ps_secondaries").cast(F64, strict=False).fill_null(0),
    )

    # ---- model features ---------------------------------------------------
    eff = c("eff_desc")
    suffix_base = (c("n_suffix_variants") > 0) & (eff == 0)
    a = a.with_columns(suffix_base=suffix_base)
    cand = (eff > 0) | (c("n_range_desc") > 0) | c("suffix_base") | (c("n_offparcel_desc") > 0)
    a = a.with_columns(is_candidate=cand)
    cand = c("is_candidate")
    n_other = c("n_other_dwellings")
    small = cand & (n_other >= 1) & (n_other <= int(fcfg["small_group_max_dwellings"]))
    strata = c("total_strata_count")
    strata_known = strata.is_not_null() & (strata > 0)
    n_bld = c("n_dwelling_buildings")
    real_children = c("n_desc") - c("n_desc_secondary") - c("n_desc_ancillary")
    ps, n_ps = c("_ps"), c("_nps")

    f = {
        "log_children": eff.log1p(),
        "single_child": (eff > 0) & (eff <= 1.5) & (c("n_range_desc") == 0),
        "only_secondary_children": (c("n_desc_secondary") > 0) & (real_children == 0),
        "only_ancillary_children": (c("n_desc_ancillary") > 0) & (c("n_desc") == c("n_desc_ancillary")),
        "log_range_children": c("eff_range").log1p(),
        "log_offparcel_children": c("eff_offparcel").log1p(),
        "stacked_children": pl.when(c("n_desc") > 0).then(c("n_desc_stacked") / c("n_desc").clip(lower_bound=1))
        .otherwise(0.0),
        "suffix_base": c("suffix_base"),
        "suffix_multi_variant": c("suffix_base") & (c("n_suffix_variants") >= 2),
        "suffix_single_variant": c("suffix_base") & (c("n_suffix_variants") == 1),
        "strata_matches_siblings": small & strata_known & (strata <= n_other),
        "strata_exceeds_siblings": small & strata_known & (strata >= n_other + 1),
        "building_surplus": small & (n_bld >= n_other + 1),
        "building_deficit": small & (n_bld >= 1) & (n_bld <= n_other),
        "own_building": cand & c("building_pid").is_not_null() & (c("n_same_building") == 0),
        "outside_buildings": cand & ~c("in_building_spatial") & (n_bld >= 1) & (c("n_others_in_buildings") >= 1),
        "multistorey": cand & (c("max_levels").fill_null(0) >= 3),
        "vertical_strata": cand & (c("vertical_strata_count").fill_null(0) >= 2),
        "parcel_level_geocode": cand & (c("geocode_cat") == "parcel") & (c("n_others_bu_geocode") >= 1),
        "gnaf_primary_linked": n_ps >= 1,
        "gnaf_primary_flag_only": (ps == "P") & (n_ps == 0),
        "gnaf_secondary": ps == "S",
        "legal_common_property": _regex_flag("legal_parcel_id", fcfg["legal_common_property_patterns"]),
        "legal_strata_lot": _regex_flag("legal_parcel_id", fcfg["legal_strata_lot_patterns"]),
        "vanity_name": cand & ((c("vanity_name") != "") | c("comp_structure").str.starts_with("NAME "))
        & (c("comp_flat") == ""),
        "leaf_subaddress": (c("mask") != 0) & ~cand,
        "leaf_street_address": (c("mask") == 0) & ~cand,
        "single_address_parcel": (c("n_parcel_others") == 0) & (c("n_offparcel_desc") == 0),
        "has_parent": c("parent_pid") != "",
        "alias": c("is_alias"),
    }
    a = a.with_columns([e.fill_null(False).cast(F64).alias(f"f_{k}") if k not in
                        ("log_children", "log_range_children", "log_offparcel_children", "stacked_children")
                        else e.cast(F64).alias(f"f_{k}") for k, e in f.items()])
    drop = [col for col in a.columns if col.startswith("_is_")] + ["_row", "_leaf", "_ps", "_nps"]
    return a.drop(drop)

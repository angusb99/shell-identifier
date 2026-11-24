"""Feature matrix for the dwelling model (polars).

Four families, all computed per address relative to its parcel:

* text schematics    - flat/level/building components, flat type category, unit
                       numbering style (101 on level 1), suffixes, ranges
* geocode placement  - geocode type, distance to the parcel centroid and boundary,
                       stacking with other addresses, inside/near a building
* building composition - size and storeys of the address's building, how many
                       addresses share it, buildings / floor area / strata on the parcel
* nesting context    - the shell-identifier features (children, ranges, suffix
                       siblings, G-NAF primary/secondary, legal parcel) and its shell logit
"""

from __future__ import annotations

import numpy as np
import polars as pl
import shapely

from ..frames import c, left_join, num
from ..geo import Layer, dissolve_by

NODE_TYPES = ["street", "dwelling", "commercial", "ancillary", "secondary", "level", "structure"]
GEOCODE_CATS = ["parcel", "building", "unit", "other", ""]
CONTEXT_COLS = [
    "specificity", "n_desc", "eff_desc", "n_desc_dwelling", "n_desc_commercial", "n_desc_ancillary",
    "n_desc_secondary", "n_desc_level", "n_desc_structure", "n_desc_stacked", "n_desc_buildings",
    "n_range_desc", "n_range_numbers", "n_offparcel_desc", "n_suffix_variants", "n_parcel_others",
    "n_other_dwellings", "n_other_streets", "n_ancestors", "n_same_building", "n_others_in_buildings",
    "n_buildings", "n_dwelling_buildings", "max_levels", "total_strata_count", "horizontal_strata_count",
    "vertical_strata_count", "parcel_area_m2", "n_ps_secondaries",
]
F64 = pl.Float64
NEAR_BUILDING_M = 100.0


def _flag(e: pl.Expr) -> pl.Expr:
    return e.fill_null(False).cast(F64)


def _parcel_geometry(s: pl.DataFrame, parcels_m: Layer) -> tuple[np.ndarray, np.ndarray]:
    """Parcel geometry for each address (multi-record parcels unioned) and a has-parcel mask."""
    keys, geoms = dissolve_by(parcels_m.df["parcel_pid"].cast(pl.Utf8).to_numpy(), parcels_m.geom)
    idx = s.select(c("parcel_pid").replace_strict(pl.Series(keys), pl.Series(np.arange(len(keys))), default=-1,
                                                  return_dtype=pl.Int64)).to_series().to_numpy()
    has = idx >= 0
    out = np.full(len(idx), None, dtype=object)
    out[has] = geoms[idx[has]]
    return out, has


def _placement(s: pl.DataFrame, parcels_m: Layer) -> dict[str, np.ndarray]:
    pts = shapely.points(s["_x"].to_numpy(), s["_y"].to_numpy())
    geom, has = _parcel_geometry(s, parcels_m)
    area, d_cent, d_edge = (np.full(len(s), np.nan) for _ in range(3))
    if has.any():
        g = geom[has]
        area[has] = shapely.area(g)
        d_cent[has] = shapely.distance(pts[has], shapely.centroid(g))
        d_edge[has] = shapely.distance(pts[has], shapely.boundary(g))
    scale = np.sqrt(area)
    return {"_parcel_area": area, "g_dist_centroid_rel": d_cent / scale, "g_dist_edge_rel": d_edge / scale,
            "g_dist_edge_m": d_edge, "g_no_parcel": (~has).astype(float), "_pts": pts}


def _building_features(s: pl.DataFrame, buildings_m: Layer | None, pts: np.ndarray) -> pl.DataFrame:
    """Per-address building columns (building area/levels/use, nearest building) and parcel building totals."""
    n = s.height
    if buildings_m is None or not len(buildings_m):
        return pl.DataFrame({k: np.full(n, np.nan) for k in (
            "b_area_log", "b_levels", "g_dist_building", "_p_bld_area", "_p_floor", "b_use_residential",
            "b_use_commercial", "p_share_commercial_area")})
    area = shapely.area(buildings_m.geom)
    bdf = buildings_m.df
    use = (c("building_use").cast(pl.Utf8).fill_null("").str.to_uppercase() if "building_use" in bdf.columns
           else pl.lit(None, dtype=pl.Utf8))
    floor = c("total_floor_area").cast(F64, strict=False) if "total_floor_area" in bdf.columns else pl.lit(None, dtype=F64)
    b = bdf.select(
        c("building_pid").cast(pl.Utf8),
        c("parcel_pid").cast(pl.Utf8) if "parcel_pid" in bdf.columns else pl.lit(None, dtype=pl.Utf8).alias("parcel_pid"),
        c("est_levels").cast(F64, strict=False).alias("_levels"),
        pl.Series("_area", area),
        use.alias("_use"),
        floor.alias("_floor"),
    ).with_columns(
        _floor=c("_floor").fill_null(c("_area") * c("_levels").fill_null(1)),
        _commercial=c("_use").str.contains("COMMERC|INDUSTR|RETAIL|OFFICE|MIXED"),
        _residential=c("_use").str.contains("RESIDEN"),
    ).unique("building_pid", keep="first", maintain_order=True)
    by_parcel = (b.filter(c("parcel_pid").is_not_null()).group_by("parcel_pid").agg(
        c("_area").sum().alias("_p_bld_area"), c("_floor").sum().alias("_p_floor"),
        (c("_area") * c("_commercial").cast(F64)).sum().alias("_p_comm_area")))
    tree = shapely.STRtree(buildings_m.geom)
    (qi, _), dist = tree.query_nearest(pts, max_distance=NEAR_BUILDING_M, return_distance=True, all_matches=False)
    near = np.full(n, NEAR_BUILDING_M)
    near[qi] = dist
    out = left_join(s.select("building_pid", "parcel_pid"), b.drop("parcel_pid"), "building_pid")
    out = left_join(out, by_parcel, "parcel_pid")
    has_use = "building_use" in bdf.columns
    return out.select(
        b_area_log=c("_area").log1p(),
        b_levels=c("_levels"),
        g_dist_building=pl.Series(near),
        _p_bld_area=c("_p_bld_area"),
        _p_floor=c("_p_floor"),
        b_use_residential=c("_residential").cast(F64) if has_use else pl.lit(None, dtype=F64),
        b_use_commercial=c("_commercial").cast(F64) if has_use else pl.lit(None, dtype=F64),
        p_share_commercial_area=(c("_p_comm_area") / c("_p_bld_area")) if has_use else pl.lit(None, dtype=F64),
    )


def dwelling_features(scored: pl.DataFrame, parcels_m: Layer, buildings_m: Layer | None) -> pl.DataFrame:
    """Float feature frame aligned to ``scored`` (null = unknown; the classifier handles it)."""
    s = scored
    place = _placement(s, parcels_m)
    pts = place.pop("_pts")
    bfeat = _building_features(s, buildings_m, pts)
    s = pl.concat([s.select([k for k in s.columns if not k.startswith("_p_")]),
                   pl.DataFrame({k: v for k, v in place.items()}), bfeat], how="horizontal")

    n_parcel = (c("n_parcel_others") + c("in_ctx").cast(pl.Int64)).clip(lower_bound=1).cast(F64)
    stack = pl.len().over(["parcel_pid", "xy_key"]).cast(F64)
    flat_num, level_num = num("flat_number"), num("level_number")
    b_addr = pl.when(c("building_pid").is_not_null()).then(c("n_same_building") + 1).otherwise(0.0)
    F: dict[str, pl.Expr] = {}

    # ---- text schematics ------------------------------------------------------
    for t in NODE_TYPES:
        F[f"t_node_{t}"] = _flag(c("node_type") == t)
    F["t_has_flat"] = _flag(c("comp_flat") != "")
    F["t_has_level"] = _flag(c("comp_level") != "")
    F["t_has_structure"] = _flag(c("comp_structure") != "")
    F["t_has_vanity"] = _flag(c("vanity_name") != "")
    F["t_is_range"] = _flag(c("num_last").is_not_null())
    F["t_has_suffix"] = _flag(c("num_suffix") != "")
    F["t_lot_only"] = _flag(c("is_lot_only"))
    F["t_flat_type_unknown"] = _flag((c("comp_flat") != "") & (c("flat_type_code") == ""))
    F["t_flat_num"] = flat_num
    F["t_flat_num_ge100"] = _flag(flat_num >= 100)
    F["t_flat_prefix"] = _flag(c("flat_number_prefix") != "")
    F["t_level_num"] = level_num
    F["t_level_ground"] = _flag(c("level_type_code").is_in(["G", "LG", "UG"]))
    F["t_level_basement"] = _flag(c("level_type_code").is_in(["B", "SB", "P"]))
    F["t_flat_matches_level"] = _flag((flat_num // 100) == level_num)

    # ---- geocode placement ------------------------------------------------------
    for g in GEOCODE_CATS:
        F[f"g_cat_{g or 'none'}"] = _flag(c("geocode_cat").fill_null("") == g)
    for k in ("g_dist_centroid_rel", "g_dist_edge_rel", "g_dist_edge_m"):
        F[k] = c(k)
    F["g_stack_size"] = stack.log1p()
    F["g_stack_share"] = stack / n_parcel
    F["g_at_modal_point"] = _flag(stack == stack.max().over("parcel_pid"))
    F["g_in_building"] = _flag(c("in_building_spatial"))
    F["g_building_by_link"] = _flag(c("building_match") == "link")
    F["g_snapped"] = _flag(c("parcel_match") == "snapped")
    F["g_no_parcel"] = c("g_no_parcel")

    # ---- building composition -----------------------------------------------------
    F["b_area_log"] = c("b_area_log")
    F["b_levels"] = c("b_levels")
    F["g_dist_building"] = c("g_dist_building")
    F["p_building_coverage"] = c("_p_bld_area") / c("_parcel_area")
    F["p_floor_area_log"] = c("_p_floor").log1p()
    F["p_floor_area_per_addr_log"] = (c("_p_floor") / n_parcel).log1p()
    F["b_use_residential"] = c("b_use_residential")
    F["b_use_commercial"] = c("b_use_commercial")
    F["p_share_commercial_area"] = c("p_share_commercial_area")
    F["b_addr_count"] = b_addr
    F["b_area_per_addr_log"] = c("b_area_log") - b_addr.clip(lower_bound=1).log1p()
    F["p_addr_per_dwelling_building"] = n_parcel / c("n_dwelling_buildings").clip(lower_bound=1)

    # ---- parcel composition ---------------------------------------------------------
    is_bare = c("mask") == 0
    n_bare = is_bare.cast(F64).sum().over("parcel_pid")
    F["c_parcel_addresses_log"] = n_parcel.log1p()
    F["c_share_flats"] = (c("comp_flat") != "").cast(F64).mean().over("parcel_pid")
    F["c_n_bare"] = n_bare
    F["c_only_bare"] = _flag(is_bare & (n_bare == 1))
    F["c_specificity_rank"] = c("specificity").rank("average").over("parcel_pid") / pl.len().over("parcel_pid")

    # ---- nesting context and the expert shell model -----------------------------------
    for k in CONTEXT_COLS:
        F[f"x_{k}"] = num(k) if k in s.columns else pl.lit(None, dtype=F64)
    F["x_stacked_ratio"] = pl.when(c("n_desc") > 0).then(c("n_desc_stacked") / c("n_desc").clip(lower_bound=1)) \
        .otherwise(0.0)
    ps = c("primary_secondary").cast(pl.Utf8).fill_null("").str.to_uppercase()
    F["x_ps_primary"] = _flag(ps == "P")
    F["x_ps_secondary"] = _flag(ps == "S")
    F["x_shell_logit"] = c("shell_logit")
    for k in [k for k in s.columns if k.startswith("f_")]:
        F[k] = c(k)
    out = s.select([e.cast(F64).alias(k) for k, e in F.items()])
    # NaN from 0/0 or log of nothing is "unknown" to the classifier, same as null
    return out.with_columns(pl.all().fill_nan(None))

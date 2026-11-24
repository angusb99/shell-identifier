"""Feature matrix for the dwelling model.

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

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

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


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parcel_geoms(parcels_m: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """One geometry per parcel_pid (multi-record parcels are unioned)."""
    dup = parcels_m["parcel_pid"].duplicated(keep=False)
    single = parcels_m.loc[~dup].set_index("parcel_pid").geometry
    if dup.any():
        merged = parcels_m.loc[dup].dissolve("parcel_pid").geometry
        single = pd.concat([single, merged])
    return gpd.GeoSeries(single, crs=parcels_m.crs)


def dwelling_features(scored: pd.DataFrame, parcels_m: gpd.GeoDataFrame,
                      buildings_m: gpd.GeoDataFrame | None) -> pd.DataFrame:
    """Numeric feature frame aligned to ``scored`` (NaN = unknown; the classifier handles it)."""
    s = scored.reset_index(drop=True)
    F: dict[str, pd.Series | np.ndarray] = {}

    # ---- text schematics ------------------------------------------------------
    for t in NODE_TYPES:
        F[f"t_node_{t}"] = (s["node_type"] == t).astype(float)
    F["t_has_flat"] = (s["comp_flat"] != "").astype(float)
    F["t_has_level"] = (s["comp_level"] != "").astype(float)
    F["t_has_structure"] = (s["comp_structure"] != "").astype(float)
    F["t_has_vanity"] = (s["vanity_name"] != "").astype(float)
    F["t_is_range"] = s["num_last"].notna().astype(float)
    F["t_has_suffix"] = (s["num_suffix"] != "").astype(float)
    F["t_lot_only"] = s["is_lot_only"].astype(float)
    F["t_flat_type_unknown"] = ((s["comp_flat"] != "") & (s["flat_type_code"] == "")).astype(float)
    flat_num = _num(s["flat_number"])
    level_num = _num(s["level_number"])
    F["t_flat_num"] = flat_num
    F["t_flat_num_ge100"] = (flat_num >= 100).astype(float)
    F["t_flat_prefix"] = (s["flat_number_prefix"] != "").astype(float)
    F["t_level_num"] = level_num
    F["t_level_ground"] = s["level_type_code"].isin(["G", "LG", "UG"]).astype(float)
    F["t_level_basement"] = s["level_type_code"].isin(["B", "SB", "P"]).astype(float)
    F["t_flat_matches_level"] = ((flat_num // 100) == level_num).astype(float)

    # ---- geocode placement ------------------------------------------------------
    gcat = s["geocode_cat"].fillna("")
    for c in GEOCODE_CATS:
        F[f"g_cat_{c or 'none'}"] = (gcat == c).astype(float)
    pts = shapely.points(s["_x"].to_numpy(), s["_y"].to_numpy())
    pgeom = _parcel_geoms(parcels_m)
    geom = pgeom.reindex(s["parcel_pid"]).to_numpy()
    has_parcel = ~pd.isna(geom)
    area = np.full(len(s), np.nan)
    d_cent = np.full(len(s), np.nan)
    d_edge = np.full(len(s), np.nan)
    if has_parcel.any():
        g = geom[has_parcel]
        area[has_parcel] = shapely.area(g)
        d_cent[has_parcel] = shapely.distance(pts[has_parcel], shapely.centroid(g))
        d_edge[has_parcel] = shapely.distance(pts[has_parcel], shapely.boundary(g))
    scale = np.sqrt(area)
    F["g_dist_centroid_rel"] = d_cent / scale
    F["g_dist_edge_rel"] = d_edge / scale
    F["g_dist_edge_m"] = d_edge
    key = s["parcel_pid"] + "|" + s["xy_key"]
    stack = key.map(key.value_counts())
    n_parcel = (s["n_parcel_others"] + s["in_ctx"].astype(int)).clip(lower=1)
    F["g_stack_size"] = np.log1p(stack)
    F["g_stack_share"] = stack / n_parcel
    F["g_at_modal_point"] = (stack == stack.groupby(s["parcel_pid"]).transform("max")).astype(float)
    F["g_in_building"] = s["in_building_spatial"].astype(float)
    F["g_building_by_link"] = (s["building_match"] == "link").astype(float)
    F["g_snapped"] = (s["parcel_match"] == "snapped").astype(float)
    F["g_no_parcel"] = (~has_parcel).astype(float)

    # ---- building composition -----------------------------------------------------
    nan = np.full(len(s), np.nan)
    for c in ("b_area_log", "b_levels", "g_dist_building", "p_building_coverage", "p_floor_area_log",
              "p_floor_area_per_addr_log", "b_use_residential", "b_use_commercial", "p_share_commercial_area"):
        F[c] = nan
    if buildings_m is not None and not buildings_m.empty:
        b = buildings_m.set_index("building_pid")
        barea = b.geometry.area
        F["b_area_log"] = np.log1p(s["building_pid"].map(barea)).to_numpy()
        F["b_levels"] = s["building_pid"].map(b["est_levels"]).to_numpy()
        tree = shapely.STRtree(b.geometry.to_numpy())
        idx, dist = tree.query_nearest(pts, max_distance=100.0, return_distance=True, all_matches=False)
        near = np.full(len(s), 100.0)
        near[idx[0]] = dist
        F["g_dist_building"] = near
        floor = b["total_floor_area"].fillna(barea * b["est_levels"].fillna(1))
        by_parcel = pd.DataFrame({"area": barea, "floor": floor, "parcel": b["parcel_pid"]}).groupby("parcel").sum()
        F["p_building_coverage"] = (s["parcel_pid"].map(by_parcel["area"]) / area).to_numpy()
        pfloor = s["parcel_pid"].map(by_parcel["floor"])
        F["p_floor_area_log"] = np.log1p(pfloor).to_numpy()
        F["p_floor_area_per_addr_log"] = np.log1p(pfloor / n_parcel).to_numpy()
        if "building_use" in b:
            use = b["building_use"].fillna("").astype(str).str.upper()
            commercial = use.str.contains("COMMERC|INDUSTR|RETAIL|OFFICE|MIXED")
            buse = s["building_pid"].map(use)
            F["b_use_residential"] = np.where(buse.notna(), buse.str.contains("RESIDEN").astype(float), np.nan)
            F["b_use_commercial"] = np.where(buse.notna(), s["building_pid"].map(commercial).astype(float), np.nan)
            ca = pd.Series(np.where(commercial, barea, 0.0), index=b.index).groupby(b["parcel_pid"]).sum()
            F["p_share_commercial_area"] = (s["parcel_pid"].map(ca) / s["parcel_pid"].map(by_parcel["area"])).to_numpy()
    F["b_addr_count"] = np.where(s["building_pid"].notna(), s["n_same_building"] + 1, 0)
    F["b_area_per_addr_log"] = np.asarray(F["b_area_log"]) - np.log1p(np.clip(F["b_addr_count"], 1, None))
    F["p_addr_per_dwelling_building"] = n_parcel / s["n_dwelling_buildings"].clip(lower=1)

    # ---- parcel composition ---------------------------------------------------------
    by_parcel_id = s["parcel_pid"]
    F["c_parcel_addresses_log"] = np.log1p(n_parcel)
    F["c_share_flats"] = (s["comp_flat"] != "").groupby(by_parcel_id).transform("mean").to_numpy()
    n_bare = (s["mask"] == 0).groupby(by_parcel_id).transform("sum")
    F["c_n_bare"] = n_bare.to_numpy()
    F["c_only_bare"] = ((s["mask"] == 0) & (n_bare == 1)).astype(float)
    F["c_specificity_rank"] = s.groupby("parcel_pid")["specificity"].rank(pct=True).to_numpy()

    # ---- nesting context and the expert shell model -----------------------------------
    for c in CONTEXT_COLS:
        F[f"x_{c}"] = _num(s[c]) if c in s else np.nan
    F["x_stacked_ratio"] = np.where(s["n_desc"] > 0, s["n_desc_stacked"] / s["n_desc"].clip(lower=1), 0.0)
    ps = s["primary_secondary"].fillna("").astype(str).str.upper()
    F["x_ps_primary"] = (ps == "P").astype(float)
    F["x_ps_secondary"] = (ps == "S").astype(float)
    F["x_shell_logit"] = s["shell_logit"]
    for c in [c for c in s.columns if c.startswith("f_")]:
        F[c] = s[c].astype(float)
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in F.items()}, index=s.index)

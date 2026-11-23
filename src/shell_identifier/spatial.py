"""Spatial assignment of addresses to base parcels and buildings, and per-parcel context."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS


def metric_crs(crs, configured: str = "auto") -> str:
    """Equal-area metric CRS for Australia matching the datum of ``crs``."""
    if configured and configured != "auto":
        return configured
    epsg = CRS.from_user_input(crs).to_epsg() if crs else None
    if epsg in (4283, 3577) or (epsg is not None and 28348 <= epsg <= 28358):
        return "EPSG:3577"  # GDA94 / Australian Albers
    return "EPSG:9473"      # GDA2020 / Australian Albers


def _points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"_i": np.arange(len(gdf))}, geometry=gdf.geometry.values, crs=gdf.crs)


def assign_parcels(addr_m: gpd.GeoDataFrame, parcels_m: gpd.GeoDataFrame, snap_m: float) -> pd.DataFrame:
    """Parcel for each address: containing base parcel, else nearest within ``snap_m``.

    Returns a frame aligned to ``addr_m`` with ``parcel_pid``, ``parcel_match``
    (within / snapped / none) and ``parcel_hits`` (number of containing parcels).
    """
    pts = _points(addr_m)
    right = gpd.GeoDataFrame({"parcel_pid": parcels_m["parcel_pid"].values, "_area": parcels_m.area.values},
                             geometry=parcels_m.geometry.values, crs=parcels_m.crs)
    hit = gpd.sjoin(pts, right, how="inner", predicate="within")
    hits = hit.groupby("_i").size()
    best = hit.sort_values(["_i", "_area"]).drop_duplicates("_i").set_index("_i")["parcel_pid"]

    out = pd.DataFrame(index=np.arange(len(addr_m)))
    out["parcel_pid"] = best.reindex(out.index)
    out["parcel_match"] = np.where(out["parcel_pid"].notna(), "within", "none")
    out["parcel_hits"] = hits.reindex(out.index).fillna(0).astype(int)

    missing = out["parcel_pid"].isna().to_numpy()
    if missing.any() and snap_m > 0:
        near = gpd.sjoin_nearest(pts[missing], right, how="inner", max_distance=snap_m, distance_col="_dist")
        near = near.sort_values(["_i", "_dist"]).drop_duplicates("_i").set_index("_i")
        out.loc[near.index, "parcel_pid"] = near["parcel_pid"]
        out.loc[near.index, "parcel_match"] = "snapped"
    out.index = addr_m.index
    return out


def assign_buildings(addr_m: gpd.GeoDataFrame, buildings_m: gpd.GeoDataFrame | None,
                     links: pd.DataFrame | None, use_links: bool = True, link_min_conf: float = 0) -> pd.DataFrame:
    """Building for each address: footprint containing the geocode, else the link table.

    Returns ``building_pid``, ``building_match`` (spatial / link / '') and
    ``in_building_spatial`` aligned to ``addr_m``.
    """
    out = pd.DataFrame(index=addr_m.index)
    out["building_pid"] = pd.Series(np.nan, index=addr_m.index, dtype=object)
    out["building_match"] = ""
    out["in_building_spatial"] = False
    if buildings_m is not None and not buildings_m.empty:
        pts = _points(addr_m)
        right = gpd.GeoDataFrame({"building_pid": buildings_m["building_pid"].values, "_area": buildings_m.area.values},
                                 geometry=buildings_m.geometry.values, crs=buildings_m.crs)
        hit = gpd.sjoin(pts, right, how="inner", predicate="within")
        best = hit.sort_values(["_i", "_area"]).drop_duplicates("_i").set_index("_i")["building_pid"]
        idx = addr_m.index[best.index.to_numpy()]
        out.loc[idx, "building_pid"] = best.to_numpy()
        out.loc[idx, "building_match"] = "spatial"
        out.loc[idx, "in_building_spatial"] = True

    if use_links and links is not None and not links.empty and "address_detail_pid" in links:
        lk = links.copy()
        lk["relationship_confidence"] = pd.to_numeric(lk.get("relationship_confidence"), errors="coerce").fillna(0)
        lk = lk[lk["relationship_confidence"] >= link_min_conf]
        lk["_prim"] = lk.get("primary_building", pd.Series("", index=lk.index)).fillna("").str.upper().eq("YES")
        lk = lk.sort_values(["_prim", "relationship_confidence"], ascending=False).drop_duplicates("address_detail_pid")
        linked = addr_m["address_pid"].map(lk.set_index("address_detail_pid")["building_pid"])
        fill = out["building_pid"].isna() & linked.notna()
        out.loc[fill, "building_pid"] = linked[fill]
        out.loc[fill, "building_match"] = "link"
    return out


def buildings_to_parcels(buildings_m: gpd.GeoDataFrame, parcels_m: gpd.GeoDataFrame,
                         building_cadastre: pd.DataFrame | None = None) -> pd.Series:
    """Parcel of each building: the link table where it names a base parcel, else its representative point."""
    result = pd.Series(np.nan, index=buildings_m.index, dtype=object)
    base_ids = set(parcels_m["parcel_pid"])
    if building_cadastre is not None and not building_cadastre.empty:
        bc = building_cadastre[building_cadastre["cadastre_pid"].isin(base_ids)].copy()
        bc["relationship_confidence"] = pd.to_numeric(bc["relationship_confidence"], errors="coerce").fillna(0)
        bc = bc.sort_values("relationship_confidence", ascending=False).drop_duplicates("building_pid")
        result = buildings_m["building_pid"].map(bc.set_index("building_pid")["cadastre_pid"]).astype(object)
    todo = result.isna().to_numpy()
    if todo.any():
        pts = gpd.GeoDataFrame({"_i": np.flatnonzero(todo)},
                               geometry=buildings_m.geometry[todo].representative_point().values, crs=buildings_m.crs)
        hit = gpd.sjoin(pts, parcels_m[["parcel_pid", "geometry"]], how="inner", predicate="within")
        hit = hit.drop_duplicates("_i")
        result.iloc[hit["_i"].to_numpy()] = hit["parcel_pid"].to_numpy()
    return result


def parcel_context(parcels_m: gpd.GeoDataFrame, buildings_m: gpd.GeoDataFrame | None, min_dwelling_area: float) -> pd.DataFrame:
    """One row per parcel_pid: area, strata counts, building counts and tallest building."""
    p = parcels_m.copy()
    p["_area"] = p.area
    agg = {"_area": "sum"}
    for col in ("total_strata_count", "horizontal_strata_count", "vertical_strata_count"):
        if col in p:
            agg[col] = "max"
    ctx = p.groupby("parcel_pid").agg(agg).rename(columns={"_area": "parcel_area_m2"})
    for col in ("total_strata_count", "horizontal_strata_count", "vertical_strata_count"):
        if col not in ctx:
            ctx[col] = np.nan

    ctx["n_buildings"] = 0
    ctx["n_dwelling_buildings"] = 0
    ctx["max_levels"] = np.nan
    if buildings_m is not None and not buildings_m.empty and "parcel_pid" in buildings_m:
        b = buildings_m[buildings_m["parcel_pid"].notna()].copy()
        b["_area"] = b.area
        b["_dw"] = b["_area"] >= min_dwelling_area
        g = b.groupby("parcel_pid")
        ctx["n_buildings"] = g.size().reindex(ctx.index).fillna(0).astype(int)
        ctx["n_dwelling_buildings"] = g["_dw"].sum().reindex(ctx.index).fillna(0).astype(int)
        ctx["max_levels"] = g["est_levels"].max().reindex(ctx.index)
    return ctx

"""Load Geoscape Cadastre (layer ``cadastre``) and select the base parcels.

Geoscape Cadastre stores one single-part polygon per record; multi-part parcels
appear as several records sharing ``cadastre_pid``. The base representation is
flagged with ``base_parcel = "Yes"``; strata/stratum parcels overlay it.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ._io import lower_columns

# 10-character aliases used by the Shapefile / MapInfo deliveries.
CADASTRE_ALIASES = {
    "CD_PLY_PID": "cadastre_polygon_pid", "CAD_PID": "cadastre_pid", "CNTRB_ID": "contributor_id",
    "DT_CREATE": "date_created", "DT_MOD": "date_modified", "PARCEL_ID": "parcel_id",
    "CNTRB_STTS": "contributor_status", "TITLE_STTS": "title_status", "PARCL_TYPE": "parcel_type",
    "BASE_PARCL": "base_parcel", "STRATA_TYP": "strata_type", "TOT_STR_CN": "total_strata_count",
    "HOR_STR_CN": "horizontal_strata_count", "VER_STR_CN": "vertical_strata_count",
}
STRATA_COUNT_COLS = ["total_strata_count", "horizontal_strata_count", "vertical_strata_count"]


def load_cadastre(path: str | Path, layer: str | None = None, bbox=None, use_base_parcels: bool = True):
    """Return ``(base_parcels, strata_parcels)`` GeoDataFrames.

    ``base_parcels`` gains ``parcel_pid`` (= ``cadastre_pid``) and strata counts;
    where the strata count attributes are empty they are derived by counting
    strata polygons whose representative point falls in the base parcel.
    """
    gdf = gpd.read_file(path, layer=layer, bbox=bbox, engine="pyogrio")
    gdf = lower_columns(gdf, CADASTRE_ALIASES)
    if "cadastre_pid" not in gdf:
        raise KeyError("cadastre layer has no cadastre_pid / CAD_PID column")
    ptype = gdf.get("parcel_type", pd.Series("", index=gdf.index)).fillna("").str.upper()
    base_flag = gdf.get("base_parcel", pd.Series("", index=gdf.index)).fillna("").str.upper()

    if use_base_parcels and (base_flag == "YES").any():
        is_base = base_flag == "YES"
    elif (ptype == "LOT").any():
        is_base = ptype == "LOT"
    else:
        is_base = pd.Series(True, index=gdf.index)
    is_strata = ptype.isin(["STRATA", "STRATUM"])

    base = gdf[is_base].copy()
    strata = gdf[is_strata].copy()
    base["parcel_pid"] = base["cadastre_pid"].astype(str)
    for col in STRATA_COUNT_COLS:
        base[col] = pd.to_numeric(base[col], errors="coerce") if col in base else np.nan

    if not strata.empty and base[STRATA_COUNT_COLS].isna().all(axis=1).any():
        pts = gpd.GeoDataFrame(
            {"strata_type": strata.get("strata_type", pd.Series("", index=strata.index)).fillna("").str.upper()},
            geometry=strata.geometry.representative_point(), crs=strata.crs,
        )
        hit = gpd.sjoin(pts, base[["parcel_pid", "geometry"]], how="inner", predicate="within")
        derived = pd.DataFrame({
            "total_strata_count": hit.groupby("parcel_pid").size(),
            "horizontal_strata_count": hit[hit["strata_type"] == "HORIZONTAL"].groupby("parcel_pid").size(),
            "vertical_strata_count": hit[hit["strata_type"] == "VERTICAL"].groupby("parcel_pid").size(),
        }).fillna(0)
        for col in STRATA_COUNT_COLS:
            fill = base["parcel_pid"].map(derived[col])
            base[col] = base[col].fillna(fill)
    return base.reset_index(drop=True), strata.reset_index(drop=True)

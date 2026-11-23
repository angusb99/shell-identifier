"""Load Geoscape Buildings: the ``buildings`` footprints plus the aspatial
``building_address`` and ``building_cadastre`` link tables."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ._io import lower_columns, read_table

BUILDING_ALIASES = {
    "BLD_PID": "building_pid", "BLD_AREA": "building_area", "TOT_FLR_AR": "total_floor_area",
    "ROOF_HGT": "roof_height", "EAVE_HGT": "eave_height", "BLD_USE": "building_use",
    "ADDR_CNT": "address_count", "PRIM_BLD": "primary_building_flag",
}


def load_buildings(path: str | Path, layer: str | None = "buildings", bbox=None,
                   height_table: str | Path | None = None, storey_height_m: float = 3.1) -> gpd.GeoDataFrame:
    """Load building footprints and estimate storeys (``est_levels``).

    Storeys come from ``total_floor_area / building_area`` when both exist,
    otherwise from ``roof_height / storey_height_m`` (Building Height insight
    pack, joined from ``height_table`` on ``building_pid`` if given).
    """
    try:
        gdf = gpd.read_file(path, layer=layer, bbox=bbox, engine="pyogrio")
    except Exception:
        if layer is None:
            raise
        gdf = gpd.read_file(path, bbox=bbox, engine="pyogrio")
    gdf = lower_columns(gdf, BUILDING_ALIASES)
    if height_table:
        heights = lower_columns(read_table(height_table), BUILDING_ALIASES)
        gdf = gdf.merge(heights[["building_pid", "roof_height"]], on="building_pid", how="left", suffixes=("", "_hp"))
    for col in ("building_area", "total_floor_area", "roof_height"):
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce") if col in gdf else np.nan
    levels = gdf["total_floor_area"] / gdf["building_area"]
    levels = levels.where(levels.notna(), gdf["roof_height"] / storey_height_m)
    gdf["est_levels"] = np.clip(np.round(levels), 1, None)
    return gdf.reset_index(drop=True)


def load_building_address(path: str | Path, layer: str | None = "building_address") -> pd.DataFrame:
    df = lower_columns(read_table(path, layer))
    df["relationship_confidence"] = pd.to_numeric(df.get("relationship_confidence"), errors="coerce")
    return df


def load_building_cadastre(path: str | Path, layer: str | None = "building_cadastre") -> pd.DataFrame:
    df = lower_columns(read_table(path, layer))
    df["relationship_confidence"] = pd.to_numeric(df.get("relationship_confidence"), errors="coerce")
    return df

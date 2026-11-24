"""Geoscape Buildings: footprints with estimated storeys, plus the optional
``building_address`` and ``building_cadastre`` link tables (polars + shapely)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import shapely

from ..frames import c
from ..geo import Layer, reproject
from ._io import open_table, read_table, rename_lower

BUILDING_ALIASES = {
    "BLD_PID": "building_pid", "BLD_AREA": "building_area", "TOT_FLR_AR": "total_floor_area",
    "ROOF_HGT": "roof_height", "EAVE_HGT": "eave_height", "BLD_USE": "building_use",
    "ADDR_CNT": "address_count", "PRIM_BLD": "primary_building_flag",
}
AREA_CRS = "EPSG:9473"


def prepare_buildings(raw: Layer, storey_height_m: float = 3.1, heights: pl.DataFrame | None = None) -> Layer:
    """Canonical building layer with ``building_pid`` and ``est_levels``.

    Storeys come from ``total_floor_area / building_area`` when both exist,
    otherwise from ``roof_height / storey_height_m``. A missing ``building_area``
    is measured from the footprint.
    """
    df = raw.df
    if "building_pid" not in df.columns:
        raise KeyError(f"buildings have no building_pid / BLD_PID column; columns are {df.columns}")
    df = df.with_columns(c("building_pid").cast(pl.Utf8))
    if heights is not None and "roof_height" in heights.columns:
        df = df.join(heights.select(c("building_pid").cast(pl.Utf8), c("roof_height").alias("_roof_hp")),
                     on="building_pid", how="left", maintain_order="left")
    num = {k: (c(k).cast(pl.Float64, strict=False) if k in df.columns else pl.lit(None, dtype=pl.Float64)).alias(k)
           for k in ("building_area", "total_floor_area", "roof_height")}
    df = df.with_columns(*num.values())
    if "_roof_hp" in df.columns:
        df = df.with_columns(roof_height=pl.coalesce(c("roof_height"), c("_roof_hp").cast(pl.Float64, strict=False)))
        df = df.drop("_roof_hp")
    if df["building_area"].null_count() and raw.geom is not None and len(df):
        measured = shapely.area(reproject(raw.geom, raw.crs, AREA_CRS)) if raw.crs else shapely.area(raw.geom)
        df = df.with_columns(building_area=c("building_area").fill_null(pl.Series(measured)))
    levels = c("total_floor_area") / c("building_area")
    levels = pl.when(levels.is_not_null() & levels.is_finite()).then(levels).otherwise(c("roof_height") / storey_height_m)
    df = df.with_columns(est_levels=levels.round(0).clip(lower_bound=1))
    return Layer(df, raw.geom, raw.crs, dict(raw.meta))


def load_buildings(path: str | Path, layer: str | None = "buildings", bbox=None,
                   height_table: str | Path | None = None, storey_height_m: float = 3.1,
                   where: str | None = None) -> Layer:
    """Read building footprints (any supported format) and estimate storeys."""
    df, geom, crs = open_table(path, layer).read(where=where, bbox=bbox)
    if geom is None:
        raise ValueError(f"buildings {path} have no geometry column")
    heights = rename_lower(read_table(height_table), BUILDING_ALIASES) if height_table else None
    return prepare_buildings(Layer(rename_lower(df, BUILDING_ALIASES), geom, crs), storey_height_m, heights)


def canonical_links(df: pl.DataFrame) -> pl.DataFrame:
    """Link table with string ids; ``address_pid`` is the G-NAF ADDRESS_DETAIL_PID."""
    df = rename_lower(df, BUILDING_ALIASES)
    if "address_detail_pid" in df.columns:
        df = df.drop("address_pid", strict=False).rename({"address_detail_pid": "address_pid"})
    casts = [c(k).cast(pl.Utf8) for k in ("address_pid", "building_pid", "cadastre_pid") if k in df.columns]
    if "relationship_confidence" in df.columns:
        casts.append(c("relationship_confidence").cast(pl.Float64, strict=False))
    return df.with_columns(casts)


def load_building_address(path: str | Path, layer: str | None = "building_address") -> pl.DataFrame:
    return canonical_links(read_table(path, layer))


def load_building_cadastre(path: str | Path, layer: str | None = "building_cadastre") -> pl.DataFrame:
    return canonical_links(read_table(path, layer))


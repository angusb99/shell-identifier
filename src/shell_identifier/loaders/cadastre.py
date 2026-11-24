"""Geoscape Cadastre: select the base parcels and their strata counts (polars + shapely).

Geoscape Cadastre stores one single-part polygon per record; multi-part parcels
appear as several records sharing ``cadastre_pid``. The base representation is
flagged with ``base_parcel = "Yes"``; strata/stratum parcels overlay it.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import shapely

from ..frames import c
from ..geo import Layer, points_within
from ._io import open_table, rename_lower

# 10-character aliases used by the Shapefile / MapInfo deliveries.
CADASTRE_ALIASES = {
    "CD_PLY_PID": "cadastre_polygon_pid", "CAD_PID": "cadastre_pid", "CNTRB_ID": "contributor_id",
    "DT_CREATE": "date_created", "DT_MOD": "date_modified", "PARCEL_ID": "parcel_id",
    "CNTRB_STTS": "contributor_status", "TITLE_STTS": "title_status", "PARCL_TYPE": "parcel_type",
    "BASE_PARCL": "base_parcel", "STRATA_TYP": "strata_type", "TOT_STR_CN": "total_strata_count",
    "HOR_STR_CN": "horizontal_strata_count", "VER_STR_CN": "vertical_strata_count",
}
STRATA_COUNT_COLS = ["total_strata_count", "horizontal_strata_count", "vertical_strata_count"]


def _upper(df: pl.DataFrame, col: str) -> pl.Series:
    """Upper-cased text column ('' for every row when the column is absent)."""
    if col not in df.columns:
        return pl.Series(col, [""] * df.height, dtype=pl.Utf8)
    return df.get_column(col).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_uppercase()


def _nonempty(col: str) -> pl.Expr:
    v = c(col).cast(pl.Utf8).str.strip_chars()
    return pl.when(v != "").then(v)


def prepare_cadastre(raw: Layer, use_base_parcels: bool = True) -> tuple[Layer, Layer]:
    """Split a cadastre layer (lower-case Geoscape names) into ``(base_parcels, strata_parcels)``.

    The parcel key ``parcel_pid`` is taken as given (``cadastre_pid``, or ``parcel_id``
    for a lot/plan key), falling back to ``polygon_pid`` (``cadastre_polygon_pid``) where
    it is empty; polygons sharing a key are parts of one parcel. Without ``base_parcel`` /
    ``parcel_type`` every polygon is treated as a base parcel. Strata counts are Float64;
    where empty they are derived by counting strata polygons inside each base parcel.
    """
    df = raw.df
    if "parcel_pid" not in df.columns and "cadastre_pid" in df.columns:
        df = df.with_columns(parcel_pid=c("cadastre_pid"))
    keys = [_nonempty(k) for k in ("parcel_pid", "polygon_pid") if k in df.columns]
    if not keys:
        raise KeyError(f"cadastre has no parcel id (cadastre_pid / parcel_id / cadastre_polygon_pid); "
                       f"columns are {df.columns}")
    df = df.with_columns(parcel_pid=pl.coalesce(keys))
    if df["parcel_pid"].null_count():
        raise ValueError(f"cadastre: {df['parcel_pid'].null_count():,} polygons have no parcel id")
    ptype = _upper(df, "parcel_type")
    base_flag = _upper(df, "base_parcel")
    if use_base_parcels and (base_flag == "YES").any():
        is_base = (base_flag == "YES").to_numpy()
    elif (ptype == "LOT").any():
        is_base = (ptype == "LOT").to_numpy()
    else:   # a base-parcel extract without type columns: everything except roads
        is_base = (ptype != "ROAD").to_numpy()
    is_strata = ptype.is_in(["STRATA", "STRATUM"]).to_numpy()

    df = df.with_columns(parcel_pid=c("parcel_pid").cast(pl.Utf8),
                         *[(c(k).cast(pl.Float64, strict=False) if k in df.columns else pl.lit(None, dtype=pl.Float64))
                           .alias(k) for k in STRATA_COUNT_COLS])
    layer = Layer(df, raw.geom, raw.crs, dict(raw.meta))
    base, strata = layer.filter(is_base), layer.filter(is_strata)

    no_counts = base.df.select(pl.all_horizontal([c(k).is_null() for k in STRATA_COUNT_COLS])).to_series()
    if len(strata) and no_counts.any():
        rp = shapely.point_on_surface(strata.geom)
        i, j = points_within(shapely.get_x(rp), shapely.get_y(rp), base.geom)
        stype = _upper(strata.df, "strata_type")
        hits = pl.DataFrame({"parcel_pid": base.df["parcel_pid"].gather(j), "_t": stype.gather(i)})
        derived = hits.group_by("parcel_pid").agg(
            pl.len().cast(pl.Float64).alias("_tot"),
            (c("_t") == "HORIZONTAL").sum().cast(pl.Float64).alias("_hor"),
            (c("_t") == "VERTICAL").sum().cast(pl.Float64).alias("_ver"),
        )
        bdf = base.df.join(derived, on="parcel_pid", how="left", maintain_order="left").with_columns(
            total_strata_count=c("total_strata_count").fill_null(c("_tot")),
            horizontal_strata_count=c("horizontal_strata_count").fill_null(c("_hor")),
            vertical_strata_count=c("vertical_strata_count").fill_null(c("_ver")),
        ).drop("_tot", "_hor", "_ver")
        base = base.with_df(bdf)
    return base, strata


def load_cadastre(path: str | Path, layer: str | None = None, bbox=None, use_base_parcels: bool = True,
                  where: str | None = None) -> tuple[Layer, Layer]:
    """Read a Geoscape Cadastre table (any supported format) and return ``(base, strata)`` layers."""
    df, geom, crs = open_table(path, layer or "cadastre").read(where=where, bbox=bbox)
    if geom is None:
        raise ValueError(f"cadastre {path} has no geometry column")
    return prepare_cadastre(Layer(rename_lower(df, CADASTRE_ALIASES), geom, crs), use_base_parcels)

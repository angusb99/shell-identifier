"""Spatial assignment of addresses to base parcels and buildings, and per-parcel context.

Geometry work is vectorised shapely (STRtree) on numpy arrays; every result is a
polars frame aligned to the input rows or keyed by ``parcel_pid``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import shapely
from pyproj import CRS

from .frames import c, left_join
from .geo import Layer, nearest, points_within


def metric_crs(crs, configured: str = "auto") -> str:
    """Equal-area metric CRS for Australia matching the datum of ``crs``."""
    if configured and configured != "auto":
        return configured
    epsg = CRS.from_user_input(crs).to_epsg() if crs else None
    if epsg in (4283, 3577) or (epsg is not None and 28348 <= epsg <= 28358):
        return "EPSG:3577"  # GDA94 / Australian Albers
    return "EPSG:9473"      # GDA2020 / Australian Albers


def _smallest_containing(x, y, layer: Layer, id_col: str) -> pl.DataFrame:
    """For each point: the smallest polygon containing it and how many contain it."""
    i, j = points_within(x, y, layer.geom)
    ids = layer.df.get_column(id_col).cast(pl.Utf8)
    hits = pl.DataFrame({"_i": i, "_a": shapely.area(layer.geom)[j], id_col: ids.gather(j)})
    best = hits.sort(["_i", "_a"]).unique("_i", keep="first", maintain_order=True).select("_i", id_col)
    counts = hits.group_by("_i").agg(pl.len().cast(pl.Int64).alias("_hits"))
    return best.join(counts, on="_i", how="left")


def assign_parcels(x: np.ndarray, y: np.ndarray, parcels_m: Layer, snap_m: float) -> pl.DataFrame:
    """Parcel for each point: containing base parcel, else nearest within ``snap_m``.

    Returns a frame aligned to the points with ``parcel_pid`` ('' when none),
    ``parcel_match`` (within / snapped / none) and ``parcel_hits``.
    """
    n = len(x)
    base = pl.DataFrame({"_i": np.arange(n, dtype=np.int64)})
    out = left_join(base, _smallest_containing(x, y, parcels_m, "parcel_pid"), "_i")
    missing = out.filter(c("parcel_pid").is_null())["_i"].to_numpy()
    snapped = None
    if len(missing) and snap_m > 0:
        i, j, _ = nearest(x[missing], y[missing], parcels_m.geom, snap_m)
        ids = parcels_m.df.get_column("parcel_pid").cast(pl.Utf8)
        snapped = pl.DataFrame({"_i": missing[i], "_snap": ids.gather(j)})
    if snapped is not None and snapped.height:
        out = left_join(out, snapped, "_i")
    else:
        out = out.with_columns(_snap=pl.lit(None, dtype=pl.Utf8))
    return out.select(
        parcel_pid=pl.coalesce(c("parcel_pid"), c("_snap"), pl.lit("")),
        parcel_match=pl.when(c("parcel_pid").is_not_null()).then(pl.lit("within"))
        .when(c("_snap").is_not_null()).then(pl.lit("snapped")).otherwise(pl.lit("none")),
        parcel_hits=c("_hits").fill_null(0),
    )


def assign_buildings(address_pid: pl.Series, x: np.ndarray, y: np.ndarray, buildings_m: Layer | None,
                     links: pl.DataFrame | None, use_links: bool = True, link_min_conf: float = 0) -> pl.DataFrame:
    """Building for each address: footprint containing the geocode, else the link table.

    Returns ``building_pid`` (null when none), ``building_match`` (spatial / link / '')
    and ``in_building_spatial``, aligned to the addresses.
    """
    n = len(x)
    out = pl.DataFrame({"_i": np.arange(n, dtype=np.int64), "address_pid": address_pid.cast(pl.Utf8)})
    if buildings_m is not None and len(buildings_m):
        hit = _smallest_containing(x, y, buildings_m, "building_pid").select("_i", c("building_pid").alias("_spatial"))
        out = left_join(out, hit, "_i")
    else:
        out = out.with_columns(_spatial=pl.lit(None, dtype=pl.Utf8))

    linked = pl.lit(None, dtype=pl.Utf8)
    if use_links and links is not None and links.height and {"address_pid", "building_pid"} <= set(links.columns):
        lk = links.with_columns(
            address_pid=c("address_pid").cast(pl.Utf8),
            building_pid=c("building_pid").cast(pl.Utf8),
            _conf=c("relationship_confidence").cast(pl.Float64, strict=False).fill_null(0)
            if "relationship_confidence" in links.columns else pl.lit(0.0),
            _prim=c("primary_building").cast(pl.Utf8).fill_null("").str.to_uppercase().is_in(["YES", "Y", "TRUE", "1"])
            if "primary_building" in links.columns else pl.lit(False),
        ).filter(c("_conf") >= link_min_conf)
        lk = (lk.sort(["_prim", "_conf"], descending=True, maintain_order=True)
              .unique("address_pid", keep="first", maintain_order=True)
              .select("address_pid", c("building_pid").alias("_link")))
        out = left_join(out, lk, "address_pid")
        linked = c("_link")
    else:
        out = out.with_columns(_link=pl.lit(None, dtype=pl.Utf8))
    return out.select(
        building_pid=pl.coalesce(c("_spatial"), linked),
        building_match=pl.when(c("_spatial").is_not_null()).then(pl.lit("spatial"))
        .when(c("_link").is_not_null()).then(pl.lit("link")).otherwise(pl.lit("")),
        in_building_spatial=c("_spatial").is_not_null(),
    )


def buildings_to_parcels(buildings_m: Layer, parcels_m: Layer, building_cadastre: pl.DataFrame | None = None
                         ) -> pl.Series:
    """Parcel of each building: the link table where it names a base parcel, else its representative point."""
    bid = buildings_m.df.get_column("building_pid").cast(pl.Utf8)
    out = pl.DataFrame({"_i": np.arange(len(bid), dtype=np.int64), "building_pid": bid})
    base_ids = parcels_m.df.get_column("parcel_pid").cast(pl.Utf8).unique()
    if building_cadastre is not None and building_cadastre.height and "cadastre_pid" in building_cadastre.columns:
        bc = (building_cadastre
              .with_columns(building_pid=c("building_pid").cast(pl.Utf8), cadastre_pid=c("cadastre_pid").cast(pl.Utf8),
                            _conf=c("relationship_confidence").cast(pl.Float64, strict=False).fill_null(0)
                            if "relationship_confidence" in building_cadastre.columns else pl.lit(0.0))
              .filter(c("cadastre_pid").is_in(base_ids.implode()))
              .sort("_conf", descending=True, maintain_order=True)
              .unique("building_pid", keep="first", maintain_order=True)
              .select("building_pid", c("cadastre_pid").alias("_link")))
        out = left_join(out, bc, "building_pid")
    else:
        out = out.with_columns(_link=pl.lit(None, dtype=pl.Utf8))
    todo = out.filter(c("_link").is_null())["_i"].to_numpy()
    if len(todo):
        rp = shapely.point_on_surface(buildings_m.geom[todo])
        i, j = points_within(shapely.get_x(rp), shapely.get_y(rp), parcels_m.geom)
        order = np.lexsort((j, i))
        i, j = i[order], j[order]
        first = np.r_[True, i[1:] != i[:-1]] if len(i) else np.empty(0, dtype=bool)
        pids = parcels_m.df.get_column("parcel_pid").cast(pl.Utf8)
        spatial = pl.DataFrame({"_i": todo[i[first]], "_sp": pids.gather(j[first])})
        out = left_join(out, spatial, "_i")
    else:
        out = out.with_columns(_sp=pl.lit(None, dtype=pl.Utf8))
    return out.select(pl.coalesce(c("_link"), c("_sp")).alias("parcel_pid")).to_series()


def parcel_context(parcels_m: Layer, buildings_m: Layer | None, min_dwelling_area: float) -> pl.DataFrame:
    """One row per parcel_pid: area, strata counts, building counts and tallest building."""
    strata_cols = [k for k in ("total_strata_count", "horizontal_strata_count", "vertical_strata_count")
                   if k in parcels_m.df.columns]
    p = parcels_m.df.select(c("parcel_pid").cast(pl.Utf8), *[c(k).cast(pl.Float64, strict=False) for k in strata_cols]
                            ).with_columns(_area=pl.Series(shapely.area(parcels_m.geom)))
    ctx = p.group_by("parcel_pid", maintain_order=True).agg(
        c("_area").sum().alias("parcel_area_m2"), *[c(k).max() for k in strata_cols])
    for k in ("total_strata_count", "horizontal_strata_count", "vertical_strata_count"):
        if k not in ctx.columns:
            ctx = ctx.with_columns(pl.lit(None, dtype=pl.Float64).alias(k))
    if buildings_m is not None and len(buildings_m) and "parcel_pid" in buildings_m.df.columns:
        b = (buildings_m.df.select("parcel_pid", c("est_levels").cast(pl.Float64))
             .with_columns(_area=pl.Series(shapely.area(buildings_m.geom))).filter(c("parcel_pid").is_not_null()))
        agg = b.group_by("parcel_pid").agg(
            pl.len().cast(pl.Int64).alias("n_buildings"),
            (c("_area") >= min_dwelling_area).sum().cast(pl.Int64).alias("n_dwelling_buildings"),
            c("est_levels").max().alias("max_levels"),
        )
        ctx = left_join(ctx, agg, "parcel_pid")
    else:
        ctx = ctx.with_columns(n_buildings=pl.lit(None, dtype=pl.Int64), n_dwelling_buildings=pl.lit(None, dtype=pl.Int64),
                               max_levels=pl.lit(None, dtype=pl.Float64))
    return ctx.with_columns(c("n_buildings").fill_null(0), c("n_dwelling_buildings").fill_null(0))

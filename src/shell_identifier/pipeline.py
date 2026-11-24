"""End-to-end pipeline: components -> spatial assignment -> context features -> scores (all polars)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .components import build_components, format_label
from .config import load_config
from .features import compute_features
from .frames import c, ensure_columns, left_join, lookup
from .geo import Layer, reproject_xy
from .scoring import score
from .spatial import assign_buildings, assign_parcels, buildings_to_parcels, metric_crs, parcel_context

# Columns written to address_scores.csv (feature columns go to address_features.csv).
OUTPUT_COLUMNS = [
    "address_pid", "address_label", "parcel_pid", "parcel_match", "building_pid", "building_match",
    "shell_probability", "shell_class", "parent_pid", "parent_label", "node_type",
    "n_desc", "n_range_desc", "n_offparcel_desc", "n_suffix_variants", "n_parcel_others",
    "primary_secondary", "alias_principal", "geocode_type", "reasons",
]


def run_pipeline(
    addresses: Layer,
    parcels: Layer,
    buildings: Layer | None = None,
    building_address: pl.DataFrame | None = None,
    building_cadastre: pl.DataFrame | None = None,
    cfg: dict | None = None,
) -> dict:
    """Score every address for shell likelihood.

    Parameters are the outputs of :mod:`shell_identifier.loaders` / :mod:`shell_identifier.assets`.
    Returns ``{"addresses": pl.DataFrame, "parcels": pl.DataFrame, "crs": str}``; the
    address frame carries every feature column, the scores and ``x`` / ``y`` in ``crs``.
    """
    ctx = prepare_context(addresses, parcels, buildings, building_address, building_cadastre, cfg)
    return {"addresses": ctx["scored"], "parcels": summarise_parcels(ctx["scored"], ctx["parcel_ctx"]),
            "crs": ctx["crs"]}


def prepare_context(
    addresses: Layer,
    parcels: Layer,
    buildings: Layer | None = None,
    building_address: pl.DataFrame | None = None,
    building_cadastre: pl.DataFrame | None = None,
    cfg: dict | None = None,
) -> dict:
    """Spatial assignment, context features and shell scores, plus the metric layers they used.

    Returns ``scored`` (polars, ``x`` / ``y`` in the address CRS), ``parcels_m`` and
    ``buildings_m`` (metric CRS layers; buildings carry ``parcel_pid``), ``parcel_ctx``,
    ``crs`` and ``crs_m``.
    """
    cfg = cfg or load_config()
    scfg, bcfg = cfg["spatial"], cfg["buildings"]
    crs_m = metric_crs(addresses.crs, scfg.get("metric_crs", "auto"))

    addr = build_components(addresses.df)
    addr = addr.with_columns(format_label(addr))
    x_m, y_m = reproject_xy(*addresses.xy(), addresses.crs, crs_m)
    parcels_m = parcels.to_crs(crs_m)
    buildings_m = buildings.to_crs(crs_m) if buildings is not None and len(buildings) else None

    pa = assign_parcels(x_m, y_m, parcels_m, float(scfg["parcel_snap_distance_m"]))
    ba = assign_buildings(addr["address_pid"], x_m, y_m, buildings_m, building_address,
                          use_links=bool(bcfg["use_building_address_link"]),
                          link_min_conf=float(bcfg["link_min_confidence"]))
    addr = addr.drop([col for col in pa.columns + ba.columns + ["_x", "_y"] if col in addr.columns])
    addr = pl.concat([addr, pa, ba], how="horizontal").with_columns(_x=pl.Series(x_m), _y=pl.Series(y_m))
    if buildings_m is not None:
        bdf = buildings_m.df.drop("parcel_pid", strict=False)
        buildings_m = buildings_m.with_df(bdf.with_columns(buildings_to_parcels(buildings_m, parcels_m,
                                                                                building_cadastre)))
    pctx = parcel_context(parcels_m, buildings_m, float(scfg["min_dwelling_building_area_m2"]))

    addr = ensure_columns(addr, {"alias_principal": "", "primary_secondary": "", "legal_parcel_id": "",
                                 "geocode_type": "", "n_ps_secondaries": 0})
    feats = compute_features(addr, pctx, cfg)
    feats = feats.with_columns(
        parent_label=lookup(c("parent_pid"), feats["address_pid"], feats["address_label"], default=""))
    scored = score(feats, cfg)
    return {"scored": scored, "parcels_m": parcels_m, "buildings_m": buildings_m, "parcel_ctx": pctx,
            "crs": addresses.crs, "crs_m": crs_m}


def summarise_parcels(scored: pl.DataFrame, pctx: pl.DataFrame) -> pl.DataFrame:
    """One row per parcel that has addresses: counts per class and a pattern label."""
    s = scored.filter(c("parcel_pid") != "")
    if s.is_empty():
        return pl.DataFrame()
    p = c("shell_probability")
    s = s.with_columns(_shell=p >= 0.8, _likely=(p >= 0.6) & (p < 0.8), _unc=(p >= 0.4) & (p < 0.6),
                       _sub=c("mask") != 0, _range=c("num_last").is_not_null(), _suffix=c("num_suffix") != "")
    ranked = s.sort("shell_probability", descending=True, maintain_order=True)
    out = ranked.group_by("parcel_pid", maintain_order=True).agg(
        pl.len().alias("n_addresses"),
        c("_shell").sum().alias("n_shell"),
        c("_likely").sum().alias("n_likely_shell"),
        c("_unc").sum().alias("n_uncertain"),
        c("shell_probability").max().round(4).alias("max_probability"),
        c("_sub").sum().alias("n_subaddresses"),
        c("street_key").n_unique().alias("n_streets"),
        c("_range").any().alias("has_range"),
        c("_suffix").any().alias("has_suffix"),
        c("address_pid").first().alias("top_candidate_pid"),
        c("address_label").first().alias("top_candidate_label"),
        c("address_pid").filter(c("_shell") | c("_likely")).str.join(";").alias("shell_pids"),
    ).sort("parcel_pid")
    n_sh = c("n_shell") + c("n_likely_shell")
    out = out.with_columns(pattern=pl.when(c("n_addresses") == 1).then(pl.lit("SINGLE_ADDRESS"))
                           .when(c("n_uncertain") > 0).then(pl.lit("NEEDS_REVIEW"))
                           .when(c("has_range") & (n_sh > 0)).then(pl.lit("RANGE_PRECINCT"))
                           .when((c("n_subaddresses") > 0) & (n_sh > 0)).then(pl.lit("MULTI_UNIT_WITH_SHELL"))
                           .when(c("n_subaddresses") > 0).then(pl.lit("MULTI_UNIT_NO_SHELL"))
                           .when(c("has_suffix") & (n_sh > 0)).then(pl.lit("SUFFIX_GROUP_WITH_SHELL"))
                           .when(c("has_suffix")).then(pl.lit("SUFFIX_GROUP_ALL_OCCUPIABLE"))
                           .when(c("n_streets") > 1).then(pl.lit("MULTI_FRONTAGE"))
                           .otherwise(pl.lit("MULTI_ADDRESS")))
    keep = ["parcel_pid", "n_buildings", "n_dwelling_buildings", "max_levels", "total_strata_count",
            "horizontal_strata_count", "vertical_strata_count", "parcel_area_m2"]
    return left_join(out, pctx.select([k for k in keep if k in pctx.columns]), "parcel_pid")


def write_geopackage(df: pl.DataFrame, path: Path, layer: str, crs) -> None:
    """Point GeoPackage from a frame with ``x`` / ``y`` (geopandas only at this output edge)."""
    import geopandas as gpd

    pdf = df.drop([k for k in ("x", "y") if k in df.columns]).to_pandas()
    gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(df["x"].to_numpy(), df["y"].to_numpy()), crs=crs)
    gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")


def write_outputs(result: dict, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = result["addresses"]
    paths = {
        "scores_csv": out_dir / "address_scores.csv",
        "features_csv": out_dir / "address_features.csv",
        "scores_gpkg": out_dir / "address_scores.gpkg",
        "parcels_csv": out_dir / "parcel_summary.csv",
    }
    cols = [k for k in OUTPUT_COLUMNS if k in scored.columns]
    table = scored.select(cols).with_columns(c("shell_probability").round(4))
    table.write_csv(paths["scores_csv"])
    feat_cols = ["address_pid", "parcel_pid", "shell_logit", "shell_probability"] + \
        sorted(k for k in scored.columns if k.startswith("f_"))
    scored.select(feat_cols).write_csv(paths["features_csv"])
    write_geopackage(table.with_columns(scored.select("x", "y")), paths["scores_gpkg"], "address_scores",
                     result.get("crs"))
    result["parcels"].write_csv(paths["parcels_csv"])
    return paths

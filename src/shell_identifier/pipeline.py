"""End-to-end pipeline: components -> spatial assignment -> context features -> scores."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .components import build_components, format_label
from .config import load_config
from .features import compute_features
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
    addresses: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
    building_address: pd.DataFrame | None = None,
    building_cadastre: pd.DataFrame | None = None,
    cfg: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Score every address for shell likelihood.

    Parameters are the outputs of the loaders in :mod:`shell_identifier.loaders`.
    Returns ``{"addresses": GeoDataFrame, "parcels": DataFrame}`` where the
    address frame carries every feature column as well as the scores.
    """
    ctx = prepare_context(addresses, parcels, buildings, building_address, building_cadastre, cfg)
    return {"addresses": ctx["scored"], "parcels": summarise_parcels(ctx["scored"], ctx["parcel_ctx"])}


def prepare_context(
    addresses: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
    building_address: pd.DataFrame | None = None,
    building_cadastre: pd.DataFrame | None = None,
    cfg: dict | None = None,
) -> dict:
    """Spatial assignment, context features and shell scores, plus the metric layers they used.

    Returns ``scored`` (GeoDataFrame in the input CRS), ``parcels_m`` and
    ``buildings_m`` (metric CRS, buildings carry ``parcel_pid``), ``parcel_ctx``
    and ``crs_m``.
    """
    cfg = cfg or load_config()
    scfg = cfg["spatial"]
    crs_m = metric_crs(addresses.crs, scfg.get("metric_crs", "auto"))

    addr = build_components(addresses)
    addr["address_label"] = format_label(addr)
    addr_m = addr.to_crs(crs_m)
    parcels_m = parcels.to_crs(crs_m)
    buildings_m = buildings.to_crs(crs_m) if buildings is not None else None

    addr = addr.join(assign_parcels(addr_m, parcels_m, float(scfg["parcel_snap_distance_m"])))
    bcfg = cfg["buildings"]
    addr = addr.join(assign_buildings(addr_m, buildings_m, building_address,
                                      use_links=bool(bcfg["use_building_address_link"]),
                                      link_min_conf=float(bcfg["link_min_confidence"])))
    if buildings_m is not None:
        buildings_m = buildings_m.copy()
        buildings_m["parcel_pid"] = buildings_to_parcels(buildings_m, parcels_m, building_cadastre)
    pctx = parcel_context(parcels_m, buildings_m, float(scfg["min_dwelling_building_area_m2"]))

    addr["_x"] = addr_m.geometry.x.to_numpy()
    addr["_y"] = addr_m.geometry.y.to_numpy()
    for col in ("alias_principal", "primary_secondary", "legal_parcel_id", "geocode_type"):
        if col not in addr:
            addr[col] = ""
    if "n_ps_secondaries" not in addr:
        addr["n_ps_secondaries"] = 0

    feats = compute_features(pd.DataFrame(addr.drop(columns="geometry")), pctx, cfg)
    labels = feats.set_index("address_pid")["address_label"]
    feats["parent_label"] = feats["parent_pid"].map(labels).fillna("")
    scored = score(feats, cfg)
    scored = gpd.GeoDataFrame(scored, geometry=addr.geometry.to_numpy(), crs=addresses.crs)
    return {"scored": scored, "parcels_m": parcels_m, "buildings_m": buildings_m, "parcel_ctx": pctx, "crs_m": crs_m}


def summarise_parcels(scored: pd.DataFrame, pctx: pd.DataFrame) -> pd.DataFrame:
    """One row per parcel that has addresses: counts per class and a pattern label."""
    s = scored[scored["parcel_pid"] != ""]
    if s.empty:
        return pd.DataFrame()
    p = s["shell_probability"]
    s = s.assign(
        _shell=p >= 0.8, _likely=(p >= 0.6) & (p < 0.8), _unc=(p >= 0.4) & (p < 0.6),
        _sub=s["mask"] != 0, _range=s["num_last"].notna(), _suffix=s["num_suffix"] != "",
    )
    g = s.groupby("parcel_pid")
    out = pd.DataFrame({
        "n_addresses": g.size(),
        "n_shell": g["_shell"].sum(),
        "n_likely_shell": g["_likely"].sum(),
        "n_uncertain": g["_unc"].sum(),
        "max_probability": g["shell_probability"].max().round(4),
        "n_subaddresses": g["_sub"].sum(),
        "n_streets": g["street_key"].nunique(),
        "has_range": g["_range"].any(),
        "has_suffix": g["_suffix"].any(),
    })
    top = s.sort_values("shell_probability", ascending=False).drop_duplicates("parcel_pid").set_index("parcel_pid")
    out["top_candidate_pid"] = top["address_pid"]
    out["top_candidate_label"] = top["address_label"]
    shells = s[s["_shell"] | s["_likely"]].groupby("parcel_pid")["address_pid"].agg(";".join)
    out["shell_pids"] = shells.reindex(out.index).fillna("")

    pattern = np.select(
        [
            out["n_addresses"] == 1,
            out["n_uncertain"] > 0,
            out["has_range"] & (out["n_shell"] + out["n_likely_shell"] > 0),
            (out["n_subaddresses"] > 0) & (out["n_shell"] + out["n_likely_shell"] > 0),
            out["n_subaddresses"] > 0,
            out["has_suffix"] & (out["n_shell"] + out["n_likely_shell"] > 0),
            out["has_suffix"],
            out["n_streets"] > 1,
        ],
        [
            "SINGLE_ADDRESS", "NEEDS_REVIEW", "RANGE_PRECINCT", "MULTI_UNIT_WITH_SHELL",
            "MULTI_UNIT_NO_SHELL", "SUFFIX_GROUP_WITH_SHELL", "SUFFIX_GROUP_ALL_OCCUPIABLE", "MULTI_FRONTAGE",
        ],
        "MULTI_ADDRESS",
    )
    out["pattern"] = pattern
    out = out.join(pctx[["n_buildings", "n_dwelling_buildings", "max_levels", "total_strata_count",
                         "horizontal_strata_count", "vertical_strata_count", "parcel_area_m2"]], how="left")
    return out.reset_index()


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
    cols = [c for c in OUTPUT_COLUMNS if c in scored]
    table = pd.DataFrame(scored[cols]).copy()
    table["shell_probability"] = table["shell_probability"].round(4)
    table.to_csv(paths["scores_csv"], index=False)
    feat_cols = ["address_pid", "parcel_pid", "shell_logit", "shell_probability"] + \
        sorted(c for c in scored.columns if c.startswith("f_"))
    pd.DataFrame(scored[feat_cols]).to_csv(paths["features_csv"], index=False)
    gpkg = scored[cols + ["geometry"]].copy()
    gpkg.to_file(paths["scores_gpkg"], layer="address_scores", driver="GPKG", engine="pyogrio")
    result["parcels"].to_csv(paths["parcels_csv"], index=False)
    return paths

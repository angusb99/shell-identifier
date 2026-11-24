"""Run the pipeline on polars LazyFrames / DataFrames instead of files.

The four inputs are passed directly, so you choose the exact files and read them
yourself (``pl.scan_parquet``, ``pl.scan_csv`` ...). Only the needed columns are
collected. The default column names follow this layout:

=================  ===================================================================
``labelled_asset`` ``ADDRESS_ID``, ``PD_FLAG``
``ds_asset``       ``ADDRESS_ID``, G-NAF address fields (``FLAT_TYPE_CODE`` / ``FLAT_TYPE``,
                   ``FLAT_NUMBER``, ``LEVEL_*``, ``NUMBER_FIRST``, ``STREET_NAME`` ...),
                   ``LATITUDE`` / ``LONGITUDE`` (or ``LAT`` / ``LON`` / ``LONG``)
``cad``            ``parcel_id``, ``cadastre_polygon_pid``, a WKT polygon column (base parcels)
``bld``            ``building_pid``, a WKT polygon column
=================  ===================================================================

Names are matched case-insensitively; WKT / WKB geometry columns are found by
name (anything with wkt / geom / shape in it) or by their values. Override any
mapping with ``columns={"cad": {"geometry": "GEOM_TEXT"}, ...}``.

Example::

    import polars as pl
    from shell_identifier.api import verify_frames

    result = verify_frames(
        labelled_asset=pl.scan_parquet("D:/data/pd_labels_2026.parquet"),
        ds_asset=pl.scan_parquet("D:/data/address_2026.parquet"),
        cad=pl.scan_csv("D:/data/cadastre_base.csv"),
        bld=pl.scan_csv("D:/data/buildings.csv"),
        out_dir="results/dwellings",
    )
    result["assignments"]     # one row per address: dwelling / primary shell / out of scope
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import polars as pl

from .assets import Inputs, check_datasets, datasets_config, load_inputs
from .config import load_config

FrameLike = pl.LazyFrame | pl.DataFrame

FRAME_COLUMNS: dict[str, dict] = {
    "labelled_asset": {"address_pid": "ADDRESS_ID", "label": "PD_FLAG"},
    "ds_asset": {"address_pid": "ADDRESS_ID"},
    "cad": {"parcel_pid": "parcel_id", "polygon_pid": "cadastre_polygon_pid", "geometry": None},
    "bld": {"building_pid": "building_pid", "geometry": None},
    "parcel_counts": {"parcel_pid": "parcel_id", "n_dwellings": "n_dwellings"},
}


def frame_datasets(labelled_asset: FrameLike | None = None, ds_asset: FrameLike | None = None,
                   cad: FrameLike | None = None, bld: FrameLike | None = None, *,
                   counts: FrameLike | None = None, crs: str | dict = "EPSG:7844",
                   columns: dict[str, dict] | None = None, pd_values=None, oos_values=None,
                   bld_address: FrameLike | None = None, bld_cadastre: FrameLike | None = None) -> dict:
    """A datasets dict (as from a datasets YAML) whose sources are the given frames.

    ``crs`` is the CRS of the coordinates / WKT (one string, or ``{"cad": ..., "bld": ...}``).
    ``columns`` overrides column names per dataset: ``{role: {canonical: your_column}}``.
    """
    frames = {"labelled_asset": labelled_asset, "ds_asset": ds_asset, "cad": cad, "bld": bld,
              "parcel_counts": counts, "bld_address": bld_address, "bld_cadastre": bld_cadastre}
    over: dict = {}
    for role in frames:
        cols = {**FRAME_COLUMNS.get(role, {}), **((columns or {}).get(role, {}))}
        over[role] = {"columns": cols} if cols else {}
        if role in ("ds_asset", "cad", "bld"):
            over[role]["crs"] = crs.get(role, "EPSG:7844") if isinstance(crs, dict) else crs
    if pd_values is not None:
        over["labelled_asset"]["pd_values"] = list(pd_values)
    if oos_values is not None:
        over["labelled_asset"]["oos_values"] = list(oos_values)
    ds = datasets_config(None, over)
    for role, frame in frames.items():
        if frame is not None:
            if not isinstance(frame, (pl.LazyFrame, pl.DataFrame)):
                raise TypeError(f"{role} must be a polars LazyFrame or DataFrame, got {type(frame).__name__}")
            ds[role]["path"] = frame
            ds[role]["table"] = role
            ds[role]["format"] = ds[role].get("format", "auto")
    return ds


def load_frames(labelled_asset: FrameLike | None = None, ds_asset: FrameLike | None = None,
                cad: FrameLike | None = None, bld: FrameLike | None = None, *, cfg: dict | None = None,
                bbox=None, log: Callable = print, **kw) -> Inputs:
    """Canonical inputs (addresses, parcels, buildings, labels, counts) from frames."""
    ds = frame_datasets(labelled_asset, ds_asset, cad, bld, **kw)
    return load_inputs(ds, cfg, labels=labelled_asset is not None, bbox=bbox, log=log)


def check_frames(labelled_asset: FrameLike | None = None, ds_asset: FrameLike | None = None,
                 cad: FrameLike | None = None, bld: FrameLike | None = None, **kw) -> str:
    """Which column each field maps to, and what wasn't found (nothing is collected but a sample)."""
    return check_datasets(frame_datasets(labelled_asset, ds_asset, cad, bld, **kw))


def _context(inputs: Inputs, cfg: dict):
    from .pipeline import prepare_context

    return prepare_context(*inputs.pipeline_args(), cfg)


def verify_frames(*, labelled_asset: FrameLike, ds_asset: FrameLike, cad: FrameLike, bld: FrameLike | None = None,
                  counts: FrameLike | None = None, out_dir: str | Path | None = None, settings=None,
                  cfg: dict | None = None, write_features: bool = False, gpkg: bool = True,
                  log: Callable = print, **kw) -> dict:
    """Fit the dwelling model on the labelled parcels, move misplaced PD labels, find primary shells.

    Returns the result dict (``assignments``, ``parcels``, ``moves``, ``bundle`` ...); with
    ``out_dir`` the usual CSV / parquet / GeoPackage / model files are written there too
    (``result["paths"]``). ``settings`` is a :class:`~shell_identifier.dwellings.EMSettings`.
    """
    from .dwellings import verify, write_outputs

    cfg = cfg or load_config()
    inputs = load_frames(labelled_asset, ds_asset, cad, bld, counts=counts, cfg=cfg, log=log, **kw)
    result = verify(_context(inputs, cfg), inputs.labels, settings, counts=inputs.counts, log=log)
    if out_dir:
        result["paths"] = write_outputs(result, out_dir, gpkg=gpkg, write_features=write_features)
    return result


def predict_frames(model, *, ds_asset: FrameLike, cad: FrameLike, bld: FrameLike | None = None,
                   counts: FrameLike | None = None, out_dir: str | Path | None = None, cfg: dict | None = None,
                   gpkg: bool = True, log: Callable = print, **kw) -> dict:
    """Apply a model from :func:`verify_frames` (``result["bundle"]`` or a ``dwelling_model.joblib`` path)."""
    from .dwellings import predict, write_outputs

    if isinstance(model, (str, Path)):
        import joblib

        model = joblib.load(model)
    cfg = cfg or load_config()
    inputs = load_frames(None, ds_asset, cad, bld, counts=counts, cfg=cfg, log=log, **kw)
    result = predict(_context(inputs, cfg), model, counts=inputs.counts)
    if out_dir:
        result["paths"] = write_outputs(result, out_dir, gpkg=gpkg)
    return result


def score_frames(*, ds_asset: FrameLike, cad: FrameLike, bld: FrameLike | None = None,
                 out_dir: str | Path | None = None, cfg: dict | None = None, log: Callable = print, **kw) -> dict:
    """The label-free shell scorer: ``{"addresses", "parcels"}`` (written to ``out_dir`` if given)."""
    from .pipeline import summarise_parcels, write_outputs

    cfg = cfg or load_config()
    inputs = load_frames(None, ds_asset, cad, bld, cfg=cfg, log=log, **kw)
    ctx = _context(inputs, cfg)
    result = {"addresses": ctx["scored"], "parcels": summarise_parcels(ctx["scored"], ctx["parcel_ctx"]),
              "crs": ctx["crs"]}
    if out_dir:
        result["paths"] = write_outputs(result, out_dir)
    return result

"""The four input datasets - ``labelled_asset``, ``ds_asset``, ``bld``, ``cad`` - located by path.

A datasets YAML (template: ``default_datasets.yaml``; write one with
``shell-identifier datasets template``) gives each dataset a ``path``, a
``table`` name and a column mapping ``canonical: your_column``. Anything left
out falls back to the template and to the usual G-NAF / Geoscape names, so a
working file can be as small as four paths.

Everything up to the spatial joins is polars: tables are scanned lazily with
only the mapped columns, filtered, renamed to canonical names and typed.
:func:`load_inputs` returns canonical :class:`~shell_identifier.geo.Layer` /
``pl.DataFrame`` objects ready for :func:`shell_identifier.pipeline.prepare_context`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import shapely
import yaml

from .frames import c
from .geo import Layer, crs_string
from .loaders import (canonical_links, is_gnaf_release, load_gnaf, open_table, parse_column, prepare_buildings,
                      prepare_cadastre)
from .loaders._io import TableSource
from .loaders.buildings import BUILDING_ALIASES
from .loaders.cadastre import CADASTRE_ALIASES
from .dwellings.labels import labels_from_frame

ROLES = ("labelled_asset", "ds_asset", "bld", "cad", "bld_address", "bld_cadastre", "parcel_counts")
DESCRIPTIONS = {
    "labelled_asset": "PD / out-of-scope labels",
    "ds_asset": "addresses",
    "bld": "building footprints",
    "cad": "cadastre parcels",
    "bld_address": "building-address links",
    "bld_cadastre": "building-cadastre links",
    "parcel_counts": "trusted dwelling counts",
}
# Layer names tried when the configured table isn't in a multi-layer GeoPackage / folder.
TABLE_FALLBACKS = {"bld": ["buildings", "building"], "cad": ["cadastre", "parcels", "parcel"],
                   "bld_address": ["building_address"], "bld_cadastre": ["building_cadastre"],
                   "ds_asset": ["address", "addresses", "address_detail"], "labelled_asset": ["labels", "pd_labels"]}

# Names tried, in order, after the configured one (all case-insensitive).
ALIASES: dict[str, list[str]] = {
    "address_pid": ["address_detail_pid", "address_pid", "address_id", "adp", "gnaf_pid", "gnaf_id"],
    "flat_type_code": ["flat_type_code", "flat_type"],
    "level_type_code": ["level_type_code", "level_type"],
    "street_type_code": ["street_type_code", "street_type"],
    "street_suffix_code": ["street_suffix_code", "street_suffix_type", "street_suffix"],
    "locality_name": ["locality_name", "locality", "suburb", "suburb_name"],
    "state": ["state_abbreviation", "state", "state_code", "state_abbrev"],
    "postcode": ["postcode", "post_code"],
    "geocode_type": ["geocode_type_code", "geocode_type"],
    "ps_primary_pid": ["ps_primary_pid", "primary_pid"],
    "longitude": ["longitude", "lon", "long", "lng"],
    "latitude": ["latitude", "lat"],
    "address_text": ["address_text", "full_address", "address_string", "address_label", "address"],
    "label": ["label", "pd_label", "dwelling_label", "pd_flag", "is_pd", "dwelling_type"],
    "parcel_pid": ["cadastre_pid", "cad_pid", "parcel_pid"],
    "building_pid": ["building_pid", "bld_pid"],
    "n_dwellings": ["n_dwellings", "dwellings", "dwelling_count", "k"],
}
for _short, _long in {**CADASTRE_ALIASES, **BUILDING_ALIASES}.items():
    ALIASES.setdefault(_long, [_long]).append(_short.lower())

ADDRESS_COMPONENTS = ["number_first", "street_name"]
# Columns that are looked for under their usual names even when the YAML leaves them null.
AUTO_DETECT = {"ps_primary_pid", "address_text"}
GEOMETRY_ROLES = {"bld", "cad"}


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

def template_text() -> str:
    return resources.files("shell_identifier").joinpath("default_datasets.yaml").read_text(encoding="utf-8")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def datasets_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Template defaults, overlaid with the YAML at ``path`` and then ``overrides``.

    ``overrides`` is ``{role: {key: value}}``, e.g. ``{"ds_asset": {"path": "D:/data"}}``.
    Relative paths resolve against ``root`` (or the YAML's folder, or the working directory).
    """
    cfg = yaml.safe_load(template_text())
    base_dir = Path.cwd()
    if path:
        with open(path, encoding="utf-8") as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
        base_dir = Path(path).resolve().parent
    # paths given as overrides (command line, Python) are relative to the working directory
    overrides = copy.deepcopy(overrides or {})
    for spec in overrides.values():
        if not isinstance(spec, dict):
            continue
        for key in ("path", "height_table", "localities"):
            v = spec.get(key)
            if v and "://" not in str(v) and not Path(str(v)).is_absolute():
                spec[key] = str(Path.cwd() / str(v))
    cfg = _merge(cfg, overrides)
    root = Path(cfg["root"]) if cfg.get("root") else None
    if root is not None and not root.is_absolute():
        root = base_dir / root
    for role in ROLES:
        spec = cfg.setdefault(role, {})
        spec.setdefault("columns", {})
        p = spec.get("path")
        if p and "://" not in str(p) and not Path(str(p)).is_absolute():
            spec["path"] = str((root or base_dir) / str(p))
        elif not p and root is not None and role in ("labelled_asset", "ds_asset", "bld", "cad"):
            spec["path"] = str(root)
        for key in ("height_table", "localities"):
            v = spec.get(key)
            if v and not Path(str(v)).is_absolute():
                spec[key] = str((root or base_dir) / str(v))
    return cfg


# ---------------------------------------------------------------------------------------------
# column resolution
# ---------------------------------------------------------------------------------------------

@dataclass
class Resolved:
    """A dataset opened and its columns resolved to canonical names."""

    role: str
    spec: dict
    source: TableSource
    mapping: dict[str, str] = field(default_factory=dict)      # canonical -> actual
    missing: list[str] = field(default_factory=list)            # canonical names not found
    notes: list[str] = field(default_factory=list)


def _open(role: str, spec: dict) -> TableSource:
    path, table = spec.get("path"), spec.get("table")
    try:
        return open_table(path, table)
    except FileNotFoundError:
        for alt in TABLE_FALLBACKS.get(role, []):
            try:
                return open_table(path, alt)
            except FileNotFoundError:
                continue
        raise


def resolve(role: str, spec: dict) -> Resolved:
    """Open ``spec['path']`` / ``spec['table']`` and map every configured column."""
    ts = _open(role, spec)
    r = Resolved(role, spec, ts)
    for canon, configured in spec.get("columns", {}).items():
        if canon == "geometry":
            continue
        if configured:
            candidates = [str(configured)] + ALIASES.get(canon, [canon])
        elif canon in AUTO_DETECT:
            candidates = ALIASES.get(canon, [canon])
        else:                      # explicitly set to null: don't use this column
            continue
        actual = next((ts.find(n) for n in candidates if ts.find(n)), None)
        if actual:
            r.mapping[canon] = actual
            if configured and actual.lower() != str(configured).lower():
                r.notes.append(f"{canon}: '{configured}' not found, using '{actual}'")
        elif configured:
            r.missing.append(canon)
    return r


def _geometry_arg(spec: dict) -> str:
    g = spec.get("columns", {}).get("geometry")
    return g if g else "auto"


def _read(r: Resolved, bbox=None) -> tuple[pl.DataFrame, np.ndarray | None, str | None]:
    """Mapped columns renamed to canonical names, plus geometry and CRS."""
    cols = list(dict.fromkeys(r.mapping.values()))
    want_geom = r.role in GEOMETRY_ROLES or (r.role == "ds_asset" and not {"longitude", "latitude"} <= set(r.mapping))
    df, geom, crs = r.source.read(cols, geometry=_geometry_arg(r.spec) if want_geom else None,
                                  where=r.spec.get("filter"), bbox=bbox if r.role in GEOMETRY_ROLES else None)
    rename, seen = {}, set()
    for canon, actual in r.mapping.items():
        if actual in df.columns and actual not in seen:
            rename[actual] = canon
            seen.add(actual)
    df = df.select(list(rename)).rename(rename)
    for canon, actual in r.mapping.items():          # one source column feeding two canonical names
        if canon not in df.columns and rename.get(actual) in df.columns:
            df = df.with_columns(c(rename[actual]).alias(canon))
    configured = r.spec.get("crs")
    if configured and str(configured).lower() != "auto":
        crs = crs_string(configured)
    return df, geom, crs


# ---------------------------------------------------------------------------------------------
# loaders per role
# ---------------------------------------------------------------------------------------------

def _localities(spec: dict) -> set[str] | None:
    p = spec.get("localities")
    if not p:
        return None
    return {line.strip().upper() for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()}


def load_addresses(spec: dict, bbox=None, log: Callable = print) -> Layer:
    """``ds_asset`` -> canonical point layer (``x`` / ``y`` in its CRS)."""
    fmt = str(spec.get("format") or "auto").lower()
    path = spec.get("path")
    if fmt == "gnaf" or (fmt == "auto" and path and is_gnaf_release(path)):
        layer = load_gnaf(path, states=spec.get("states"), crs=spec.get("crs") if spec.get("crs") not in
                          (None, "auto") else "EPSG:7844")
        log(f"ds_asset: G-NAF release {path}: {len(layer):,} active addresses")
    else:
        r = resolve("ds_asset", spec)
        df, geom, crs = _read(r)
        for note in r.notes:
            log(f"ds_asset: {note}")
        has_components = all(k in df.columns for k in ADDRESS_COMPONENTS)
        if fmt == "text" or (not has_components and "address_text" in df.columns):
            if "address_text" not in df.columns:
                raise KeyError("ds_asset: text format needs an address_text column (columns.address_text)")
            df = parse_column(df, "address_text", _localities(spec))
            log(f"ds_asset: parsed {df.height:,} single-line addresses")
        elif not has_components:
            raise KeyError(f"ds_asset: no address components ({ADDRESS_COMPONENTS}) or address_text column found "
                           f"in {r.source.description}; columns are {r.source.columns}")
        if "address_pid" not in df.columns:
            log("ds_asset: no address id column found; numbering rows")
            df = df.with_columns(address_pid=pl.format("ROW{}", pl.int_range(pl.len()).cast(pl.Utf8).str.zfill(9)))
        if "longitude" in df.columns and "latitude" in df.columns:
            df = df.with_columns(x=c("longitude").cast(pl.Float64, strict=False),
                                 y=c("latitude").cast(pl.Float64, strict=False)).drop("longitude", "latitude")
        elif geom is not None:
            pts = np.where(shapely.get_type_id(geom) == 0, geom, shapely.point_on_surface(geom))
            df = df.with_columns(x=pl.Series(shapely.get_x(pts)), y=pl.Series(shapely.get_y(pts)))
        else:
            raise KeyError("ds_asset: needs longitude / latitude columns or a point geometry column")
        if "date_retired" in df.columns:
            before = df.height
            df = df.filter(c("date_retired").cast(pl.Utf8).fill_null("").str.strip_chars() == "").drop("date_retired")
            if df.height < before:
                log(f"ds_asset: dropped {before - df.height:,} retired addresses")
        df = df.with_columns(c("address_pid").cast(pl.Utf8))
        if "ps_primary_pid" in df.columns:
            n_sec = (df.filter(c("ps_primary_pid").is_not_null() & (c("ps_primary_pid").cast(pl.Utf8) != ""))
                     .group_by(c("ps_primary_pid").cast(pl.Utf8).alias("address_pid"))
                     .agg(pl.len().cast(pl.Int64).alias("n_ps_secondaries")))
            df = df.join(n_sec, on="address_pid", how="left", maintain_order="left").with_columns(
                c("n_ps_secondaries").fill_null(0))
        dupes = df.height - df["address_pid"].n_unique()
        if dupes:
            log(f"ds_asset: {dupes:,} duplicate address ids; keeping the first row of each")
            df = df.unique("address_pid", keep="first", maintain_order=True)
        crs = crs or "EPSG:7844"
        layer = Layer(df, None, crs_string(crs))
        log(f"ds_asset: {r.source.description}: {len(layer):,} addresses")
    missing = layer.df.filter(c("x").is_null() | c("y").is_null()).height
    if missing:
        log(f"ds_asset: dropped {missing:,} addresses without coordinates")
        layer = layer.with_df(layer.df.filter(c("x").is_not_null() & c("y").is_not_null()))
    if bbox is not None:
        layer = layer.bounds_filter(bbox)
    return layer


def load_parcels(spec: dict, bbox=None, log: Callable = print) -> tuple[Layer, Layer]:
    """``cad`` -> (base parcels, strata parcels)."""
    r = resolve("cad", spec)
    if "parcel_pid" not in r.mapping:
        raise KeyError(f"cad: no parcel id column (columns.parcel_pid) in {r.source.description}; "
                       f"columns are {r.source.columns}")
    df, geom, crs = _read(r, bbox)
    if geom is None:
        raise KeyError(f"cad: no geometry column found in {r.source.description} (set columns.geometry)")
    base, strata = prepare_cadastre(Layer(df, geom, crs), bool(spec.get("use_base_parcels", True)))
    log(f"cad: {r.source.description}: {len(base):,} base parcel polygons, {len(strata):,} strata polygons")
    return base, strata


def load_building_layer(spec: dict, storey_height_m: float = 3.1, bbox=None, log: Callable = print) -> Layer:
    """``bld`` -> building footprints with ``est_levels``."""
    r = resolve("bld", spec)
    if "building_pid" not in r.mapping:
        raise KeyError(f"bld: no building id column (columns.building_pid) in {r.source.description}; "
                       f"columns are {r.source.columns}")
    df, geom, crs = _read(r, bbox)
    if geom is None:
        raise KeyError(f"bld: no geometry column found in {r.source.description} (set columns.geometry)")
    heights = None
    if spec.get("height_table"):
        heights = open_table(spec["height_table"]).read(geometry=None)[0]
        heights = heights.rename({k: k.lower() for k in heights.columns})
    layer = prepare_buildings(Layer(df, geom, crs), storey_height_m, heights)
    log(f"bld: {r.source.description}: {len(layer):,} footprints")
    return layer


def load_table(role: str, spec: dict, log: Callable = print) -> pl.DataFrame:
    """An aspatial table with canonical column names."""
    r = resolve(role, spec)
    df, _, _ = _read(r)
    log(f"{role}: {r.source.description}: {df.height:,} rows")
    return df


def load_label_table(spec: dict, log: Callable = print) -> pl.DataFrame:
    """``labelled_asset`` -> ``address_pid, y, label_raw``."""
    r = resolve("labelled_asset", spec)
    for canon in ("address_pid", "label"):
        if canon not in r.mapping:
            raise KeyError(f"labelled_asset: no {canon} column (columns.{canon}) in {r.source.description}; "
                           f"columns are {r.source.columns}")
    df, _, _ = _read(r)
    labels = labels_from_frame(df, "address_pid", "label", spec.get("pd_values"), spec.get("oos_values"))
    counts = labels["y"].value_counts()
    n = {int(k): int(v) for k, v in counts.iter_rows()}
    log(f"labelled_asset: {r.source.description}: {labels.height:,} labelled addresses - {n.get(1, 0):,} PD, "
        f"{n.get(0, 0):,} out of scope, {n.get(-1, 0):,} unrecognised values")
    if n.get(-1, 0):
        unknown = labels.filter(c("y") == -1)["label_raw"].value_counts(sort=True).head(8)
        log(f"labelled_asset: unrecognised label values (treated as unlabelled): "
            f"{dict(unknown.iter_rows())}; set pd_values / oos_values to map them")
    return labels


def _gpkg_layers(path) -> list[str]:
    if not path or not str(path).lower().endswith(".gpkg") or not Path(str(path)).exists():
        return []
    import pyogrio

    return [str(row[0]) for row in pyogrio.list_layers(str(path))]


@dataclass
class Inputs:
    addresses: Layer | None = None
    parcels: Layer | None = None
    strata: Layer | None = None
    buildings: Layer | None = None
    building_address: pl.DataFrame | None = None
    building_cadastre: pl.DataFrame | None = None
    labels: pl.DataFrame | None = None
    counts: pl.DataFrame | None = None

    def pipeline_args(self) -> tuple:
        return self.addresses, self.parcels, self.buildings, self.building_address, self.building_cadastre


def load_inputs(datasets: dict, cfg: dict | None = None, labels: bool = False, bbox=None,
                log: Callable = print) -> Inputs:
    """Load every dataset with a path. ``ds_asset`` and ``cad`` are required; ``labelled_asset`` when ``labels``."""
    from .config import load_config

    cfg = cfg or load_config()
    for role in ("ds_asset", "cad") + (("labelled_asset",) if labels else ()):
        if not datasets.get(role, {}).get("path"):
            raise ValueError(f"no path for {role} ({DESCRIPTIONS[role]}); set it in the datasets YAML or with "
                             f"--{role.replace('_', '-')}")
    out = Inputs()
    out.addresses = load_addresses(datasets["ds_asset"], bbox, log)
    out.parcels, out.strata = load_parcels(datasets["cad"], bbox, log)
    bld = datasets.get("bld", {})
    if bld.get("path"):
        out.buildings = load_building_layer(bld, float(cfg["spatial"]["storey_height_m"]), bbox, log)
        layers = {lay.lower() for lay in _gpkg_layers(bld["path"])}
        for role in ("bld_address", "bld_cadastre"):
            spec = datasets.get(role, {})
            if not spec.get("path") and str(spec.get("table", "")).lower() in layers:
                datasets[role] = {**spec, "path": bld["path"]}
    else:
        log("bld: no buildings given - building features will be empty (the model still runs, less accurately)")
    if datasets.get("bld_address", {}).get("path"):
        out.building_address = canonical_links(load_table("bld_address", datasets["bld_address"], log))
    if datasets.get("bld_cadastre", {}).get("path"):
        out.building_cadastre = canonical_links(load_table("bld_cadastre", datasets["bld_cadastre"], log))
    if labels:
        out.labels = load_label_table(datasets["labelled_asset"], log)
    if datasets.get("parcel_counts", {}).get("path"):
        k = load_table("parcel_counts", datasets["parcel_counts"], log)
        if not {"parcel_pid", "n_dwellings"} <= set(k.columns):
            # no recognised names: take the first two columns (parcel id, count)
            k = open_table(datasets["parcel_counts"]["path"], datasets["parcel_counts"].get("table")).read(
                geometry=None)[0]
            k = k.select(k.columns[0], k.columns[1]).rename({k.columns[0]: "parcel_pid", k.columns[1]: "n_dwellings"})
        out.counts = k.select(c("parcel_pid").cast(pl.Utf8), c("n_dwellings").cast(pl.Int64, strict=False))
    return out


# ---------------------------------------------------------------------------------------------
# checking a datasets file against the data
# ---------------------------------------------------------------------------------------------

def _row_count(ts: TableSource) -> int | None:
    try:
        if ts.kind == "gdal":
            import pyogrio

            return int(pyogrio.read_info(ts.source, layer=ts.layer).get("features", -1))
        return int(ts._scan().select(pl.len()).collect().item())
    except Exception:
        return None


def check_datasets(datasets: dict) -> str:
    """Plain-text report: where each dataset resolved to, row counts, and the column mapping."""
    lines = []
    for role in ROLES:
        spec = datasets.get(role, {})
        title = f"{role} ({DESCRIPTIONS[role]})"
        if not spec.get("path"):
            lines += [f"{title}: no path set", ""]
            continue
        if role == "ds_asset" and (str(spec.get("format")).lower() == "gnaf" or is_gnaf_release(spec["path"])):
            lines += [f"{title}: G-NAF PSV release at {spec['path']}", ""]
            continue
        try:
            r = resolve(role, spec)
        except Exception as exc:  # report, don't raise: this is a diagnostic
            lines += [f"{title}: ERROR {exc}", ""]
            continue
        ts = r.source
        n = _row_count(ts)
        geom = ts.geometry_column or next((ts.find(g) for g in ("geometry", "geom", "wkb_geometry", "shape", "wkt")
                                           if ts.find(g)), None)
        lines.append(f"{title}: {ts.description}" + (f", {n:,} rows" if n is not None else ""))
        if role in GEOMETRY_ROLES or role == "ds_asset":
            crs = spec.get("crs") if spec.get("crs") not in (None, "auto") else (ts.crs or "unknown")
            lines.append(f"  geometry: {geom or 'none found'}; crs: {crs}")
        for canon, actual in r.mapping.items():
            lines.append(f"  {canon:<24} <- {actual}")
        for canon in r.missing:
            lines.append(f"  {canon:<24}    (not found)")
        for note in r.notes:
            lines.append(f"  note: {note}")
        unmapped = [col for col in ts.columns if col not in r.mapping.values() and col != geom]
        if unmapped:
            lines.append(f"  unused columns: {', '.join(unmapped[:30])}{' ...' if len(unmapped) > 30 else ''}")
        lines.append("")
    return "\n".join(lines)

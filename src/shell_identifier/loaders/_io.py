"""Read any table the pipeline accepts into polars (+ shapely geometry when it has one).

A dataset is located by ``path`` and an optional ``table`` name:

* ``path`` is a file (``.parquet``, ``.csv``, ``.psv``, ``.gpkg``, ``.shp`` ...): ``table``
  picks the layer inside a GeoPackage / FileGDB / SQLite and is ignored otherwise;
* ``path`` is a folder: ``table`` names the file (``<table>.parquet``, ``<table>.csv`` ...) or a
  sub-folder of parquet parts (``<table>/*.parquet``); a ``.gdb`` folder is a FileGDB;
* ``path`` is a glob or an object-store URI (``s3://...``, ``az://...``): read by polars directly.

Parquet / CSV / IPC are scanned lazily by polars with only the needed columns;
GDAL formats come through ``pyogrio.read_arrow``. Geometry columns may be WKB
(binary or hex) or WKT; GeoParquet and GDAL CRS metadata are honoured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import shapely

from ..geo import crs_string, decode_geometry, reproject

PARQUET_EXT = {".parquet", ".pq", ".geoparquet"}
CSV_EXT = {".csv": ",", ".psv": "|", ".tsv": "\t", ".txt": "|"}
IPC_EXT = {".ipc", ".arrow", ".feather"}
GDAL_EXT = {".gpkg", ".sqlite", ".db", ".shp", ".geojson", ".json", ".fgb", ".gml", ".tab", ".mif", ".kml",
            ".gdb", ".zip", ".geojsonseq"}
TABLE_EXT = list(PARQUET_EXT) + list(CSV_EXT) + list(IPC_EXT) + sorted(GDAL_EXT)
GEOMETRY_CANDIDATES = ["geometry", "geom", "wkb_geometry", "shape", "the_geom", "wkt", "geometry_wkt", "wkb",
                       "geometry_wkb", "geog"]


def _is_remote_or_glob(src: str) -> bool:
    return "://" in src or any(ch in src for ch in "*?[")


@dataclass
class TableSource:
    """A resolved table: where it is, how to read it, its columns and CRS metadata."""

    kind: str                      # parquet | csv | ipc | gdal
    source: str
    layer: str | None = None
    separator: str = ","
    columns: list[str] = field(default_factory=list)
    geometry_column: str | None = None   # GDAL geometry (read separately) or detected WKB/WKT column
    crs: str | None = None
    description: str = ""
    frame: pl.LazyFrame | None = None    # kind == "frame": an in-memory LazyFrame / DataFrame

    def find(self, name: str | None) -> str | None:
        """Actual column name matching ``name`` case-insensitively."""
        if not name:
            return None
        low = {col.lower(): col for col in self.columns}
        return low.get(str(name).lower())

    def read(self, columns: list[str] | None = None, geometry: str | None = "auto", where: str | None = None,
             bbox=None, bbox_crs="EPSG:4326") -> tuple[pl.DataFrame, np.ndarray | None, str | None]:
        """Read ``columns`` (actual names; None = all) and the geometry.

        ``geometry`` is a column name, "auto" (detect) or None (attributes only).
        ``where`` is an optional SQL predicate applied before anything else.
        Returns ``(frame, geometries or None, crs)``.
        """
        gcol = None
        if geometry and self.kind != "gdal":
            gcol = (self.find(geometry) if geometry != "auto" else None) or self.detect_geometry()
        if self.kind == "gdal":
            return self._read_gdal(columns, read_geometry=geometry is not None, where=where, bbox=bbox,
                                   bbox_crs=bbox_crs)
        cols = None if columns is None else list(dict.fromkeys([col for col in columns if col] + ([gcol] if gcol else [])))
        lf = self._scan()
        if where:
            lf = lf.filter(pl.sql_expr(where))
        df = (lf.select(cols) if cols is not None else lf).collect()
        geom = None
        if gcol and gcol in df.columns:
            geom = decode_geometry(df.get_column(gcol))
            df = df.drop(gcol)
        if geom is not None and bbox is not None:
            box = shapely.box(*bbox)
            if self.crs and crs_string(bbox_crs) != self.crs:
                box = reproject(np.array([box]), bbox_crs, self.crs)[0]
            keep = shapely.intersects(geom, box)
            df, geom = df.filter(pl.Series(keep)), geom[keep]
        return df, geom, self.crs

    def detect_geometry(self) -> str | None:
        """The geometry column: GDAL / GeoParquet metadata, a usual name, a name containing
        wkt / geom / shape, or else the first text column whose values look like WKT / hex WKB."""
        if self.geometry_column:
            return self.geometry_column
        named = next((self.find(g) for g in GEOMETRY_CANDIDATES if self.find(g)), None)
        if named:
            return named
        hinted = [col for col in self.columns if any(h in col.lower() for h in ("wkt", "geom", "shape", "wkb"))]
        if hinted:
            return hinted[0]
        schema = self._scan().collect_schema()
        text = [col for col, dt in schema.items() if dt in (pl.Utf8, pl.Binary)]
        if not text:
            return None
        sample = self._scan().select(text).head(20).collect()
        for col in text:
            vals = sample[col].drop_nulls()
            if vals.len() and vals.dtype == pl.Utf8 and vals.str.contains(_WKT_RX).all():
                return col
        return None

    def _scan(self) -> pl.LazyFrame:
        if self.kind == "frame":
            return self.frame
        if self.kind == "parquet":
            return pl.scan_parquet(self.source)
        if self.kind == "ipc":
            return pl.scan_ipc(self.source)
        lf = pl.scan_csv(self.source, separator=self.separator, infer_schema=False, null_values=[""],
                         encoding="utf8-lossy", truncate_ragged_lines=True)
        names = lf.collect_schema().names()
        clean = {n: n.lstrip("﻿").strip() for n in names if n != n.lstrip("﻿").strip()}
        return lf.rename(clean) if clean else lf

    def _read_gdal(self, columns, read_geometry: bool, where, bbox, bbox_crs):
        import pyogrio

        if bbox is not None and self.crs and crs_string(bbox_crs) != self.crs:
            box = reproject(np.array([shapely.box(*bbox)]), bbox_crs, self.crs)[0]
            bbox = tuple(shapely.bounds(box))
        meta, table = pyogrio.read_arrow(self.source, layer=self.layer, columns=columns, where=where,
                                         read_geometry=read_geometry, bbox=bbox)
        df = pl.from_arrow(table)
        if isinstance(df, pl.Series):
            df = df.to_frame()
        gname = meta.get("geometry_name") or "wkb_geometry"
        geom = None
        if read_geometry and gname in df.columns:
            geom = shapely.from_wkb(df.get_column(gname).to_numpy(), on_invalid="warn")
            df = df.drop(gname)
        crs = crs_string(meta.get("crs")) if meta.get("crs") else self.crs
        return df, geom, crs


_WKT_RX = r"(?i)^\s*(SRID=\d+;\s*)?(MULTI)?(POINT|POLYGON|LINESTRING|GEOMETRYCOLLECTION)\b|^0[01]0[1-7]0000"


def is_frame(obj) -> bool:
    return isinstance(obj, (pl.LazyFrame, pl.DataFrame))


def frame_source(frame: pl.LazyFrame | pl.DataFrame, name: str = "frame") -> TableSource:
    """Wrap a polars LazyFrame / DataFrame so it reads like any other table."""
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    ts = TableSource("frame", name, frame=lf, description=f"polars {type(frame).__name__} {name}")
    ts.columns = lf.collect_schema().names()
    return ts


def _geoparquet_crs(path: str) -> tuple[str | None, str | None]:
    """(geometry column, CRS) from GeoParquet metadata of a local file, if present."""
    try:
        import pyarrow.parquet as pq

        md = pq.read_schema(path).metadata or {}
        geo = md.get(b"geo")
        if not geo:
            return None, None
        info = json.loads(geo)
        col = info.get("primary_column")
        crs = info.get("columns", {}).get(col, {}).get("crs", "OGC:CRS84")
        from pyproj import CRS

        return col, (crs_string(CRS.from_json_dict(crs)) if isinstance(crs, dict) else crs_string(crs))
    except Exception:  # metadata is a convenience; never fail a read on it
        return None, None


def _from_file(src: str, layer: str | None) -> TableSource:
    p = Path(src)
    ext = p.suffix.lower()
    if ext in PARQUET_EXT or (_is_remote_or_glob(src) and ".parquet" in src.lower()):
        ts = TableSource("parquet", src, description=f"parquet {src}")
        ts.columns = pl.scan_parquet(src).collect_schema().names()
        if not _is_remote_or_glob(src):
            gcol, crs = _geoparquet_crs(src)
            ts.geometry_column, ts.crs = (ts.find(gcol) if gcol else None), crs
        return ts
    if ext in CSV_EXT:
        ts = TableSource("csv", src, separator=CSV_EXT[ext], description=f"delimited text {src}")
        ts.columns = ts._scan().collect_schema().names()
        return ts
    if ext in IPC_EXT:
        ts = TableSource("ipc", src, description=f"arrow ipc {src}")
        ts.columns = pl.scan_ipc(src).collect_schema().names()
        return ts
    if ext in GDAL_EXT or p.is_dir():
        import pyogrio

        layers = [str(row[0]) for row in pyogrio.list_layers(src)]
        chosen = None
        if layer:
            chosen = next((lay for lay in layers if lay.lower() == str(layer).lower()), None)
        if chosen is None:
            if len(layers) == 1 or (not layer and layers):
                chosen = layers[0]
            else:
                raise FileNotFoundError(f"layer/table {layer!r} not in {src}; it has {layers}")
        info = pyogrio.read_info(src, layer=chosen)
        ts = TableSource("gdal", src, layer=chosen, description=f"{info.get('driver', 'GDAL')} {src} layer {chosen}")
        ts.columns = [str(f) for f in info["fields"]]
        ts.geometry_column = info.get("geometry_name") or ("wkb_geometry" if info.get("geometry_type") else None)
        if not info.get("geometry_type"):
            ts.geometry_column = None
        ts.crs = crs_string(info["crs"]) if info.get("crs") else None
        return ts
    raise ValueError(f"don't know how to read {src} (extension {ext!r})")


def open_table(path, table: str | None = None) -> TableSource:
    """Resolve ``path`` (+ ``table``) to a readable :class:`TableSource`; ``path`` may be a polars frame."""
    if is_frame(path):
        return frame_source(path, table or "frame")
    src = str(path)
    if _is_remote_or_glob(src):
        return _from_file(src, table)
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"{src} does not exist")
    if p.is_dir() and p.suffix.lower() != ".gdb":
        if not table:
            raise ValueError(f"{src} is a folder; give the table (file) name to read from it")
        want = str(table).lower()
        entries = sorted(p.iterdir())
        for e in entries:
            if e.is_file() and e.stem.lower() == want and e.suffix.lower() in TABLE_EXT:
                return _from_file(str(e), table)
        for e in entries:
            if e.is_dir() and e.name.lower() == want:
                if e.suffix.lower() == ".gdb":
                    return _from_file(str(e), table)
                parts = sorted(e.rglob("*.parquet"))
                if parts:
                    ts = _from_file(str(parts[0]), table)
                    ts.source = str(e / "**" / "*.parquet")
                    ts.description = f"parquet dataset {e} ({len(parts)} parts)"
                    return ts
                csvs = sorted(e.rglob("*.csv"))
                if csvs:
                    ts = _from_file(str(csvs[0]), table)
                    ts.source = str(e / "**" / "*.csv")
                    return ts
        available = sorted({e.stem for e in entries if (e.is_file() and e.suffix.lower() in TABLE_EXT) or e.is_dir()})
        raise FileNotFoundError(f"no table {table!r} in folder {src}; found {available}")
    return _from_file(src, table)


def read_psv(path: str | Path, wanted: list[str] | None = None, sep: str = "|") -> pl.DataFrame:
    """Read a G-NAF style PSV as strings (empty -> null); ``wanted`` columns are matched case-insensitively."""
    ts = TableSource("csv", str(path), separator=sep)
    ts.columns = ts._scan().collect_schema().names()
    cols = None
    if wanted:
        cols = [col for col in (ts.find(w) for w in wanted) if col]
    df, _, _ = ts.read(cols, geometry=None)
    return df


def read_table(path: str | Path, layer: str | None = None) -> pl.DataFrame:
    """Read an aspatial table (attributes only) from any supported source."""
    df, _, _ = open_table(path, layer).read(geometry=None)
    return df


def rename_lower(df: pl.DataFrame, aliases: dict[str, str] | None = None) -> pl.DataFrame:
    """Lower-case column names and expand 10-character shapefile aliases."""
    aliases = {k.upper(): v for k, v in (aliases or {}).items()}
    mapping = {col: aliases.get(col.upper(), col.lower()) for col in df.columns}
    seen, final = set(), {}
    for old, new in mapping.items():
        if new in seen:          # two source columns collapsing to one name: keep the first
            continue
        seen.add(new)
        final[old] = new
    df = df.select(list(final))
    return df.rename({k: v for k, v in final.items() if k != v})

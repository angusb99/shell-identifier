"""Geometry next to polars: a frame plus an aligned shapely array, and the spatial helpers.

polars has no geometry type, so a :class:`Layer` holds attributes in a
``pl.DataFrame`` and geometries in a numpy array of shapely objects, row for
row. Point tables (addresses) skip the array and keep coordinates in ``x`` /
``y`` columns, which is far lighter for millions of rows. All spatial work is
vectorised shapely (STRtree) + pyproj; results come back as polars frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import polars as pl
import shapely
from pyproj import CRS, Transformer


@lru_cache(maxsize=32)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)


def crs_string(crs) -> str | None:
    """Normalise any CRS input to 'EPSG:xxxx' where possible (else WKT)."""
    if crs is None or crs == "":
        return None
    obj = CRS.from_user_input(crs)
    epsg = obj.to_epsg()
    return f"EPSG:{epsg}" if epsg else obj.to_wkt()


def same_crs(a, b) -> bool:
    if a is None or b is None:
        return True
    return CRS.from_user_input(a) == CRS.from_user_input(b)


def reproject_xy(x: np.ndarray, y: np.ndarray, src, dst) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if same_crs(src, dst):
        return x, y
    return _transformer(crs_string(src), crs_string(dst)).transform(x, y)


def reproject(geom: np.ndarray, src, dst) -> np.ndarray:
    if same_crs(src, dst) or len(geom) == 0:
        return geom
    tr = _transformer(crs_string(src), crs_string(dst))
    return shapely.transform(geom, lambda x, y: tr.transform(x, y), interleaved=False)


def decode_geometry(values: pl.Series) -> np.ndarray:
    """WKB (binary or hex text) or WKT text -> shapely array.

    GeoArrow extension columns (``geoarrow.wkb`` / ``geoarrow.wkt``, as written by
    geopandas and GDAL to GeoParquet) are unwrapped to their storage first.
    """
    if getattr(values.dtype, "is_extension", lambda: False)():
        values = values.ext.storage()
    if values.dtype == pl.Binary:
        return shapely.from_wkb(values.to_numpy(), on_invalid="warn")
    text = values.cast(pl.Utf8)
    sample = text.drop_nulls().head(50).to_list()
    if sample and all(s[:2] in ("00", "01") and all(ch in "0123456789abcdefABCDEF" for ch in s[:40]) for s in sample):
        return shapely.from_wkb(text.to_numpy(), on_invalid="warn")
    return shapely.from_wkt(text.to_numpy(), on_invalid="warn")


@dataclass
class Layer:
    """Attributes (polars) + aligned geometries (shapely), or a point table with ``x`` / ``y``."""

    df: pl.DataFrame
    geom: np.ndarray | None = None
    crs: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.geom is not None and len(self.geom) != self.df.height:
            raise ValueError(f"geometry length {len(self.geom)} != {self.df.height} rows")

    def __len__(self) -> int:
        return self.df.height

    @property
    def is_points(self) -> bool:
        return self.geom is None

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.df["x"].to_numpy().astype(float), self.df["y"].to_numpy().astype(float)

    def with_df(self, df: pl.DataFrame) -> "Layer":
        return replace(self, df=df)

    def filter(self, mask) -> "Layer":
        mask = np.asarray(mask, dtype=bool)
        return replace(self, df=self.df.filter(pl.Series(mask)),
                       geom=None if self.geom is None else self.geom[mask])

    def to_crs(self, crs) -> "Layer":
        crs = crs_string(crs)
        if same_crs(self.crs, crs):
            return replace(self, crs=crs or self.crs)
        if self.geom is None:
            x, y = reproject_xy(*self.xy(), self.crs, crs)
            return replace(self, df=self.df.with_columns(x=pl.Series(x), y=pl.Series(y)), crs=crs)
        return replace(self, geom=reproject(self.geom, self.crs, crs), crs=crs)

    def geometry(self) -> np.ndarray:
        """Shapely geometries (points are built from x / y on demand)."""
        if self.geom is not None:
            return self.geom
        return shapely.points(*self.xy())

    def bounds_filter(self, bbox, bbox_crs="EPSG:4326") -> "Layer":
        """Rows intersecting ``bbox`` (minx, miny, maxx, maxy) given in ``bbox_crs``."""
        box = shapely.box(*bbox)
        if not same_crs(self.crs, bbox_crs):
            box = reproject(np.array([box]), bbox_crs, self.crs)[0]
        if self.geom is None:
            x, y = self.xy()
            return self.filter(shapely.contains_xy(box, x, y))
        return self.filter(shapely.intersects(self.geom, box))

    def to_geopandas(self):
        import geopandas as gpd

        return gpd.GeoDataFrame(self.df.drop([c for c in ("x", "y") if self.geom is None and c in self.df.columns])
                                .to_pandas(), geometry=list(self.geometry()), crs=self.crs)

    @classmethod
    def from_geopandas(cls, gdf, points: bool = False) -> "Layer":
        crs = crs_string(gdf.crs) if gdf.crs is not None else None
        attrs = gdf.drop(columns=gdf.geometry.name)
        df = pl.from_pandas(attrs.reset_index(drop=True))
        if points:
            return cls(df.with_columns(x=pl.Series(gdf.geometry.x.to_numpy()), y=pl.Series(gdf.geometry.y.to_numpy())),
                       None, crs)
        return cls(df, np.asarray(gdf.geometry.values, dtype=object), crs)


def points_within(x: np.ndarray, y: np.ndarray, geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (point, polygon) for points strictly inside polygons."""
    if len(geoms) == 0 or len(x) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    tree = shapely.STRtree(geoms)
    i, j = tree.query(shapely.points(x, y), predicate="within")
    return i.astype(np.int64), j.astype(np.int64)


def nearest(x: np.ndarray, y: np.ndarray, geoms: np.ndarray, max_distance: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point, the nearest polygon within ``max_distance`` (one match, ties by index)."""
    if len(geoms) == 0 or len(x) == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, np.empty(0)
    tree = shapely.STRtree(geoms)
    (i, j), d = tree.query_nearest(shapely.points(x, y), max_distance=max_distance, return_distance=True,
                                   all_matches=True)
    order = np.lexsort((j, d, i))
    i, j, d = i[order], j[order], d[order]
    first = np.r_[True, i[1:] != i[:-1]] if len(i) else np.empty(0, dtype=bool)
    return i[first].astype(np.int64), j[first].astype(np.int64), d[first]


def dissolve_by(keys: np.ndarray, geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One geometry per distinct key (multi-record keys are unioned). Returns (keys, geoms)."""
    uniq, inv = np.unique(keys.astype(str), return_inverse=True)
    out = np.empty(len(uniq), dtype=object)
    counts = np.bincount(inv, minlength=len(uniq))
    single = counts[inv] == 1
    out[inv[single]] = geoms[single]
    multi = np.flatnonzero(~single)
    if len(multi):
        order = multi[np.argsort(inv[multi], kind="stable")]
        groups = np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)
        for g in groups:
            out[inv[g[0]]] = shapely.union_all(geoms[g])
    return uniq, out

"""Load free-text addresses (one string per row plus coordinates) via the rule-based parser."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

from ..parser import FIELDS, parse_address


def load_text_addresses(
    path: str | Path,
    address_col: str = "address",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    id_col: str | None = None,
    crs: str = "EPSG:7844",
    localities: Iterable[str] | None = None,
    sep: str = ",",
) -> gpd.GeoDataFrame:
    """Parse a CSV of address strings into the same shape as :func:`load_gnaf`.

    Any extra columns are kept. Recognised optional columns: ``alias_principal``
    (P/A), ``primary_secondary`` (P/S), ``legal_parcel_id`` and ``geocode_type``.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], sep=sep)
    loc = {l.upper() for l in localities} if localities else None
    parsed = pd.DataFrame([parse_address(t, loc) for t in raw[address_col].fillna("")], index=raw.index)
    parsed["parse_warnings"] = parsed["parse_warnings"].map("; ".join)

    keep = raw.drop(columns=[c for c in raw.columns if c in FIELDS], errors="ignore")
    df = pd.concat([keep, parsed], axis=1)
    df["address_pid"] = raw[id_col] if id_col else [f"TXT{i:09d}" for i in range(len(raw))]
    df["input_text"] = raw[address_col]
    for col in ("alias_principal", "primary_secondary", "legal_parcel_id", "geocode_type"):
        if col not in df:
            df[col] = ""
    df["n_ps_secondaries"] = 0
    lon = pd.to_numeric(df.pop(lon_col), errors="coerce")
    lat = pd.to_numeric(df.pop(lat_col), errors="coerce")
    ok = lon.notna() & lat.notna()
    gdf = gpd.GeoDataFrame(df[ok], geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs=crs)
    gdf["source"] = "text"
    gdf.attrs["dropped_without_geocode"] = int((~ok).sum())
    return gdf.reset_index(drop=True)

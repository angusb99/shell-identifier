"""Free-text addresses (one string per row plus coordinates) via the rule-based parser."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from ..frames import c
from ..geo import Layer, crs_string
from ..parser import FIELDS, parse_address
from ._io import open_table


def parse_column(df: pl.DataFrame, text_col: str, localities: Iterable[str] | None = None,
                 keep_existing: bool = False) -> pl.DataFrame:
    """Parse ``text_col`` into the G-NAF component columns (each distinct string parsed once).

    Existing component columns are replaced unless ``keep_existing``; ``input_text`` and
    ``parse_warnings`` are added.
    """
    loc = {str(v).upper() for v in localities} if localities else None
    texts = df.get_column(text_col).cast(pl.Utf8).fill_null("")
    uniq = texts.unique().to_list()
    parsed = [parse_address(t, loc) for t in uniq]
    table = pl.DataFrame({
        "_text": uniq,
        **{f: [p[f] for p in parsed] for f in FIELDS},
        "parse_warnings": ["; ".join(p["parse_warnings"]) for p in parsed],
    }, schema={"_text": pl.Utf8, **{f: pl.Utf8 for f in FIELDS}, "parse_warnings": pl.Utf8})
    drop = [f for f in FIELDS + ["parse_warnings"] if f in df.columns and not (keep_existing and f in FIELDS)]
    base = df.drop(drop).with_columns(_text=texts, input_text=texts)
    if keep_existing:
        table = table.drop([f for f in FIELDS if f in base.columns])
    return base.join(table, on="_text", how="left", maintain_order="left").drop("_text")


def load_text_addresses(
    path: str | Path,
    address_col: str = "address",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    id_col: str | None = None,
    crs: str = "EPSG:7844",
    localities: Iterable[str] | None = None,
    sep: str | None = None,
) -> Layer:
    """Parse a table of address strings into the same shape as :func:`load_gnaf`.

    Any extra columns are kept. Recognised optional columns: ``alias_principal``
    (P/A), ``primary_secondary`` (P/S), ``legal_parcel_id`` and ``geocode_type``.
    """
    ts = open_table(path)
    if sep:
        ts.separator = sep
    raw, _, _ = ts.read(geometry=None)
    df = parse_column(raw, ts.find(address_col) or address_col, localities)
    pid = pl.Series([f"TXT{i:09d}" for i in range(df.height)]) if not id_col else df.get_column(ts.find(id_col) or id_col)
    df = df.with_columns(
        address_pid=pid.cast(pl.Utf8),
        x=c(ts.find(lon_col) or lon_col).cast(pl.Float64, strict=False),
        y=c(ts.find(lat_col) or lat_col).cast(pl.Float64, strict=False),
        source=pl.lit("text"),
    )
    for col in ("alias_principal", "primary_secondary", "legal_parcel_id", "geocode_type"):
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    df = df.with_columns(n_ps_secondaries=pl.lit(0))
    ok = c("x").is_not_null() & c("y").is_not_null()
    dropped = df.filter(~ok).height
    return Layer(df.filter(ok), None, crs_string(crs), meta={"dropped_without_geocode": dropped})

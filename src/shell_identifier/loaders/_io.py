"""Small shared readers for pipe-separated and GeoPackage tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_psv(path: str | Path, wanted: list[str] | None = None, sep: str = "|") -> pd.DataFrame:
    """Read a G-NAF style PSV as strings; empty fields become NaN."""
    usecols = (lambda c: c.upper() in {w.upper() for w in wanted}) if wanted else None
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=[""],
                       usecols=usecols, encoding="utf-8-sig")


def read_table(path: str | Path, layer: str | None = None) -> pd.DataFrame:
    """Read an aspatial table from a GeoPackage/FGDB layer, or from a PSV/CSV file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".psv", ".txt"):
        return read_psv(path)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    import pyogrio

    return pyogrio.read_dataframe(path, layer=layer, read_geometry=False)


def lower_columns(df: pd.DataFrame, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Lower-case column names and expand 10-character shapefile aliases."""
    aliases = {k.upper(): v for k, v in (aliases or {}).items()}
    mapping = {}
    for col in df.columns:
        if col == "geometry":
            continue
        mapping[col] = aliases.get(col.upper(), col.lower())
    return df.rename(columns=mapping)

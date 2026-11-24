"""Address-level dwelling labels and parcel dwelling counts (polars)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..frames import c

PD_VALUES = {"PD", "1", "Y", "YES", "TRUE", "T", "PRIVATE DWELLING", "DWELLING"}


def pd_flag(col: str, pd_values=None) -> pl.Expr:
    """True where ``col`` holds a PD value (case-insensitive; booleans, 1 / 1.0 and Y / YES / TRUE work)."""
    pd_set = sorted({str(v).strip().upper() for v in (pd_values or PD_VALUES)})
    v = c(col).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_uppercase()
    v = pl.when(v.str.contains(r"^\d+\.0+$")).then(v.str.replace(r"\.0+$", "")).otherwise(v)
    return v.is_in(pd_set)


def labels_from_frame(raw: pl.DataFrame, id_col: str = "address_pid", label_col: str | None = "label",
                      pd_values=None) -> pl.DataFrame:
    """The PD addresses: ``address_pid, y (= 1), label_raw``, one row per address.

    Only rows whose ``label_col`` is a PD value are kept; every other address
    (a false / blank / other flag, or not in the table at all) is taken as not a
    private dwelling. With ``label_col=None`` every row is a PD.
    """
    cols = [id_col] + ([label_col] if label_col else [])
    missing = [col for col in cols if col not in raw.columns]
    if missing:
        raise KeyError(f"label table is missing column(s) {missing}; columns are {raw.columns}")
    keep = pd_flag(label_col, pd_values) if label_col else pl.lit(True)
    out = raw.filter(keep).select(
        address_pid=c(id_col).cast(pl.Utf8).str.strip_chars(),
        y=pl.lit(1, dtype=pl.Int64),
        label_raw=c(label_col).cast(pl.Utf8).fill_null("") if label_col else pl.lit("PD"),
    ).filter(c("address_pid").is_not_null() & (c("address_pid") != ""))
    return out.unique("address_pid", keep="first", maintain_order=True)


def load_labels(path: str | Path, id_col: str = "address_detail_pid", label_col: str | None = "label",
                pd_values=None, table: str | None = None) -> pl.DataFrame:
    """Read a label table (CSV, parquet, GeoPackage table ...) via :func:`labels_from_frame`."""
    from ..loaders import open_table

    ts = open_table(path, table)
    id_col = ts.find(id_col) or id_col
    label_col = (ts.find(label_col) or label_col) if label_col else None
    raw, _, _ = ts.read([col for col in (id_col, label_col) if col and col in ts.columns], geometry=None)
    return labels_from_frame(raw, id_col, label_col, pd_values)


def parcel_counts(s: pl.DataFrame, override: pl.DataFrame | None = None) -> pl.DataFrame:
    """Dwellings per parcel: the number of PD labels on it (or ``override`` where given).

    ``s`` needs ``parcel_pid`` and ``y`` (1 PD, 0 not PD, -1 unknown). Parcels whose
    addresses are all unknown are left out. ``override`` is an optional frame
    ``parcel_pid, n_dwellings`` of trusted counts. Returns ``parcel_pid, k``.
    """
    k = (s.filter((c("parcel_pid") != "") & (c("y") >= 0)).group_by("parcel_pid", maintain_order=True)
         .agg(c("y").sum().cast(pl.Int64).alias("k")))
    if override is not None and override.height:
        o = override.select(c(override.columns[0]).cast(pl.Utf8).alias("parcel_pid"),
                            c(override.columns[1]).cast(pl.Int64).alias("k"))
        k = pl.concat([k.filter(~c("parcel_pid").is_in(o["parcel_pid"].implode())), o])
    return k

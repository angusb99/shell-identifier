"""Address-level dwelling labels and parcel dwelling counts (polars)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..frames import c

PD_VALUES = {"PD", "1", "Y", "YES", "TRUE", "T", "PRIVATE DWELLING", "DWELLING"}
OOS_VALUES = {"OOS", "0", "N", "NO", "FALSE", "F", "OUT OF SCOPE", "OUT_OF_SCOPE", "NOT PD", "NPD", "NON PD"}


def labels_from_frame(raw: pl.DataFrame, id_col: str = "address_pid", label_col: str = "label",
                      pd_values=None, oos_values=None) -> pl.DataFrame:
    """Return ``address_pid, y, label_raw`` with y = 1 (private dwelling), 0 (out of scope), -1 (unrecognised).

    Values are matched case-insensitively against ``pd_values`` / ``oos_values``
    (defaults cover PD/OOS, 1/0, Y/N, TRUE/FALSE). Numeric and boolean label
    columns work too (1 / True = PD).
    """
    missing = [col for col in (id_col, label_col) if col not in raw.columns]
    if missing:
        raise KeyError(f"label table is missing column(s) {missing}; columns are {raw.columns}")
    pd_set = sorted({str(v).strip().upper() for v in (pd_values or PD_VALUES)})
    oos_set = sorted({str(v).strip().upper() for v in (oos_values or OOS_VALUES)})
    raw_label = c(label_col).cast(pl.Utf8)
    v = raw_label.fill_null("").str.strip_chars().str.to_uppercase()
    v = pl.when(v.str.contains(r"^\d+\.0+$")).then(v.str.replace(r"\.0+$", "")).otherwise(v)
    out = raw.select(
        address_pid=c(id_col).cast(pl.Utf8).str.strip_chars(),
        y=pl.when(v.is_in(pd_set)).then(1).when(v.is_in(oos_set)).then(0).otherwise(-1).cast(pl.Int64),
        label_raw=raw_label.fill_null(""),
    ).filter(c("address_pid").is_not_null() & (c("address_pid") != ""))
    return out.unique("address_pid", keep="last", maintain_order=True)


def load_labels(path: str | Path, id_col: str = "address_detail_pid", label_col: str = "label",
                pd_values=None, oos_values=None, table: str | None = None) -> pl.DataFrame:
    """Read a label table (CSV, parquet, GeoPackage table ...) via :func:`labels_from_frame`."""
    from ..loaders import open_table

    ts = open_table(path, table)
    cols = [ts.find(id_col) or id_col, ts.find(label_col) or label_col]
    raw, _, _ = ts.read([col for col in cols if col in ts.columns], geometry=None)
    return labels_from_frame(raw, cols[0], cols[1], pd_values, oos_values)


def parcel_counts(s: pl.DataFrame, override: pl.DataFrame | None = None) -> pl.DataFrame:
    """Dwellings per labelled parcel: the number of PD labels on it (or ``override`` where given).

    ``s`` needs ``parcel_pid`` and ``y``. A parcel is *labelled* when at least one of
    its addresses carries a recognised label. ``override`` is an optional frame
    ``parcel_pid, n_dwellings`` of trusted counts. Returns ``parcel_pid, k``.
    """
    k = (s.filter((c("parcel_pid") != "") & (c("y") >= 0)).group_by("parcel_pid", maintain_order=True)
         .agg(c("y").sum().cast(pl.Int64).alias("k")))
    if override is not None and override.height:
        o = override.select(c(override.columns[0]).cast(pl.Utf8).alias("parcel_pid"),
                            c(override.columns[1]).cast(pl.Int64).alias("k"))
        k = pl.concat([k.filter(~c("parcel_pid").is_in(o["parcel_pid"].implode())), o])
    return k

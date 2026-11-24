"""Reading address-level dwelling labels and deriving parcel dwelling counts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PD_VALUES = {"PD", "1", "Y", "YES", "TRUE", "T", "PRIVATE DWELLING", "DWELLING"}
OOS_VALUES = {"OOS", "0", "N", "NO", "FALSE", "F", "OUT OF SCOPE", "OUT_OF_SCOPE", "NOT PD", "NPD", "NON PD"}


def load_labels(path: str | Path, id_col: str = "address_detail_pid", label_col: str = "label",
                pd_values: set[str] | None = None, oos_values: set[str] | None = None) -> pd.DataFrame:
    """Return ``address_pid, y`` with y = 1 (private dwelling), 0 (out of scope), -1 (unrecognised).

    Values are matched case-insensitively against ``pd_values`` / ``oos_values``
    (defaults cover PD/OOS, 1/0, Y/N, TRUE/FALSE).
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in (id_col, label_col) if c not in raw]
    if missing:
        raise KeyError(f"label file is missing column(s) {missing}; columns are {list(raw.columns)}")
    pd_set = {v.upper() for v in (pd_values or PD_VALUES)}
    oos_set = {v.upper() for v in (oos_values or OOS_VALUES)}
    v = raw[label_col].str.strip().str.upper()
    y = np.where(v.isin(pd_set), 1, np.where(v.isin(oos_set), 0, -1))
    out = pd.DataFrame({"address_pid": raw[id_col].str.strip(), "y": y, "label_raw": raw[label_col]})
    out.attrs["unrecognised"] = int((y == -1).sum())
    return out.drop_duplicates("address_pid", keep="last").reset_index(drop=True)


def parcel_counts(parcel_pid: pd.Series, y: pd.Series, override: pd.DataFrame | None = None) -> pd.Series:
    """Dwellings per labelled parcel: the number of PD labels on it (or ``override`` where given).

    A parcel is *labelled* when at least one of its addresses carries a label.
    ``override`` is an optional frame ``parcel_pid, n_dwellings`` of trusted counts.
    """
    df = pd.DataFrame({"parcel_pid": parcel_pid.to_numpy(), "y": y.to_numpy()})
    df = df[(df["parcel_pid"] != "") & (df["y"] >= 0)]
    k = df.groupby("parcel_pid")["y"].sum().astype(int)
    if override is not None and not override.empty:
        o = override.set_index(override.columns[0])[override.columns[1]].astype(int)
        k = pd.concat([k[~k.index.isin(o.index)], o])
    return k.rename("k")

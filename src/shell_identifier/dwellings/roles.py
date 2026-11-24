"""From a dwelling assignment to address roles: dwellings, shells and out-of-scope addresses.

A *shell* is a non-dwelling address that one or more chosen dwellings nest
under. Nesting follows ``parent_pid`` from the shell identifier (unit -> tower ->
precinct, range coverage), extended with two links it does not hold:

* suffix siblings: ``100A`` (and ``UNIT 1 100A``) nest under a bare ``100``
* off-parcel children: a sub-address whose geocode slipped onto a neighbouring
  parcel nests under the bare street address it belongs to

A non-dwelling above dwellings only counts as a shell when it has two or more
dwellings beneath it or scores as shell-like on the expert shell model; a shop
with one flat beside it (70 / 70A) is an occupancy, not a shell.

The top-most shell above a parcel's dwellings is the PRIMARY_SHELL; shells
nested inside it (towers, levels) are SUB_SHELLs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DWELLING, PRIMARY, SUB, OOS = "DWELLING", "PRIMARY_SHELL", "SUB_SHELL", "OUT_OF_SCOPE"
MAX_DEPTH = 6


def extended_parents(s: pd.DataFrame) -> pd.Series:
    """parent_pid, filled with suffix-base and cross-parcel street-address parents."""
    parent = s["parent_pid"].fillna("").astype(str).copy()
    principal = ~s["is_alias"]
    bare_base = s[principal & (s["mask"] == 0) & (s["num_suffix"] == "") & s["num_last"].isna() & (s["num_key"] != "")]
    keys = ["parcel_pid", "street_key", "base_num_key"]
    base_pid = bare_base.drop_duplicates(keys).set_index(keys)["address_pid"]
    need = (parent == "") & (s["num_suffix"] != "") & s["num_last"].isna()
    if need.any():
        idx = pd.MultiIndex.from_frame(s.loc[need, keys])
        parent.loc[need] = base_pid.reindex(idx).fillna("").to_numpy()

    # Sub-addresses on a different parcel from their street address (geocode slip).
    bare = s[principal & (s["mask"] == 0) & (s["num_key"] != "")]
    by_street = bare.drop_duplicates(["street_key", "num_key"], keep=False).set_index(["street_key", "num_key"])
    need = (parent == "") & (s["mask"] != 0) & (s["num_key"] != "")
    if need.any():
        idx = pd.MultiIndex.from_frame(s.loc[need, ["street_key", "num_key"]])
        cand = by_street["address_pid"].reindex(idx).fillna("").to_numpy()
        cand_parcel = by_street["parcel_pid"].reindex(idx).fillna("").to_numpy()
        ok = cand_parcel != s.loc[need, "parcel_pid"].to_numpy()
        parent.loc[need] = np.where(ok, cand, "")
    return parent


def assign_roles(s: pd.DataFrame, dwelling: np.ndarray, shell_threshold: float = 0.5) -> pd.DataFrame:
    """Role, number of dwellings beneath, and the primary shell of each address's chain."""
    s = s.reset_index(drop=True)
    pid = s["address_pid"].astype(str)
    parent = extended_parents(s)
    parent_of = pd.Series(parent.to_numpy(), index=pid.to_numpy())
    is_dw = pd.Series(np.asarray(dwelling, dtype=bool), index=pid.to_numpy())

    # Walk up from every dwelling, counting the dwellings beneath each ancestor.
    below = pd.Series(0, index=pid.to_numpy(), dtype=int)
    cur = parent_of[is_dw.to_numpy()]
    for _ in range(MAX_DEPTH):
        cur = cur[cur != ""]
        if cur.empty:
            break
        below = below.add(cur.value_counts().reindex(below.index, fill_value=0))
        cur = pd.Series(parent_of.reindex(cur.to_numpy()).fillna("").to_numpy())

    # A non-dwelling above dwellings is a shell when it looks like one: two or more dwellings
    # beneath it, or a shell-like address on the expert score. A shop with one flat beside or
    # above it (70 / 70A) is an occupancy in its own right, so it stays out of scope.
    shell_like = pd.Series(s["shell_probability"].fillna(0).to_numpy() >= shell_threshold, index=pid.to_numpy())
    container = (below > 0) & ~is_dw & ((below >= 2) | shell_like)
    # A container is primary when no ancestor of it is also a (non-dwelling) container.
    has_container_above = pd.Series(False, index=pid.to_numpy())
    cur = parent_of.copy()
    for _ in range(MAX_DEPTH):
        hit = cur.map(container).fillna(False).astype(bool) & (cur != "")
        has_container_above |= hit
        cur = cur.map(parent_of).fillna("")
        if (cur == "").all():
            break

    role = np.where(is_dw, DWELLING, np.where(container & ~has_container_above, PRIMARY,
                                              np.where(container, SUB, OOS)))
    out = pd.DataFrame({"address_pid": pid, "role": role, "dwellings_below": below.to_numpy(),
                        "extended_parent_pid": parent.to_numpy()})
    # Primary shell of each address: the first PRIMARY_SHELL met walking up its chain.
    is_primary = pd.Series(role == PRIMARY, index=pid.to_numpy())
    top = pd.Series(np.where(is_primary, pid, ""), index=pid.to_numpy(), dtype=object)
    cur = parent_of.copy()
    for _ in range(MAX_DEPTH):
        hit = cur.map(is_primary).fillna(False).astype(bool) & (cur != "")
        top = top.where((top != "") | ~hit, cur)
        cur = cur.map(parent_of).fillna("")
    out["primary_shell_pid"] = top.to_numpy()
    return out


def pair_moves(df: pd.DataFrame) -> pd.DataFrame:
    """Pair, within each parcel, PD labels the model removed with addresses it promoted.

    Expects ``parcel_pid, address_pid, address_label, y, assigned, posterior``.
    Returns one row per move: from (labelled PD, not assigned) -> to (assigned, not labelled PD).
    """
    removed = df[(df["y"] == 1) & ~df["assigned"]].sort_values("posterior")
    added = df[(df["y"] != 1) & df["assigned"]].sort_values("posterior", ascending=False)
    removed = removed.assign(_n=removed.groupby("parcel_pid").cumcount())
    added = added.assign(_n=added.groupby("parcel_pid").cumcount())
    m = removed.merge(added, on=["parcel_pid", "_n"], how="outer", suffixes=("_from", "_to"))
    return pd.DataFrame({
        "parcel_pid": m["parcel_pid"],
        "from_pid": m["address_pid_from"], "from_label": m["address_label_from"],
        "from_posterior": m["posterior_from"].round(3),
        "to_pid": m["address_pid_to"], "to_label": m["address_label_to"],
        "to_posterior": m["posterior_to"].round(3),
        "confidence": (m["posterior_to"].fillna(1) - m["posterior_from"].fillna(0)).round(3),
    }).sort_values("confidence", ascending=False)

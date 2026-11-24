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
import polars as pl

from ..frames import c, left_join

DWELLING, PRIMARY, SUB, OOS = "DWELLING", "PRIMARY_SHELL", "SUB_SHELL", "OUT_OF_SCOPE"
MAX_DEPTH = 6


def extended_parents(s: pl.DataFrame) -> pl.Series:
    """parent_pid, filled with suffix-base and cross-parcel street-address parents."""
    keys = ["parcel_pid", "street_key", "base_num_key"]
    t = s.select("address_pid", "parent_pid", "is_alias", "mask", "num_suffix", "num_last", "num_key",
                 "parcel_pid", "street_key", "base_num_key").with_columns(
        parent=c("parent_pid").cast(pl.Utf8).fill_null(""))
    principal = ~c("is_alias")
    bare_base = (t.filter(principal & (c("mask") == 0) & (c("num_suffix") == "") & c("num_last").is_null()
                          & (c("num_key") != ""))
                 .unique(keys, keep="first", maintain_order=True).select(*keys, c("address_pid").alias("_base")))
    t = left_join(t, bare_base, keys)
    need = (c("parent") == "") & (c("num_suffix") != "") & c("num_last").is_null()
    t = t.with_columns(parent=pl.when(need).then(c("_base").fill_null("")).otherwise(c("parent")))

    # Sub-addresses on a different parcel from their street address (geocode slip).
    bare = t.filter(principal & (c("mask") == 0) & (c("num_key") != ""))
    by_street = (bare.filter(pl.len().over(["street_key", "num_key"]) == 1)
                 .select("street_key", "num_key", c("address_pid").alias("_cand"), c("parcel_pid").alias("_cand_parcel")))
    t = left_join(t, by_street, ["street_key", "num_key"])
    need = (c("parent") == "") & (c("mask") != 0) & (c("num_key") != "")
    ok = c("_cand_parcel").fill_null("") != c("parcel_pid")
    t = t.with_columns(parent=pl.when(need).then(pl.when(ok).then(c("_cand").fill_null("")).otherwise(pl.lit("")))
                       .otherwise(c("parent")))
    return t.get_column("parent")


class _Chain:
    """Vectorised walks up a pid -> parent table."""

    def __init__(self, pid: pl.Series, parent: pl.Series):
        self.pid, self.parent = pid, parent
        self.pos = pl.Series(np.arange(len(pid), dtype=np.int64))

    def up(self, cur: pl.Series) -> pl.Series:
        return cur.replace_strict(self.pid, self.parent, default="", return_dtype=pl.Utf8)

    def index(self, cur: pl.Series) -> np.ndarray:
        return cur.replace_strict(self.pid, self.pos, default=-1, return_dtype=pl.Int64).to_numpy()


def assign_roles(s: pl.DataFrame, dwelling: np.ndarray, shell_threshold: float = 0.5) -> pl.DataFrame:
    """Role, number of dwellings beneath, and the primary shell of each address's chain."""
    n = s.height
    pid = s.get_column("address_pid").cast(pl.Utf8)
    parent = extended_parents(s)
    chain = _Chain(pid, parent)
    is_dw = np.asarray(dwelling, dtype=bool)

    # Walk up from every dwelling, counting the dwellings beneath each ancestor.
    below = np.zeros(n, dtype=np.int64)
    cur = parent.filter(pl.Series(is_dw))
    for _ in range(MAX_DEPTH):
        cur = cur.filter(cur != "")
        if cur.is_empty():
            break
        ix = chain.index(cur)
        below += np.bincount(ix[ix >= 0], minlength=n)
        cur = chain.up(cur)

    # A non-dwelling above dwellings is a shell when it looks like one: two or more dwellings
    # beneath it, or a shell-like address on the expert score. A shop with one flat beside or
    # above it (70 / 70A) is an occupancy in its own right, so it stays out of scope.
    shell_like = s.get_column("shell_probability").fill_null(0).to_numpy() >= shell_threshold
    container = (below > 0) & ~is_dw & ((below >= 2) | shell_like)

    # Walk up from every address: is any ancestor a container, and which primary shell is on top?
    has_container_above = np.zeros(n, dtype=bool)
    ancestors = []
    cur = parent
    for _ in range(MAX_DEPTH):
        ix = chain.index(cur)
        if (ix < 0).all():
            break
        ancestors.append(ix)
        has_container_above |= (ix >= 0) & container[np.clip(ix, 0, None)]
        cur = chain.up(cur)

    is_primary = container & ~has_container_above
    role = np.where(is_dw, DWELLING, np.where(is_primary, PRIMARY, np.where(container, SUB, OOS)))
    top = np.where(is_primary, pid.to_numpy(), "").astype(object)
    pid_np = pid.to_numpy()
    for ix in ancestors:
        hit = (ix >= 0) & is_primary[np.clip(ix, 0, None)] & (top == "")
        top[hit] = pid_np[ix[hit]]
    return pl.DataFrame({"address_pid": pid, "role": role, "dwellings_below": below,
                         "extended_parent_pid": parent, "primary_shell_pid": pl.Series(top, dtype=pl.Utf8)})


def pair_moves(df: pl.DataFrame) -> pl.DataFrame:
    """Pair, within each parcel, PD labels the model removed with addresses it promoted.

    Expects ``parcel_pid, address_pid, address_label, y, assigned, posterior``.
    Returns one row per move: from (labelled PD, not assigned) -> to (assigned, not labelled PD).
    """
    cols = ["parcel_pid", "address_pid", "address_label", "posterior"]
    removed = (df.filter((c("y") == 1) & ~c("assigned")).sort("posterior", maintain_order=True)
               .with_columns(_n=pl.int_range(pl.len()).over("parcel_pid")).select(cols + ["_n"]))
    added = (df.filter((c("y") != 1) & c("assigned")).sort("posterior", descending=True, maintain_order=True)
             .with_columns(_n=pl.int_range(pl.len()).over("parcel_pid")).select(cols + ["_n"]))
    m = removed.join(added, on=["parcel_pid", "_n"], how="full", suffix="_to", coalesce=True)
    return m.select(
        "parcel_pid",
        from_pid=c("address_pid"), from_label=c("address_label"), from_posterior=c("posterior").round(3),
        to_pid=c("address_pid_to"), to_label=c("address_label_to"), to_posterior=c("posterior_to").round(3),
        confidence=(c("posterior_to").fill_null(1) - c("posterior").fill_null(0)).round(3),
    ).sort("confidence", descending=True, maintain_order=True)

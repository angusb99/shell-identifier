"""Small polars helpers shared by the pipeline."""

from __future__ import annotations

from typing import Callable, Iterable

import polars as pl

c = pl.col
EMPTY_TOKENS = ["NAN", "NONE", "<NA>", "NULL"]


def clean_str(col: str) -> pl.Expr:
    """Upper-case, stripped string; nulls and textual nulls ('nan', 'None', 'NULL') become ''."""
    e = c(col).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_uppercase()
    return pl.when(e.is_in(EMPTY_TOKENS)).then(pl.lit("")).otherwise(e).alias(col)


def canon_num(col: str) -> pl.Expr:
    """'01' -> '1', '12.0' -> '12'; keeps decimals like '3.04'."""
    return c(col).str.replace(r"\.0+$", "").str.replace(r"^0+(\d)", "${1}").alias(col)


def join_nonempty(exprs: Iterable[pl.Expr], sep: str) -> pl.Expr:
    """Join string expressions with ``sep``, skipping empty and null parts."""
    parts = [pl.when(e != "").then(e) for e in exprs]
    return pl.concat_str(parts, separator=sep, ignore_nulls=True).fill_null("")


def map_values(df: pl.DataFrame, col: str, fn: Callable, dtype=pl.Utf8, default=None) -> pl.Expr:
    """Apply a Python function to the distinct values of ``col`` and map them back.

    Authority lookups and regexes polars cannot express (look-arounds) run once per
    distinct value instead of once per row.
    """
    keys = df.get_column(col).unique().drop_nulls()
    values = pl.Series([fn(k) for k in keys.to_list()], dtype=dtype)
    return c(col).replace_strict(keys, values, default=default, return_dtype=dtype)


def lookup(expr: pl.Expr, keys: pl.Series, values: pl.Series, default=None) -> pl.Expr:
    """Map ``expr`` through a key -> value table (keys must be unique)."""
    return expr.replace_strict(keys, values, default=default, return_dtype=values.dtype)


def left_join(left: pl.DataFrame, right: pl.DataFrame, on, **kw) -> pl.DataFrame:
    """Left join that keeps the left frame's row order."""
    return left.join(right, on=on, how="left", maintain_order="left", **kw)


def num(col: str) -> pl.Expr:
    """Column as Float64 (unparseable -> null)."""
    return c(col).cast(pl.Float64, strict=False)


def int_text(expr: pl.Expr, missing: str = "?") -> pl.Expr:
    """Rounded integer as text, for explanation strings."""
    return expr.round(0).cast(pl.Int64, strict=False).cast(pl.Utf8).fill_null(missing)


def ensure_columns(df: pl.DataFrame, defaults: dict) -> pl.DataFrame:
    """Add any missing columns with a literal default."""
    missing = {k: v for k, v in defaults.items() if k not in df.columns}
    if not missing:
        return df
    return df.with_columns([pl.lit(v).alias(k) for k, v in missing.items()])


def markdown_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)

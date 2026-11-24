"""Canonical address components used to decide which addresses nest inside which.

Every address is reduced to:

* ``street_key``  - street name/type/suffix + locality + state
* ``num_key``     - the full street number, e.g. ``"25A"`` or ``"1-15"``
* ``base_num_key``- the number without its suffix (``"25"``) for suffix siblings
* three optional sub-address components, from least to most specific:
    ``comp_structure`` (TOWER A / BUILDING 2), ``comp_level`` (L:3), ``comp_flat`` (UNIT:5)
* ``vanity_name`` - a building name that labels the whole site ("THE ARCADIA")

Address B nests inside address A when they share street and number, every
component A has is equal in B, and B has at least one component A lacks.

All columns are polars expressions; authority lookups and the building-name
regex run once per distinct value.
"""

from __future__ import annotations

import re
from functools import lru_cache

import polars as pl

from .authority import (
    FLAT_TYPES,
    LEVEL_TYPES,
    flat_category,
    normalise_flat_type,
    normalise_level_type,
    normalise_street_suffix,
    normalise_street_type,
)
from .frames import c, canon_num, clean_str, join_nonempty, map_values
from .parser import FIELDS

MASK_BITS = {"comp_structure": 4, "comp_level": 2, "comp_flat": 1}
COMP_COLS = list(MASK_BITS)
NUMBER_FIELDS = ("flat_number", "level_number", "number_first", "number_last", "lot_number")

_STRUCT_WORDS = r"TOWER|TWR|BUILDING|BLDG|BLD|BLOCK|BLCK|BLK|WING"
_STRUCT_CANON = {"TOWER": "TOWER", "TWR": "TOWER", "BUILDING": "BUILDING", "BLDG": "BUILDING",
                 "BLD": "BUILDING", "BLOCK": "BLOCK", "BLCK": "BLOCK", "BLK": "BLOCK", "WING": "WING"}
_STRUCT_RE = re.compile(
    rf"\b(?:(?P<w1>{_STRUCT_WORDS})\s+(?!(?:THE|AND|OF|ON|AT)\b)(?P<id1>[A-Z0-9]{{1,4}})"
    rf"|(?P<id2>NORTH|SOUTH|EAST|WEST|CENTRAL|UPPER|LOWER|MAIN|[A-Z0-9]{{1,2}})\s+(?P<w2>{_STRUCT_WORDS}))\b"
)


@lru_cache(maxsize=200_000)
def split_building_name(name: str) -> tuple[str, str]:
    """Split a building name into (structural id, vanity name).

    >>> split_building_name("THE QUAY TOWER A")
    ('TOWER A', 'THE QUAY')
    >>> split_building_name("NORTH TOWER")
    ('TOWER NORTH', '')
    """
    name = re.sub(r"\s+", " ", (name or "").upper()).strip(" ,")
    if not name:
        return "", ""
    m = _STRUCT_RE.search(name)
    if not m:
        return "", name
    if m["w1"]:
        struct = f"{_STRUCT_CANON[m['w1']]} {m['id1']}"
    else:
        struct = f"{_STRUCT_CANON[m['w2']]} {m['id2']}"
    vanity = (name[: m.start()] + " " + name[m.end():]).strip(" ,-")
    return struct, re.sub(r"\s+", " ", vanity)


def _part(prefix: str) -> pl.Expr:
    """prefix + number + suffix of a G-NAF number group (e.g. flat_number_prefix, flat_number, flat_number_suffix)."""
    return c(f"{prefix}_prefix") + c(prefix) + c(f"{prefix}_suffix")


def _or_question(col: str) -> pl.Expr:
    return pl.when(c(col) == "").then(pl.lit("?")).otherwise(c(col))


def build_components(df: pl.DataFrame) -> pl.DataFrame:
    """Add canonical component columns to a frame of G-NAF style address fields."""
    out = df.with_columns([clean_str(col) if col in df.columns else pl.lit("").alias(col) for col in FIELDS])
    out = out.with_columns([canon_num(col) for col in NUMBER_FIELDS])
    out = out.with_columns(
        map_values(out, "flat_type_code", normalise_flat_type).alias("flat_type_code"),
        map_values(out, "level_type_code", normalise_level_type).alias("level_type_code"),
        map_values(out, "street_type_code", normalise_street_type).alias("street_type_code"),
        map_values(out, "street_suffix_code", normalise_street_suffix).alias("street_suffix_code"),
    )

    # --- street and number ------------------------------------------------
    nf = c("number_first").cast(pl.Float64, strict=False)
    nl = c("number_last").cast(pl.Float64, strict=False)
    has_range = nl.is_not_null() & nf.is_not_null() & (nl > nf)
    first, last, lot = _part("number_first"), _part("number_last"), _part("lot_number")
    num_key = (pl.when(nf.is_not_null()).then(pl.when(has_range).then(first + "-" + last).otherwise(first))
               .when(lot != "").then("LOT " + lot).otherwise(pl.lit("")))
    out = out.with_columns(
        street_key=join_nonempty([c("street_name"), c("street_type_code"), c("street_suffix_code")], " ")
        + "|" + c("locality_name") + "|" + c("state"),
        num_first=nf.cast(pl.Int64, strict=False),
        num_last=pl.when(has_range).then(nl).cast(pl.Int64, strict=False),
        num_key=num_key,
        num_suffix=pl.when(has_range).then(pl.lit("")).otherwise(c("number_first_suffix")),
        base_num_key=pl.when(nf.is_not_null() & ~has_range).then(c("number_first_prefix") + c("number_first"))
        .otherwise(num_key),
        is_lot_only=nf.is_null() & (lot != ""),
    )

    # --- sub-address components ---------------------------------------------
    names = out.get_column("building_name").unique()
    split = [split_building_name(n) for n in names.to_list()]
    struct_word = map_values(out, "flat_type_code",
                             lambda code: _STRUCT_CANON.get(FLAT_TYPES.get(code, ("",))[0], code))
    out = out.with_columns(
        _cat=map_values(out, "flat_type_code", flat_category),
        _struct_word=struct_word,
        _struct_name=c("building_name").replace_strict(names, pl.Series([s for s, _ in split], dtype=pl.Utf8)),
        vanity_name=c("building_name").replace_strict(names, pl.Series([v for _, v in split], dtype=pl.Utf8)),
        _flat_id=_part("flat_number"),
        _level_id=_part("level_number"),
    )
    has_flat = (c("flat_type_code") != "") | (c("_flat_id") != "")
    is_struct_flat = has_flat & (c("_cat") == "structure")
    has_level = (c("level_type_code") != "") | (c("_level_id") != "")
    out = out.with_columns(
        comp_structure=pl.when(is_struct_flat).then(c("_struct_word") + " " + c("_flat_id"))
        .otherwise(c("_struct_name")),
        comp_flat=pl.when(has_flat & ~is_struct_flat).then(_or_question("flat_type_code") + ":" + c("_flat_id"))
        .otherwise(pl.lit("")),
        comp_level=pl.when(has_level).then(_or_question("level_type_code") + ":" + c("_level_id"))
        .otherwise(pl.lit("")),
    ).with_columns(
        flat_category=pl.when(c("comp_flat") != "")
        .then(pl.when(c("_cat") == "").then(pl.lit("dwelling")).otherwise(c("_cat"))).otherwise(pl.lit("")),
    ).drop("_cat", "_struct_word", "_struct_name", "_flat_id", "_level_id")
    return finalise_components(out)


def finalise_components(df: pl.DataFrame) -> pl.DataFrame:
    """(Re)compute ``mask``, ``specificity`` and ``node_type`` from the comp columns."""
    has = {col: (c(col) != "").cast(pl.Int64) for col in COMP_COLS}
    return df.with_columns(
        mask=pl.sum_horizontal([has[col] * bit for col, bit in MASK_BITS.items()]),
        specificity=pl.sum_horizontal(list(has.values())),
        node_type=pl.when(c("comp_flat") != "").then(c("flat_category"))
        .when(c("comp_level") != "").then(pl.lit("level"))
        .when(c("comp_structure") != "").then(pl.lit("structure"))
        .otherwise(pl.lit("street")),
    )


def format_label(df: pl.DataFrame) -> pl.Series:
    """G-NAF style single-line label, e.g. ``UNIT 5, LEVEL 2, 25 SMITH STREET, TESTWOOD NSW 2999``."""
    flat_name = map_values(df, "flat_type_code", lambda code: FLAT_TYPES.get(code, (code,))[0] if code else "")
    level_name = map_values(df, "level_type_code", lambda code: LEVEL_TYPES.get(code, code) if code else "")
    first, last, lot = _part("number_first"), _part("number_last"), _part("lot_number")
    number = pl.when(last != "").then(first + "-" + last).otherwise(first)
    number = pl.when((number == "") & (lot != "")).then("LOT " + lot).otherwise(number)
    label = join_nonempty([
        c("building_name"),
        join_nonempty([flat_name, _part("flat_number")], " "),
        join_nonempty([level_name, _part("level_number")], " "),
        join_nonempty([number, c("street_name"), c("street_type_code"), c("street_suffix_code")], " "),
        join_nonempty([c("locality_name"), c("state"), c("postcode")], " "),
    ], ", ")
    return df.select(label.alias("address_label")).to_series()

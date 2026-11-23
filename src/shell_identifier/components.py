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
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .authority import (
    FLAT_TYPES,
    LEVEL_TYPES,
    flat_category,
    normalise_flat_type,
    normalise_level_type,
    normalise_street_suffix,
    normalise_street_type,
)
from .parser import FIELDS

MASK_BITS = {"comp_structure": 4, "comp_level": 2, "comp_flat": 1}
COMP_COLS = list(MASK_BITS)

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


def _clean(s: pd.Series) -> pd.Series:
    s = s.astype(object).where(s.notna(), "")
    s = s.astype(str).str.strip().str.upper()
    return s.replace({"NAN": "", "NONE": "", "<NA>": ""})


def _canon_num(s: pd.Series) -> pd.Series:
    """'01' -> '1', '12.0' -> '12'; keeps decimals like '3.04'."""
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s.str.replace(r"^0+(?=\d)", "", regex=True)


def _join(parts: list[pd.Series], sep: str) -> pd.Series:
    out = pd.Series("", index=parts[0].index, dtype=object)
    for p in parts:
        p = p.astype(object).where(p.notna(), "").astype(str).str.strip()
        both = (out != "") & (p != "")
        out = pd.Series(np.where(both, out + sep + p, np.where(out == "", p, out)), index=out.index, dtype=object)
    return out


def build_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical component columns to a frame of G-NAF style address fields."""
    out = df.copy()
    for col in FIELDS:
        out[col] = _clean(out[col]) if col in out else ""
    for col in ("flat_number", "level_number", "number_first", "number_last", "lot_number"):
        out[col] = _canon_num(out[col])

    out["flat_type_code"] = out["flat_type_code"].map(normalise_flat_type)
    out["level_type_code"] = out["level_type_code"].map(normalise_level_type)
    out["street_type_code"] = out["street_type_code"].map(normalise_street_type)
    out["street_suffix_code"] = out["street_suffix_code"].map(normalise_street_suffix)

    # --- street and number ------------------------------------------------
    out["street_key"] = _join([out["street_name"], out["street_type_code"], out["street_suffix_code"]], " ") \
        + "|" + out["locality_name"] + "|" + out["state"]
    nf = pd.to_numeric(out["number_first"], errors="coerce")
    nl = pd.to_numeric(out["number_last"], errors="coerce")
    has_range = nl.notna() & nf.notna() & (nl > nf)
    out["num_first"] = nf.astype("Int64")
    out["num_last"] = nl.where(has_range).astype("Int64")
    first = out["number_first_prefix"] + out["number_first"] + out["number_first_suffix"]
    last = out["number_last_prefix"] + out["number_last"] + out["number_last_suffix"]
    lot = out["lot_number_prefix"] + out["lot_number"] + out["lot_number_suffix"]
    out["num_key"] = np.where(
        nf.notna(),
        np.where(has_range, first + "-" + last, first),
        np.where(lot != "", "LOT " + lot, ""),
    )
    out["num_suffix"] = np.where(has_range, "", out["number_first_suffix"])
    out["base_num_key"] = np.where(nf.notna() & ~has_range, out["number_first_prefix"] + out["number_first"], out["num_key"])
    out["is_lot_only"] = nf.isna() & (lot != "")

    # --- sub-address components ---------------------------------------------
    cat = out["flat_type_code"].map(flat_category)
    flat_id = out["flat_number_prefix"] + out["flat_number"] + out["flat_number_suffix"]
    has_flat = (out["flat_type_code"] != "") | (flat_id != "")
    is_struct_flat = has_flat & (cat == "structure")
    struct_from_flat = np.where(
        is_struct_flat,
        out["flat_type_code"].map(lambda c: _STRUCT_CANON.get(FLAT_TYPES.get(c, ("",))[0], c)) + " " + flat_id,
        "",
    )
    split = out["building_name"].map(split_building_name)
    struct_from_name = split.map(lambda t: t[0])
    out["vanity_name"] = split.map(lambda t: t[1])
    out["comp_structure"] = np.where(struct_from_flat != "", struct_from_flat, struct_from_name)
    out["comp_flat"] = np.where(has_flat & ~is_struct_flat, out["flat_type_code"].replace("", "?") + ":" + flat_id, "")
    level_id = out["level_number_prefix"] + out["level_number"] + out["level_number_suffix"]
    has_level = (out["level_type_code"] != "") | (level_id != "")
    out["comp_level"] = np.where(has_level, out["level_type_code"].replace("", "?") + ":" + level_id, "")
    out["flat_category"] = np.where(out["comp_flat"] != "", cat.replace("", "dwelling"), "")
    finalise_components(out)
    return out


def finalise_components(out: pd.DataFrame) -> None:
    """(Re)compute ``mask``, ``specificity`` and ``node_type`` from the comp columns, in place."""
    out["mask"] = sum((out[c] != "").astype(int) * bit for c, bit in MASK_BITS.items())
    out["specificity"] = sum((out[c] != "").astype(int) for c in COMP_COLS)
    out["node_type"] = np.select(
        [out["comp_flat"] != "", out["comp_level"] != "", out["comp_structure"] != ""],
        [out["flat_category"], "level", "structure"],
        "street",
    )


def format_label(df: pd.DataFrame) -> pd.Series:
    """G-NAF style single-line label, e.g. ``UNIT 5, LEVEL 2, 25 SMITH STREET, TESTWOOD NSW 2999``."""
    flat_name = df["flat_type_code"].map(lambda c: FLAT_TYPES.get(c, (c,))[0] if c else "")
    flat = _join([flat_name, df["flat_number_prefix"] + df["flat_number"] + df["flat_number_suffix"]], " ")
    level_name = df["level_type_code"].map(lambda c: LEVEL_TYPES.get(c, c) if c else "")
    level = _join([level_name, df["level_number_prefix"] + df["level_number"] + df["level_number_suffix"]], " ")
    first = df["number_first_prefix"] + df["number_first"] + df["number_first_suffix"]
    last = df["number_last_prefix"] + df["number_last"] + df["number_last_suffix"]
    number = pd.Series(np.where(last != "", first + "-" + last, first), index=df.index)
    lot = df["lot_number_prefix"] + df["lot_number"] + df["lot_number_suffix"]
    number = pd.Series(np.where((number == "") & (lot != ""), "LOT " + lot, number), index=df.index)
    street = _join([number, df["street_name"], df["street_type_code"], df["street_suffix_code"]], " ")
    place = _join([df["locality_name"], df["state"], df["postcode"]], " ")
    return _join([df["building_name"], flat, level, street, place], ", ")

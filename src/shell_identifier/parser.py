"""Rule-based parser for Australian address strings.

No libpostal and no machine learning: the parser tokenises the string, peels the
postcode/state off the tail, finds the street-number "anchor", splits the street
from the locality using the G-NAF authority tables (optionally a locality
gazetteer), and reads everything before the anchor as sub-address components
(flat, level, building/tower, building name).

The result uses G-NAF ADDRESS_DETAIL column names (lower case) so parsed text
flows through exactly the same pipeline as G-NAF records.
"""

from __future__ import annotations

import re
from typing import Iterable

from .authority import (
    FLAT_TYPE_ALIASES,
    LEVEL_TYPE_ALIASES,
    LEVEL_TYPES_NUMBERLESS,
    STATES,
    STREET_SUFFIX_ABBREV_ALIASES,
    STREET_SUFFIX_WORD_ALIASES,
    STREET_TYPE_ALIASES,
    STRUCTURE_WORDS,
)

FIELDS = [
    "building_name",
    "lot_number_prefix", "lot_number", "lot_number_suffix",
    "flat_type_code", "flat_number_prefix", "flat_number", "flat_number_suffix",
    "level_type_code", "level_number_prefix", "level_number", "level_number_suffix",
    "number_first_prefix", "number_first", "number_first_suffix",
    "number_last_prefix", "number_last", "number_last_suffix",
    "street_name", "street_type_code", "street_suffix_code",
    "locality_name", "state", "postcode",
]

_NUM_RE = re.compile(
    r"^(?P<p1>[A-Z]{0,2}?)(?P<n1>\d{1,6})(?P<s1>[A-Z]{0,2})"
    r"(?:-(?P<p2>[A-Z]{0,2}?)(?P<n2>\d{1,6})(?P<s2>[A-Z]{0,2}))?$"
)
_ID_RE = re.compile(r"^(?P<pre>[A-Z]{0,3}?)(?P<num>\d{1,6}(?:\.\d{1,3})?)(?P<suf>[A-Z]{0,2})$")
_ALPHA_ID_RE = re.compile(r"^[A-Z]{1,2}$")
_ATTACHED_RE = re.compile(
    r"^(?P<kw>UNIT|U|APT|FLAT|SHOP|SUITE|STE|TH|VILLA|PH|CS|LEVEL|LVL|LEV|L)(?P<id>\d[\d.]*[A-Z]{0,2})$"
)
_ORDINAL_RE = re.compile(r"^(\d{1,3})(ST|ND|RD|TH)$")
_ORDINAL_WORDS = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5,
    "SIXTH": 6, "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10,
}
_DIRECTIONS = {"NORTH", "SOUTH", "EAST", "WEST", "CENTRAL", "UPPER", "LOWER", "MAIN"}
# Words that often start a locality and are also street types ("ST KILDA", "MT DRUITT").
_LOCALITY_LEADERS = {"ST", "MT", "MOUNT", "PORT", "POINT", "LAKE", "CAPE", "GLEN", "NORTH", "SOUTH",
                     "EAST", "WEST", "UPPER", "LOWER"}
# Flat types that are commonly written without a number.
_NUMBERLESS_FLATS = {"REAR", "CTGE", "BNGW", "PTHS", "LOFT"}
_LEVEL_ATTACHED = {"LEVEL", "LVL", "LEV", "L"}
_STATE_NAMES = {name: abbr for abbr, name in STATES.items()}


def _tokenise(text: str) -> list[str]:
    t = (text or "").upper()
    t = t.replace("–", "-").replace("—", "-").replace(" ", " ")
    t = re.sub(r"(?<=[0-9A-Z])\s*-\s*(?=[A-Z]?\d)", "-", t)      # "25 - 27" -> "25-27"
    t = re.sub(r"\s*/\s*", "/", t)                                 # "5 / 25"  -> "5/25"
    t = re.sub(r"\bNO\.?\s*(?=\d)", "", t)                         # "No. 25"  -> "25"
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)                      # drop dots not inside decimals
    t = re.sub(r"[#;:()\[\]\"]", " ", t)
    t = t.replace(",", " , ")
    return t.split()


def _is_number_token(tok: str) -> bool:
    if "/" in tok:
        return bool(_NUM_RE.match(tok.rsplit("/", 1)[1]))
    return bool(_NUM_RE.match(tok))


def _is_id_token(tok: str) -> bool:
    return bool(_ID_RE.match(tok) or _ALPHA_ID_RE.match(tok))


def _split_id(tok: str) -> tuple[str, str, str]:
    """Split a sub-address identifier into (prefix, number, suffix)."""
    m = _ID_RE.match(tok)
    if m:
        return m["pre"], m["num"], m["suf"]
    return "", "", tok


def _strip_tail(words: list[str], res: dict) -> list[str]:
    """Remove country, postcode and state from the end of the token list."""
    def drop_commas(ws):
        while ws and ws[-1] == ",":
            ws = ws[:-1]
        return ws

    words = drop_commas(words)
    if words and words[-1] in ("AUSTRALIA", "AUS", "AU"):
        words = drop_commas(words[:-1])
    for _ in range(2):  # postcode and state in either order
        if words and re.fullmatch(r"\d{4}", words[-1]) and not res["postcode"]:
            res["postcode"] = words[-1]
            words = drop_commas(words[:-1])
            continue
        if not res["state"]:
            if words and words[-1] in STATES:
                res["state"] = words[-1]
                words = drop_commas(words[:-1])
                continue
            for name, abbr in _STATE_NAMES.items():
                parts = name.split()
                if len(words) >= len(parts) and words[-len(parts):] == parts:
                    res["state"] = abbr
                    words = drop_commas(words[: -len(parts)])
                    break
    return words


def _find_anchor(tokens: list[str]) -> int | None:
    """Index of the street-number token: the last number followed by a word-only segment."""
    best = None
    for i, tok in enumerate(tokens):
        if not _is_number_token(tok):
            continue
        seg = []
        for t in tokens[i + 1:]:
            if t == ",":
                if seg:
                    break
                continue
            seg.append(t)
        if not seg or not seg[0][0].isalpha() or seg[0] in STRUCTURE_WORDS:
            continue
        if any(_is_number_token(t) for t in seg):
            continue
        best = i
    return best


def _split_street_locality(rem: list[str], localities: set[str] | None, expect_locality: bool):
    """Split tokens after the street number into (name, type, suffix, locality)."""
    words, comma_after = [], []
    for t in rem:
        if t == ",":
            if comma_after:
                comma_after[-1] = True
            continue
        words.append(t)
        comma_after.append(False)
    n = len(words)
    if n == 0:
        return [], "", "", []

    gaz_start = None
    if localities:
        for k in range(1, n + 1):
            if " ".join(words[k:]) in localities:
                gaz_start = k
                break

    def suffix_len(j: int) -> int:
        if j + 1 >= n:
            return 0
        w = words[j + 1]
        if w in STREET_SUFFIX_ABBREV_ALIASES and len(w) <= 3:
            return 1
        if w in STREET_SUFFIX_WORD_ALIASES and (j + 2 == n or comma_after[j + 1] or gaz_start == j + 2):
            return 1
        return 0

    best_j, best_score = None, float("-inf")
    for j in range(1, n):
        if words[j] not in STREET_TYPE_ALIASES:
            continue
        s_len = suffix_len(j)
        end = j + s_len
        loc_len = n - end - 1
        score = 0.0
        if comma_after[end]:
            score += 2.0
        if 1 <= loc_len <= 4:
            score += 1.0
        elif loc_len == 0:
            score -= 2.0 if expect_locality else 0.5
        nxt = words[j + 1] if j + 1 < n else None
        if nxt and s_len == 0 and nxt in STREET_TYPE_ALIASES and nxt not in _LOCALITY_LEADERS:
            score -= 1.0
        if gaz_start is not None and gaz_start == end + 1:
            score += 3.0
        if score > best_score:
            best_j, best_score = j, score

    if best_j is None:
        # No street type: street name runs to the first comma / gazetteer locality / one word.
        if gaz_start is not None and gaz_start >= 1:
            stop = gaz_start
        elif any(comma_after):
            stop = comma_after.index(True) + 1
        else:
            stop = 1
        return words[:stop], "", "", words[stop:]

    s_len = suffix_len(best_j)
    suffix = words[best_j + 1] if s_len else ""
    return words[:best_j], words[best_j], suffix, words[best_j + 1 + s_len:]


def _parse_subaddress(tokens: list[str], res: dict, warnings: list[str]) -> str | None:
    """Fill flat/level/building fields from the tokens before the street number.

    Returns a flat type that was named but not yet given a number (it may be
    completed by a slash form such as ``SHOP 5/25``).
    """
    vanity: list[str] = []
    structures: list[str] = []
    pending_flat: str | None = None
    i, n = 0, len(tokens)

    def nxt(k=1):
        return tokens[i + k] if i + k < n else None

    def set_flat(code: str, ident: str):
        if res["flat_number"] or res["flat_number_suffix"]:
            warnings.append(f"second flat component ignored: {code} {ident}")
            return
        pre, num, suf = _split_id(ident) if ident else ("", "", "")
        res.update(flat_type_code=code, flat_number_prefix=pre, flat_number=num, flat_number_suffix=suf)

    def set_level(code: str, ident: str):
        pre, num, suf = _split_id(ident) if ident else ("", "", "")
        res.update(level_type_code=code, level_number_prefix=pre, level_number=num, level_number_suffix=suf)

    while i < n:
        t = tokens[i]
        if t == ",":
            i += 1
            continue

        # Multi-word level / flat phrases ("LOWER GROUND FLOOR", "CAR SPACE").
        matched = False
        for size in (3, 2):
            phrase = " ".join(tokens[i:i + size])
            if len(tokens[i:i + size]) < size:
                continue
            if phrase in LEVEL_TYPE_ALIASES:
                code = LEVEL_TYPE_ALIASES[phrase]
                ident = tokens[i + size] if i + size < n and _is_id_token(tokens[i + size]) else ""
                set_level(code, ident)
                i += size + (1 if ident else 0)
                matched = True
                break
            if phrase in FLAT_TYPE_ALIASES and " " in phrase:
                code = FLAT_TYPE_ALIASES[phrase]
                if i + size < n and _is_id_token(tokens[i + size]):
                    set_flat(code, tokens[i + size])
                    i += size + 1
                else:
                    pending_flat = code
                    i += size
                matched = True
                break
        if matched:
            continue

        # Ordinal floors: "1ST FLOOR", "FIRST LEVEL".
        ordinal = _ORDINAL_RE.match(t)
        ord_num = int(ordinal.group(1)) if ordinal else _ORDINAL_WORDS.get(t)
        if ord_num is not None and nxt() in ("FLOOR", "FLR", "LEVEL", "FL"):
            set_level("FL" if nxt() in ("FLOOR", "FLR", "FL") else "L", str(ord_num))
            i += 2
            continue

        # Building / tower inside a complex: "TOWER A", "NORTH TOWER", "A BLOCK".
        if t in STRUCTURE_WORDS and nxt() and len(nxt()) <= 4 and _is_id_token(nxt()):
            structures.append(f"{STRUCTURE_WORDS[t]} {nxt()}")
            i += 2
            continue
        if nxt() in STRUCTURE_WORDS and (t in _DIRECTIONS or (len(t) <= 2 and _is_id_token(t))):
            structures.append(f"{STRUCTURE_WORDS[nxt()]} {t}")
            i += 2
            continue

        if t == "LOT" and nxt() and _is_id_token(nxt()):
            pre, num, suf = _split_id(nxt())
            res.update(lot_number_prefix=pre, lot_number=num, lot_number_suffix=suf)
            i += 2
            continue

        attached = _ATTACHED_RE.match(t)
        if attached and t not in FLAT_TYPE_ALIASES:
            kw, ident = attached["kw"], attached["id"]
            if kw in _LEVEL_ATTACHED:
                set_level("L", ident)
            else:
                set_flat(FLAT_TYPE_ALIASES.get(kw, kw), ident)
            i += 1
            continue

        if t in LEVEL_TYPE_ALIASES:
            code = LEVEL_TYPE_ALIASES[t]
            if nxt() and _is_id_token(nxt()):
                set_level(code, nxt())
                i += 2
                continue
            if code in LEVEL_TYPES_NUMBERLESS:
                set_level(code, "")
                i += 1
                continue

        if t in FLAT_TYPE_ALIASES and t not in STRUCTURE_WORDS:
            code = FLAT_TYPE_ALIASES[t]
            if nxt() and _is_id_token(nxt()):
                set_flat(code, nxt())
                i += 2
                continue
            if not vanity and (nxt() is None or nxt() == ","):
                pending_flat = code
                i += 1
                continue

        if _is_id_token(t) and any(ch.isdigit() for ch in t):
            if pending_flat:
                set_flat(pending_flat, t)
                pending_flat = None
            elif not res["flat_number"]:
                set_flat("", t)
                warnings.append(f"flat type not given for '{t}'")
            else:
                vanity.append(t)
            i += 1
            continue

        vanity.append(t)
        i += 1

    if pending_flat and pending_flat in _NUMBERLESS_FLATS and not res["flat_type_code"]:
        set_flat(pending_flat, "")
        pending_flat = None
    name = " ".join(structures + vanity).strip(" ,")
    res["building_name"] = name
    return pending_flat


def parse_address(text: str, localities: Iterable[str] | None = None) -> dict:
    """Parse an Australian address string into G-NAF style components.

    ``localities`` is an optional gazetteer of upper-case locality names (for
    example from the G-NAF LOCALITY table); it resolves cases such as
    ``10 HIGH ST PARK RIDGE`` where a locality word is also a street type.

    Returns a dict of the :data:`FIELDS` plus ``parse_warnings`` (a list).
    """
    res = {f: "" for f in FIELDS}
    warnings: list[str] = []
    loc_set = {l.upper() for l in localities} if localities is not None and not isinstance(localities, set) else localities

    tokens = _strip_tail(_tokenise(text), res)
    anchor = _find_anchor(tokens)
    if anchor is None:
        # No street number: perhaps "LOT 12 SMITH ST" with a non-standard lot token.
        warnings.append("no street number found")
        res["parse_warnings"] = warnings
        res["street_name"] = " ".join(t for t in tokens if t != ",")
        return res

    pre = tokens[:anchor]
    num_tok = tokens[anchor]
    rem = tokens[anchor + 1:]

    slash_left = None
    if "/" in num_tok:
        slash_left, num_tok = num_tok.rsplit("/", 1)

    is_lot = bool(pre) and pre[-1] == "LOT" and slash_left is None
    if is_lot:
        pre = pre[:-1]

    pending_flat = _parse_subaddress(pre, res, warnings)

    if slash_left:
        lefts = slash_left.split("/")
        flat_part = lefts[-1]
        for extra in lefts[:-1]:
            if not res["level_number"]:
                res.update(level_type_code="L", level_number=extra)
        attached = _ATTACHED_RE.match(flat_part)
        if attached and attached["kw"] in _LEVEL_ATTACHED:
            pre_, num_, suf_ = _split_id(attached["id"])
            res.update(level_type_code="L", level_number_prefix=pre_, level_number=num_, level_number_suffix=suf_)
        else:
            code = pending_flat or res["flat_type_code"] or ""
            ident = flat_part
            if attached:
                code = FLAT_TYPE_ALIASES.get(attached["kw"], attached["kw"])
                ident = attached["id"]
            pre_, num_, suf_ = _split_id(ident)
            res.update(flat_type_code=code, flat_number_prefix=pre_, flat_number=num_, flat_number_suffix=suf_)
            pending_flat = None
    if pending_flat:
        warnings.append(f"flat type {pending_flat} has no number")

    m = _NUM_RE.match(num_tok)
    if is_lot:
        res.update(lot_number_prefix=m["p1"], lot_number=m["n1"], lot_number_suffix=m["s1"])
    else:
        res.update(number_first_prefix=m["p1"], number_first=m["n1"], number_first_suffix=m["s1"])
        if m["n2"]:
            res.update(number_last_prefix=m["p2"], number_last=m["n2"], number_last_suffix=m["s2"])

    expect_locality = bool(res["state"] or res["postcode"])
    name, stype, suffix, locality = _split_street_locality(rem, loc_set, expect_locality)
    res["street_name"] = " ".join(name)
    res["street_type_code"] = STREET_TYPE_ALIASES.get(stype, stype) if stype else ""
    if suffix:
        res["street_suffix_code"] = STREET_SUFFIX_ABBREV_ALIASES.get(suffix) or STREET_SUFFIX_WORD_ALIASES.get(suffix, suffix)
    res["locality_name"] = " ".join(locality)
    if not stype:
        warnings.append("no street type recognised")
    if expect_locality and not locality:
        warnings.append("no locality found")
    res["parse_warnings"] = warnings
    return res

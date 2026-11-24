"""Load a G-NAF PSV release into one row per active address (polars).

Expected layout (as distributed by Geoscape / data.gov.au)::

    G-NAF/G-NAF <MONTH> <YEAR>/Standard/<STATE>_ADDRESS_DETAIL_psv.psv
    G-NAF/G-NAF <MONTH> <YEAR>/Standard/<STATE>_ADDRESS_DEFAULT_GEOCODE_psv.psv
    G-NAF/G-NAF <MONTH> <YEAR>/Standard/<STATE>_STREET_LOCALITY_psv.psv
    G-NAF/G-NAF <MONTH> <YEAR>/Standard/<STATE>_LOCALITY_psv.psv
    G-NAF/G-NAF <MONTH> <YEAR>/Standard/<STATE>_STATE_psv.psv
    G-NAF/G-NAF <MONTH> <YEAR>/Standard/<STATE>_PRIMARY_SECONDARY_psv.psv   (optional)

Files are located by recursive glob, so any parent folder works.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..frames import c
from ..geo import Layer, crs_string
from ._io import read_psv

DETAIL_COLS = [
    "ADDRESS_DETAIL_PID", "DATE_RETIRED", "BUILDING_NAME",
    "LOT_NUMBER_PREFIX", "LOT_NUMBER", "LOT_NUMBER_SUFFIX",
    "FLAT_TYPE_CODE", "FLAT_NUMBER_PREFIX", "FLAT_NUMBER", "FLAT_NUMBER_SUFFIX",
    "LEVEL_TYPE_CODE", "LEVEL_NUMBER_PREFIX", "LEVEL_NUMBER", "LEVEL_NUMBER_SUFFIX",
    "NUMBER_FIRST_PREFIX", "NUMBER_FIRST", "NUMBER_FIRST_SUFFIX",
    "NUMBER_LAST_PREFIX", "NUMBER_LAST", "NUMBER_LAST_SUFFIX",
    "STREET_LOCALITY_PID", "LOCALITY_PID", "ALIAS_PRINCIPAL", "POSTCODE",
    "LEGAL_PARCEL_ID", "CONFIDENCE", "ADDRESS_SITE_PID", "PRIMARY_SECONDARY",
]


def _find(root: Path, state: str, table: str) -> Path | None:
    hits = sorted(root.rglob(f"{state}_{table}_psv.psv"))
    return hits[-1] if hits else None


def is_gnaf_release(path: str | Path) -> bool:
    p = Path(path)
    return p.is_dir() and any(p.rglob("*_ADDRESS_DETAIL_psv.psv"))


def available_states(root: str | Path) -> list[str]:
    return sorted({p.name.split("_ADDRESS_DETAIL")[0] for p in Path(root).rglob("*_ADDRESS_DETAIL_psv.psv")})


def _active(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(c("DATE_RETIRED").is_null()).drop("DATE_RETIRED") if "DATE_RETIRED" in df.columns else df


def _upper(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({col: col.upper() for col in df.columns})


def load_gnaf(root: str | Path, states: list[str] | None = None, crs: str = "EPSG:7844") -> Layer:
    """Load active addresses for ``states`` (default: every state found under ``root``).

    G-NAF coordinates are GDA2020 (EPSG:7844) in current releases; pass
    ``crs="EPSG:4283"`` for a GDA94 release.

    Returns a point :class:`~shell_identifier.geo.Layer` (``x`` = longitude, ``y`` =
    latitude) with one row per ADDRESS_DETAIL_PID, lower-case G-NAF field names
    plus ``address_pid``, ``state``, ``geocode_type``, ``ps_primary_pid`` and
    ``n_ps_secondaries``. ``meta["localities"]`` holds the locality gazetteer.
    """
    root = Path(root)
    states = [s.upper() for s in (states or available_states(root))]
    if not states:
        raise FileNotFoundError(f"No *_ADDRESS_DETAIL_psv.psv files found under {root}")

    frames, localities = [], set()
    for state in states:
        detail_path = _find(root, state, "ADDRESS_DETAIL")
        if detail_path is None:
            raise FileNotFoundError(f"{state}_ADDRESS_DETAIL_psv.psv not found under {root}")
        detail = _active(_upper(read_psv(detail_path, DETAIL_COLS)))
        street = _active(_upper(read_psv(_find(root, state, "STREET_LOCALITY"),
                                         ["STREET_LOCALITY_PID", "STREET_NAME", "STREET_TYPE_CODE",
                                          "STREET_SUFFIX_CODE", "DATE_RETIRED"])))
        locality = _active(_upper(read_psv(_find(root, state, "LOCALITY"),
                                           ["LOCALITY_PID", "LOCALITY_NAME", "STATE_PID", "DATE_RETIRED"])))
        state_path = _find(root, state, "STATE")
        if state_path is not None:
            st = _upper(read_psv(state_path, ["STATE_PID", "STATE_ABBREVIATION"]))
            locality = locality.join(st, on="STATE_PID", how="left")
        else:
            locality = locality.with_columns(STATE_ABBREVIATION=pl.lit(state))
        localities.update(n.upper() for n in locality["LOCALITY_NAME"].drop_nulls().to_list())

        geo = _active(_upper(read_psv(_find(root, state, "ADDRESS_DEFAULT_GEOCODE"),
                                      ["ADDRESS_DETAIL_PID", "GEOCODE_TYPE_CODE", "LONGITUDE", "LATITUDE",
                                       "DATE_RETIRED"])))
        geo = geo.unique("ADDRESS_DETAIL_PID", keep="first", maintain_order=True)

        df = (detail.join(street, on="STREET_LOCALITY_PID", how="left", maintain_order="left")
              .join(locality.select("LOCALITY_PID", "LOCALITY_NAME", "STATE_ABBREVIATION"), on="LOCALITY_PID",
                    how="left", maintain_order="left")
              .join(geo, on="ADDRESS_DETAIL_PID", how="left", maintain_order="left")
              .with_columns(c("STATE_ABBREVIATION").fill_null(state)))

        ps_path = _find(root, state, "PRIMARY_SECONDARY")
        if ps_path is not None:
            ps = _active(_upper(read_psv(ps_path, ["PRIMARY_PID", "SECONDARY_PID", "DATE_RETIRED"])))
            n_sec = ps.group_by("PRIMARY_PID").agg(pl.len().alias("N_PS_SECONDARIES"))
            prim = (ps.unique("SECONDARY_PID", keep="first", maintain_order=True)
                    .select(c("SECONDARY_PID").alias("ADDRESS_DETAIL_PID"), c("PRIMARY_PID").alias("PS_PRIMARY_PID")))
            df = (df.join(n_sec.rename({"PRIMARY_PID": "ADDRESS_DETAIL_PID"}), on="ADDRESS_DETAIL_PID", how="left",
                          maintain_order="left")
                  .join(prim, on="ADDRESS_DETAIL_PID", how="left", maintain_order="left"))
        else:
            df = df.with_columns(N_PS_SECONDARIES=pl.lit(None, dtype=pl.UInt32), PS_PRIMARY_PID=pl.lit(None, dtype=pl.Utf8))
        frames.append(df)

    df = pl.concat(frames, how="diagonal_relaxed")
    df = df.rename({col: col.lower() for col in df.columns}).rename({
        "address_detail_pid": "address_pid", "state_abbreviation": "state", "geocode_type_code": "geocode_type"})
    df = df.with_columns(
        n_ps_secondaries=c("n_ps_secondaries").cast(pl.Int64, strict=False).fill_null(0),
        x=c("longitude").cast(pl.Float64, strict=False),
        y=c("latitude").cast(pl.Float64, strict=False),
        source=pl.lit("gnaf"),
    ).drop("longitude", "latitude")
    missing = df.filter(c("x").is_null() | c("y").is_null()).height
    df = df.filter(c("x").is_not_null() & c("y").is_not_null())
    return Layer(df, None, crs_string(crs), meta={"localities": localities, "dropped_without_geocode": missing})

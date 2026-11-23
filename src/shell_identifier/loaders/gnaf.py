"""Load a G-NAF PSV release into one row per active address with a point geometry.

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

import geopandas as gpd
import pandas as pd

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


def available_states(root: str | Path) -> list[str]:
    return sorted({p.name.split("_ADDRESS_DETAIL")[0] for p in Path(root).rglob("*_ADDRESS_DETAIL_psv.psv")})


def load_gnaf(root: str | Path, states: list[str] | None = None, crs: str = "EPSG:7844") -> gpd.GeoDataFrame:
    """Load active addresses for ``states`` (default: every state found under ``root``).

    G-NAF coordinates are GDA2020 (EPSG:7844) in current releases; pass
    ``crs="EPSG:4283"`` for a GDA94 release.

    The returned frame has one row per ADDRESS_DETAIL_PID with lower-case
    G-NAF field names plus ``address_pid``, ``geocode_type``,
    ``ps_primary_pid`` and ``n_ps_secondaries``. ``frame.attrs["localities"]``
    holds the locality gazetteer for the free-text parser.
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
        detail = read_psv(detail_path, DETAIL_COLS)
        detail = detail[detail["DATE_RETIRED"].isna()].drop(columns="DATE_RETIRED")

        street = read_psv(_find(root, state, "STREET_LOCALITY"),
                          ["STREET_LOCALITY_PID", "STREET_NAME", "STREET_TYPE_CODE", "STREET_SUFFIX_CODE", "DATE_RETIRED"])
        street = street[street["DATE_RETIRED"].isna()].drop(columns="DATE_RETIRED")
        locality = read_psv(_find(root, state, "LOCALITY"), ["LOCALITY_PID", "LOCALITY_NAME", "STATE_PID", "DATE_RETIRED"])
        locality = locality[locality["DATE_RETIRED"].isna()].drop(columns="DATE_RETIRED")
        state_path = _find(root, state, "STATE")
        if state_path is not None:
            st = read_psv(state_path, ["STATE_PID", "STATE_ABBREVIATION"])
            locality = locality.merge(st, on="STATE_PID", how="left")
        else:
            locality["STATE_ABBREVIATION"] = state
        localities.update(locality["LOCALITY_NAME"].dropna().str.upper())

        geo = read_psv(_find(root, state, "ADDRESS_DEFAULT_GEOCODE"),
                       ["ADDRESS_DETAIL_PID", "GEOCODE_TYPE_CODE", "LONGITUDE", "LATITUDE", "DATE_RETIRED"])
        geo = geo[geo["DATE_RETIRED"].isna()].drop(columns="DATE_RETIRED").drop_duplicates("ADDRESS_DETAIL_PID")

        df = (detail.merge(street, on="STREET_LOCALITY_PID", how="left")
                    .merge(locality[["LOCALITY_PID", "LOCALITY_NAME", "STATE_ABBREVIATION"]], on="LOCALITY_PID", how="left")
                    .merge(geo, on="ADDRESS_DETAIL_PID", how="left"))
        df["STATE_ABBREVIATION"] = df["STATE_ABBREVIATION"].fillna(state)

        ps_path = _find(root, state, "PRIMARY_SECONDARY")
        if ps_path is not None:
            ps = read_psv(ps_path, ["PRIMARY_PID", "SECONDARY_PID", "DATE_RETIRED"])
            ps = ps[ps["DATE_RETIRED"].isna()]
            n_sec = ps.groupby("PRIMARY_PID").size().rename("N_PS_SECONDARIES")
            prim = ps.drop_duplicates("SECONDARY_PID").set_index("SECONDARY_PID")["PRIMARY_PID"].rename("PS_PRIMARY_PID")
            df = df.merge(n_sec, left_on="ADDRESS_DETAIL_PID", right_index=True, how="left")
            df = df.merge(prim, left_on="ADDRESS_DETAIL_PID", right_index=True, how="left")
        else:
            df["N_PS_SECONDARIES"] = pd.NA
            df["PS_PRIMARY_PID"] = pd.NA
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={
        "address_detail_pid": "address_pid",
        "state_abbreviation": "state",
        "geocode_type_code": "geocode_type",
    })
    df["n_ps_secondaries"] = pd.to_numeric(df["n_ps_secondaries"], errors="coerce").fillna(0).astype(int)
    lon = pd.to_numeric(df.pop("longitude"), errors="coerce")
    lat = pd.to_numeric(df.pop("latitude"), errors="coerce")
    missing = lon.isna() | lat.isna()
    if missing.any():
        df = df[~missing]
        lon, lat = lon[~missing], lat[~missing]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lon, lat), crs=crs)
    gdf["source"] = "gnaf"
    gdf.attrs["localities"] = localities
    gdf.attrs["dropped_without_geocode"] = int(missing.sum())
    return gdf.reset_index(drop=True)

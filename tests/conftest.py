"""Helpers for building tiny hand-made parcels in a metric CRS."""

import numpy as np
import polars as pl
import pytest
from shapely.geometry import box

from shell_identifier.config import load_config
from shell_identifier.geo import Layer
from shell_identifier.pipeline import run_pipeline

CRS = "EPSG:7856"  # metres; the pipeline reprojects to Albers internally


def addr(pid, x, y, street="TEST", stype="STREET", num=None, sfx="", last=None, flat_type="", flat=None,
         level_type="", level=None, building="", alias="P", ps="", n_ps=0, legal="", geocode="PC"):
    return {
        "address_pid": pid, "x": float(x), "y": float(y), "street_name": street, "street_type_code": stype,
        "locality_name": "TESTWOOD", "state": "NSW", "postcode": "2999",
        "number_first": "" if num is None else str(num), "number_first_suffix": sfx,
        "number_last": "" if last is None else str(last), "flat_type_code": flat_type,
        "flat_number": "" if flat is None else str(flat), "level_type_code": level_type,
        "level_number": "" if level is None else str(level), "building_name": building,
        "alias_principal": alias, "primary_secondary": ps, "n_ps_secondaries": n_ps,
        "legal_parcel_id": legal, "geocode_type": geocode,
    }


def _layer(rows: list[dict]) -> Layer:
    attrs = [{k: v for k, v in r.items() if k != "geometry"} for r in rows]
    return Layer(pl.DataFrame(attrs), np.array([r["geometry"] for r in rows], dtype=object), CRS)


class Case:
    def __init__(self):
        self.addresses, self.parcels, self.buildings = [], [], []

    def parcel(self, pid, x0, y0, x1, y1, **attrs):
        self.parcels.append({"parcel_pid": pid, "geometry": box(x0, y0, x1, y1), **attrs})
        return self

    def building(self, pid, x0, y0, x1, y1, levels=1):
        area = (x1 - x0) * (y1 - y0)
        self.buildings.append({"building_pid": pid, "geometry": box(x0, y0, x1, y1), "building_area": area,
                               "total_floor_area": area * levels, "est_levels": float(levels)})
        return self

    def add(self, *rows):
        self.addresses.extend(rows)
        return self

    def run(self, cfg=None):
        a = Layer(pl.DataFrame(self.addresses), None, CRS)
        keys = sorted({k for p in self.parcels for k in p})
        parcels = _layer([{k: p.get(k) for k in keys} for p in self.parcels])
        buildings = _layer(self.buildings) if self.buildings else None
        res = run_pipeline(a, parcels, buildings, None, None, cfg or load_config())
        self.result = res["addresses"]
        self._rows = {r["address_pid"]: r for r in self.result.iter_rows(named=True)}
        return self

    def p(self, pid):
        return float(self._rows[pid]["shell_probability"])

    def row(self, pid) -> dict:
        return self._rows[pid]


@pytest.fixture
def case():
    return Case()

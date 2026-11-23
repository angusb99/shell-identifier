"""Synthetic Geoscape Cadastre, G-NAF and Geoscape Buildings with shell ground truth.

A fictional suburb is laid out as a grid of streets and blocks. Each lot is
filled with one *scenario* (house, duplex, walk-up, tower precinct, ...) that
creates addresses, building footprints, strata parcels and G-NAF
primary/secondary links, and records for every address whether it is a shell.

Realistic noise is injected: partial primary/secondary linkage, unpopulated
strata counts, stacked vs spread unit geocodes, geocodes that slip onto the
neighbouring lot, missing building captures, aliases and retired records.

Outputs (``out_dir``)::

    G-NAF/G-NAF <MONTH> <YEAR>/Standard/NSW_*_psv.psv
    G-NAF/G-NAF <MONTH> <YEAR>/Authority Code/Authority_Code_*_psv.psv
    Cadastre/NSW_CADASTRE.gpkg            layer: cadastre
    Buildings/NSW_BUILDINGS.gpkg          layers: buildings, building_address, building_cadastre
    truth/shell_truth.csv                 ground truth per ADDRESS_DETAIL_PID
    text/addresses_free_text.csv          the same addresses as messy free text
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point, box

from .. import authority as aut

LOCAL_CRS = "EPSG:7856"   # GDA2020 / MGA zone 56 (metres) for construction
OUT_CRS = "EPSG:7844"     # GDA2020 geographic, as delivered by Geoscape
AREA_CRS = "EPSG:9473"    # GDA2020 / Australian Albers for the cadastre `area` attribute
ORIGIN = (318_000.0, 6_252_000.0)
RELEASE = "G-NAF AUGUST 2026"
STATE = "NSW"

BLOCK_LEN, LOT_DEPTH, ROAD_HALF = 300.0, 45.0, 10.0
ROW_PITCH = 2 * LOT_DEPTH + 2 * ROAD_HALF        # 110 m between E-W street centrelines
COL_PITCH = BLOCK_LEN + 2 * ROAD_HALF            # 320 m between N-S street centrelines

EW_NAMES = ["BANKSIA", "WATTLE", "ACACIA", "MYRTLE", "GREVILLEA", "KURRAJONG", "JACARANDA",
            "TELOPEA", "WARATAH", "BORONIA", "CASUARINA", "MELALEUCA", "ANGOPHORA", "LILLY PILLY"]
EW_TYPES = ["STREET", "AVENUE", "ROAD", "PARADE", "CRESCENT", "STREET", "DRIVE"]
NS_NAMES = ["CHARLES", "VICTORIA", "GEORGE", "ELIZABETH", "WILLIAM", "MACQUARIE", "PHILLIP",
            "HUNTER", "BLIGH", "KENT", "SUSSEX", "CASTLEREAGH", "PITT", "O'CONNELL"]
LOCALITIES = [("TESTWOOD", "2999"), ("SAMPLE HILL", "2998"), ("MOCKTON PARK", "2997")]
VANITY = ["THE ARCADIA", "THE MERIDIAN", "PARKSIDE", "THE ASCOT", "HARBOUR VIEW", "THE ELMS"]
TOWER_VANITY = ["THE ASH", "THE BIRCH", "THE CEDAR"]

# name: (weight, (min width, max width), street numbers consumed)
SCENARIOS: dict[str, tuple[float, tuple[float, float], int]] = {
    "house": (0.30, (14, 18), 1),
    "vacant": (0.03, (14, 18), 1),
    "house_rear": (0.04, (16, 20), 1),
    "granny_unit": (0.03, (16, 20), 1),
    "dual_occupied": (0.03, (16, 20), 1),
    "duplex_shell": (0.09, (18, 22), 1),
    "triplex_occupied": (0.03, (24, 28), 1),
    "shop_top": (0.02, (10, 14), 1),
    "townhouses": (0.05, (30, 40), 1),
    "walkup": (0.08, (30, 40), 1),
    "units_no_parent": (0.04, (24, 30), 1),
    "mixed_use": (0.04, (20, 30), 1),
    "office": (0.02, (30, 40), 1),
    "highrise": (0.03, (80, 100), 8),
    "range_townhouses": (0.03, (27, 45), 0),   # numbers depend on width
    "retirement": (0.015, (90, 100), 1),
    "corner": (0.04, (18, 22), 1),
    "split_precinct": (0.01, (45, 55), 1),
}


@dataclass
class Locality:
    pid: str
    name: str
    postcode: str


@dataclass
class Street:
    name: str
    type_code: str
    suffix: str = ""
    by_locality: dict = field(default_factory=dict)  # locality name -> street_locality_pid


@dataclass
class Site:
    sid: int
    x0: float
    w: float
    y_front: float
    d: float
    facing: int                # -1: lot runs south of its frontage, +1: north
    street: Street
    locality: Locality
    numbers: list[int]
    scenario: str = ""
    lot_no: int = 1
    dp: str = ""
    sp: str = ""
    strata: list = field(default_factory=list)    # (polygon, "Horizontal"/"Vertical")
    counts: dict = field(default_factory=dict)    # strata counts (None when unpopulated)
    parcel_polys: list = field(default_factory=list)
    cross: tuple | None = None                    # (Street, number) for corner lots

    def xy(self, u: float, v: float) -> tuple[float, float]:
        return self.x0 + u, self.y_front + self.facing * v

    def rect(self, u0: float, v0: float, u1: float, v1: float):
        xa, ya = self.xy(u0, v0)
        xb, yb = self.xy(u1, v1)
        return box(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    @property
    def polygon(self):
        return self.rect(0, 0, self.w, self.d)

    def fcs(self):
        return self.xy(self.w / 2, 6.0)

    def pc(self):
        return self.xy(self.w / 2, self.d / 2)


class Builder:
    def __init__(self, seed: int = 42, p_slip: float = 0.02, p_missing_building: float = 0.03,
                 p_merged_footprint: float = 0.3, p_stacked_suffix: float = 0.2):
        self.rng = np.random.default_rng(seed)
        self.p_slip = p_slip
        self.p_missing_building = p_missing_building
        self.p_merged_footprint = p_merged_footprint   # attached dwellings captured as one footprint
        self.p_stacked_suffix = p_stacked_suffix       # every suffix address geocoded at the property centroid
        self.addresses: list[dict] = []
        self.buildings: list[dict] = []
        self.parcels: list[dict] = []
        self.ps_links: list[tuple[str, str]] = []
        self.aliases: list[tuple[str, str]] = []
        self.sites: list[Site] = []
        self._by_pid: dict[str, dict] = {}
        self._addr_seq = itertools.count(717_000_001)
        self._site_seq = itertools.count(712_000_001)
        self._hex = itertools.count(0x1A2B3C4D5E)

    # ------------------------------------------------------------------ ids
    def hexpid(self, prefix: str) -> str:
        return f"{prefix}{next(self._hex):012x}"[:15]

    def plan(self, kind: str) -> str:
        return f"{kind}{self.rng.integers(10_000, 999_999)}"

    # ------------------------------------------------------------ geometry
    def bc(self, poly) -> tuple[float, float]:
        c = poly.centroid
        return c.x, c.y

    def uc(self, poly) -> tuple[float, float]:
        minx, miny, maxx, maxy = poly.bounds
        return (float(self.rng.uniform(minx + 1, maxx - 1)), float(self.rng.uniform(miny + 1, maxy - 1)))

    def attached(self, s: Site, polys: list, levels: int) -> list:
        """Footprints for attached dwellings; sometimes captured as one merged building."""
        if len(polys) > 1 and self.rng.random() < self.p_merged_footprint:
            minx = min(p.bounds[0] for p in polys); miny = min(p.bounds[1] for p in polys)
            maxx = max(p.bounds[2] for p in polys); maxy = max(p.bounds[3] for p in polys)
            self.building(s, box(minx, miny, maxx, maxy), levels)
        else:
            for poly in polys:
                self.building(s, poly, levels)
        return polys

    def building(self, s: Site, poly, levels: int, use: str = "Residential") -> int:
        self.buildings.append({
            "site": s.sid, "geometry": poly, "levels": int(levels), "use": use,
            "captured": bool(self.rng.random() >= self.p_missing_building),
            "pid": self.hexpid("bld"),
        })
        return len(self.buildings) - 1

    # ------------------------------------------------------------ addresses
    def addr(self, s: Site, *, xy, gt, shell: bool, role: str, num=None, sfx="", last=None,
             ftype="", fnum=None, fpre="", fsfx="", ltype="", lnum=None, bname="", legal="",
             street: Street | None = None, alias="P", slip=False, site_pid=None) -> str:
        if slip and self.rng.random() < self.p_slip:
            xy = (s.x0 - 3.0, xy[1])   # geocode lands on the western neighbour
            role = role + "_slipped"
        pid = f"GA{STATE}{next(self._addr_seq)}"
        self.addresses.append({
            "pid": pid, "site": s.sid, "street": street or s.street, "locality": s.locality,
            "number_first": num, "number_first_suffix": sfx, "number_last": last,
            "flat_type": ftype, "flat_prefix": fpre, "flat_number": fnum, "flat_suffix": fsfx,
            "level_type": ltype, "level_number": lnum, "building_name": bname,
            "x": float(xy[0]), "y": float(xy[1]), "geocode_type": gt,
            "alias": alias, "ps": "", "legal": legal, "is_shell": bool(shell), "role": role,
            "scenario": s.scenario, "address_site_pid": site_pid or str(next(self._site_seq)),
        })
        self._by_pid[pid] = self.addresses[-1]
        return pid

    def link(self, primary: str, secondaries: list[str], prob: float, flag_only_prob: float = 0.1) -> None:
        """G-NAF primary/secondary relationship, present with probability ``prob``."""
        if not secondaries or self.rng.random() >= prob:
            return
        self._by_pid[primary]["ps"] = "P"
        rows_too = self.rng.random() >= flag_only_prob
        for sec in secondaries:
            self._by_pid[sec]["ps"] = "S"
            if rows_too:
                self.ps_links.append((primary, sec))

    def strata_counts(self, s: Site, horizontal: int = 0, vertical: int = 0, populated_prob: float = 0.6) -> None:
        if self.rng.random() < populated_prob:
            s.counts = {"total": horizontal + vertical, "horizontal": horizontal, "vertical": vertical}

    def place_geocode(self, s: Site) -> tuple[tuple[float, float], str]:
        return (s.pc(), "PC") if self.rng.random() < 0.6 else (s.fcs(), "FCS")

    # ------------------------------------------------------------ scenarios
    def sc_house(self, s: Site):
        depth = self.rng.uniform(15, 20)
        house = s.rect(2, 7, s.w - 2, 7 + depth)
        self.building(s, house, self.rng.choice([1, 2], p=[0.6, 0.4]))
        if self.rng.random() < 0.3:
            self.building(s, s.rect(s.w - 6, s.d - 7, s.w - 2, s.d - 3), 1)  # shed
        xy, gt = self.place_geocode(s)
        self.addr(s, num=s.numbers[0], xy=xy, gt=gt, shell=False, role="house", legal=f"{s.lot_no}/{s.dp}")

    def sc_vacant(self, s: Site):
        self.addr(s, num=s.numbers[0], xy=s.pc(), gt="PC", shell=False, role="vacant_land", legal=f"{s.lot_no}/{s.dp}")

    def _house_plus_rear(self, s: Site):
        house = s.rect(2, 7, s.w - 2, 24)
        rear = s.rect(s.w - 10, s.d - 11, s.w - 2, s.d - 3)
        self.building(s, house, self.rng.choice([1, 2]))
        self.building(s, rear, 1)
        gt = "FCS" if self.rng.random() < 0.4 else "PC"
        xy = s.fcs() if gt == "FCS" else s.pc()
        return xy, gt, rear

    def sc_house_rear(self, s: Site):
        xy, gt, rear = self._house_plus_rear(s)
        n = s.numbers[0]
        self.addr(s, num=n, xy=xy, gt=gt, shell=False, role="house_with_secondary", legal=f"{s.lot_no}/{s.dp}")
        ftype = self.rng.choice(["REAR", "REAR", "REAR", "CTGE", "BNGW"])
        self.addr(s, num=n, ftype=str(ftype), xy=self.bc(rear), gt="BC", shell=False, role="secondary_dwelling")

    def sc_granny_unit(self, s: Site):
        xy, gt, rear = self._house_plus_rear(s)
        n = s.numbers[0]
        self.addr(s, num=n, xy=xy, gt=gt, shell=False, role="house_with_secondary", legal=f"{s.lot_no}/{s.dp}")
        ftype = str(self.rng.choice(["UNIT", "FLAT"]))
        self.addr(s, num=n, ftype=ftype, fnum=1, xy=self.bc(rear), gt="BC", shell=False, role="secondary_dwelling")

    def sc_dual_occupied(self, s: Site):
        xy, gt, rear = self._house_plus_rear(s)
        n = s.numbers[0]
        self.addr(s, num=n, xy=xy, gt=gt, shell=False, role="house", legal=f"{s.lot_no}/{s.dp}")
        self.addr(s, num=n, sfx="A", xy=self.bc(rear), gt="BC", shell=False, role="dwelling")

    def sc_duplex_shell(self, s: Site):
        w, n = s.w, s.numbers[0]
        if self.rng.random() < 0.6:   # side by side
            halves = [s.rect(1.5, 7, w / 2 - 0.75, 26), s.rect(w / 2 + 0.75, 7, w - 1.5, 26)]
            strata = [s.rect(0, 0, w / 2, s.d), s.rect(w / 2, 0, w, s.d)]
            fronts = [s.xy(w / 4, 6), s.xy(3 * w / 4, 6)]
        else:                          # front and back
            halves = [s.rect(2, 6, w - 2, 20.5), s.rect(2, 24.5, w - 2, 39)]
            strata = [s.rect(0, 0, w, 22.5), s.rect(0, 22.5, w, s.d)]
            fronts = [s.xy(w / 2, 6), s.xy(3, 30)]
        self.attached(s, halves, int(self.rng.choice([1, 2])))
        s.strata += [(p, "Horizontal") for p in strata]
        all_at_pc = self.rng.random() < self.p_stacked_suffix
        self.strata_counts(s, horizontal=2)
        s.sp = self.plan("SP")
        base_xy, base_gt = (s.pc(), "PC") if self.rng.random() < 0.8 else (s.fcs(), "FCS")
        legal = f"CP/{s.sp}" if self.rng.random() < 0.5 else ""
        style = "suffix" if self.rng.random() < 0.5 else "slash"
        base = self.addr(s, num=n, xy=base_xy, gt=base_gt, shell=True,
                         role="suffix_shell" if style == "suffix" else "street_shell", legal=legal)
        kids = []
        for i, (h, front) in enumerate(zip(halves, fronts), start=1):
            if style == "suffix":
                xy, gt = (self.bc(h), "BC") if self.rng.random() < 0.7 else (front, "FCS")
                if all_at_pc:
                    xy, gt = s.pc(), "PC"
                kids.append(self.addr(s, num=n, sfx="AB"[i - 1], xy=xy, gt=gt, shell=False, role="dwelling",
                                      legal=f"{i}/{s.sp}"))
            else:
                xy, gt = (self.bc(h), "BC") if self.rng.random() < 0.6 else (base_xy, base_gt)
                kids.append(self.addr(s, num=n, ftype="UNIT", fnum=i, xy=xy, gt=gt, shell=False, role="unit",
                                      legal=f"{i}/{s.sp}"))
        if style == "slash":
            self.link(base, kids, prob=0.6)

    def sc_triplex_occupied(self, s: Site):
        w, n = s.w, s.numbers[0]
        third = (w - 2) / 3
        s.sp = self.plan("SP")
        polys = [s.rect(1 + i * third + 0.5, 7, 1 + (i + 1) * third - 0.5, 25) for i in range(3)]
        self.attached(s, polys, int(self.rng.choice([1, 2])))
        all_at_pc = self.rng.random() < self.p_stacked_suffix
        for i, (sfx, b) in enumerate(zip(["", "A", "B"], polys)):
            s.strata.append((s.rect(i * w / 3, 0, (i + 1) * w / 3, s.d), "Horizontal"))
            xy, gt = (s.pc(), "PC") if all_at_pc else (self.bc(b), "BC")
            self.addr(s, num=n, sfx=sfx, xy=xy, gt=gt, shell=False, role="dwelling", legal=f"{i + 1}/{s.sp}")
        self.strata_counts(s, horizontal=3)

    def sc_shop_top(self, s: Site):
        n = s.numbers[0]
        b = s.rect(0.5, 0.5, s.w - 0.5, 20)
        self.building(s, b, 2, use="Commercial")
        self.addr(s, num=n, xy=s.fcs(), gt="FCS", shell=False, role="shop", legal=f"{s.lot_no}/{s.dp}")
        xy, gt = (s.fcs(), "FCS") if self.rng.random() < 0.5 else (self.bc(b), "BC")
        self.addr(s, num=n, sfx="A", xy=xy, gt=gt, shell=False, role="dwelling")

    def sc_townhouses(self, s: Site):
        n, k = s.numbers[0], int(self.rng.integers(3, 9))
        slot = (s.d - 8) / k
        ftype = "TNHS" if self.rng.random() < 0.3 else "UNIT"
        stacked = self.rng.random() < 0.3
        base_xy, base_gt = self.place_geocode(s)
        s.sp = self.plan("SP")
        base = self.addr(s, num=n, xy=base_xy, gt=base_gt, shell=True, role="street_shell",
                         legal=f"CP/{s.sp}" if self.rng.random() < 0.5 else "")
        kids = []
        for i in range(k):
            b = s.rect(5, 4 + i * slot + 0.4, s.w - 2, 4 + (i + 1) * slot - 0.4)
            self.building(s, b, 2)
            s.strata.append((s.rect(4, 4 + i * slot, s.w, 4 + (i + 1) * slot), "Horizontal"))
            xy, gt = (base_xy, base_gt) if stacked else (self.bc(b), "BC")
            kids.append(self.addr(s, num=n, ftype=ftype, fnum=i + 1, xy=xy, gt=gt, shell=False, role="unit",
                                  legal=f"{i + 1}/{s.sp}", slip=not stacked))
        self.strata_counts(s, horizontal=k)
        self.link(base, kids, prob=0.6)

    def _vanity(self, s: Site, prob: float) -> str:
        return str(self.rng.choice(VANITY)) if self.rng.random() < prob else ""

    def sc_walkup(self, s: Site):
        n = s.numbers[0]
        levels = int(self.rng.integers(3, 5))
        per_level = int(self.rng.integers(2, 7))
        b = s.rect(3, 6, s.w - 3, s.d - 6)
        self.building(s, b, levels)
        base_xy, base_gt = self.place_geocode(s)
        stacked = self.rng.random() < 0.6
        vanity = self._vanity(s, 0.3)
        unit_vanity = vanity if self.rng.random() < 0.5 else ""
        level_based = self.rng.random() < 0.5
        with_level = level_based and self.rng.random() < 0.4
        s.sp = self.plan("SP")
        site_pid = str(next(self._site_seq))
        base = self.addr(s, num=n, xy=base_xy, gt=base_gt, shell=True, role="street_shell", bname=vanity,
                         legal=f"CP/{s.sp}" if self.rng.random() < 0.6 else "", site_pid=site_pid)
        kids, i = [], 0
        for lv in range(levels):
            for j in range(per_level):
                i += 1
                fnum = (lv + 1) * 100 + j + 1 if level_based else i
                xy, gt = (base_xy, base_gt) if stacked else (self.uc(b), "UC")
                kids.append(self.addr(
                    s, num=n, ftype="UNIT", fnum=fnum, ltype="L" if with_level else "",
                    lnum=lv + 1 if with_level else None, bname=unit_vanity, xy=xy, gt=gt, shell=False,
                    role="unit", legal=f"{i}/{s.sp}", slip=not stacked, site_pid=site_pid if stacked else None))
        n_cars = 0
        if self.rng.random() < 0.15:
            n_cars = max(2, i // 2)
            for c in range(n_cars):
                self.addr(s, num=n, ftype="CARS", fnum=c + 1, xy=base_xy, gt=base_gt, shell=False,
                          role="carspace", legal=f"{i + c + 1}/{s.sp}")
        s.strata += [(b, "Vertical")] * (i + n_cars)
        self.strata_counts(s, vertical=i + n_cars)
        self.link(base, kids, prob=0.7)

    def sc_units_no_parent(self, s: Site):
        n, k = s.numbers[0], int(self.rng.integers(3, 9))
        b = s.rect(3, 6, s.w - 3, s.d - 8)
        self.building(s, b, 2)
        s.sp = self.plan("SP")
        for i in range(k):
            self.addr(s, num=n, ftype="UNIT", fnum=i + 1, xy=s.pc(), gt="PC", shell=False, role="unit",
                      legal=f"{i + 1}/{s.sp}")
        self.strata_counts(s, vertical=k)

    def sc_mixed_use(self, s: Site):
        n = s.numbers[0]
        b = s.rect(0.5, 0.5, s.w - 0.5, s.d - 10)
        self.building(s, b, 3, use="Commercial")
        stacked = self.rng.random() < 0.6
        base_xy, base_gt = s.fcs(), "FCS"
        s.sp = self.plan("SP")
        base = self.addr(s, num=n, xy=base_xy, gt=base_gt, shell=True, role="street_shell",
                         legal=f"CP/{s.sp}" if self.rng.random() < 0.5 else "")
        kids, lot = [], 0
        ground = self.rng.random() < 0.5
        for i in range(int(self.rng.integers(1, 4))):
            lot += 1
            xy, gt = (base_xy, base_gt) if stacked else (self.uc(b), "UC")
            kids.append(self.addr(s, num=n, ftype="SHOP", fnum=i + 1, ltype="G" if ground else "", xy=xy, gt=gt,
                                  shell=False, role="shop", legal=f"{lot}/{s.sp}"))
        with_level = self.rng.random() < 0.5
        for i in range(int(self.rng.integers(2, 9))):
            lot += 1
            xy, gt = (base_xy, base_gt) if stacked else (self.uc(b), "UC")
            kids.append(self.addr(s, num=n, ftype="UNIT", fnum=i + 1, ltype="L" if with_level else "",
                                  lnum=1 + i % 2 if with_level else None, xy=xy, gt=gt, shell=False,
                                  role="unit", legal=f"{lot}/{s.sp}"))
        self.strata_counts(s, vertical=lot)
        self.link(base, kids, prob=0.6)

    def sc_office(self, s: Site):
        n = s.numbers[0]
        floors = int(self.rng.integers(6, 13))
        b = s.rect(2, 4, s.w - 2, s.d - 4)
        self.building(s, b, floors + 1, use="Commercial")
        stacked = self.rng.random() < 0.7
        base_xy, base_gt = s.fcs(), "FCS"
        base = self.addr(s, num=n, xy=base_xy, gt=base_gt, shell=True, role="street_shell",
                         legal=f"{s.lot_no}/{s.dp}")

        def geo():
            return (base_xy, base_gt) if stacked else (self.uc(b), "UC")

        kids = []
        if self.rng.random() < 0.5:
            for i in range(int(self.rng.integers(1, 3))):
                xy, gt = geo()
                kids.append(self.addr(s, num=n, ftype="SHOP", fpre="G", fnum=i + 1, ltype="G", xy=xy, gt=gt,
                                      shell=False, role="shop"))
        for lv in range(1, floors + 1):
            if self.rng.random() < 0.4:   # floor split into suites
                if self.rng.random() < 0.5:
                    xy, gt = geo()
                    kids.append(self.addr(s, num=n, ltype="L", lnum=lv, xy=xy, gt=gt, shell=True, role="level_shell"))
                for j in range(int(self.rng.integers(2, 5))):
                    xy, gt = geo()
                    kids.append(self.addr(s, num=n, ftype="SE", fnum=lv * 100 + j + 1, ltype="L", lnum=lv,
                                          xy=xy, gt=gt, shell=False, role="suite"))
            else:                          # whole-floor tenancy
                xy, gt = geo()
                kids.append(self.addr(s, num=n, ltype="L", lnum=lv, xy=xy, gt=gt, shell=False, role="floor_tenancy"))
        self.link(base, kids, prob=0.5)

    def sc_highrise(self, s: Site):
        nums = s.numbers
        lo, hi = nums[0], nums[-1]
        variant = str(self.rng.choice(["named", "vanity", "numbered"]))
        n_towers = 2 if self.rng.random() < 0.7 else 3
        tower_ids = {"named": [f"{w} {i}" for w, i in zip(["TOWER"] * 3, "ABC")],
                     "vanity": TOWER_VANITY}.get(variant, ["", "", ""])
        if variant == "named" and self.rng.random() < 0.5:
            word = str(self.rng.choice(["BUILDING", "BLOCK"]))
            tower_ids = [f"{word} {i}" for i in ("123" if word == "BUILDING" else "ABC")]
        tower_nums = [lo, hi, nums[3]]
        with_level = self.rng.random() < 0.5
        s.sp = self.plan("SP")
        range_present = variant != "numbered" or self.rng.random() < 0.6
        range_xy = s.pc()
        prec = None
        if range_present:
            prec = self.addr(s, num=lo, last=hi, xy=range_xy, gt="PC", shell=True, role="precinct_shell",
                             legal=f"CP/{s.sp}" if self.rng.random() < 0.5 else "")
        lot, total = 0, 0
        span = s.w / n_towers
        for t in range(n_towers):
            u0 = t * span + (span - 22) / 2
            tower = s.rect(u0, 8, u0 + 22, 30)
            levels = int(self.rng.integers(8, 16))
            self.building(s, tower, levels)
            stacked = self.rng.random() < 0.6
            if variant == "numbered":
                num, last, bname = tower_nums[t], None, ""
            else:
                num, last, bname = lo, hi, tower_ids[t]
            container = None
            if variant == "numbered":
                container = self.addr(s, num=num, xy=self.bc(tower), gt="BC", shell=True, role="street_shell")
            elif self.rng.random() < 0.7:
                container = self.addr(s, num=num, last=last, bname=bname, xy=self.bc(tower), gt="BC", shell=True,
                                      role="building_shell")
            kids = []
            for lv in range(1, levels + 1):
                for j in range(int(self.rng.integers(2, 5))):
                    lot += 1
                    xy, gt = (self.bc(tower), "BC") if stacked else (self.uc(tower), "UC")
                    kids.append(self.addr(s, num=num, last=last, bname=bname, ftype="UNIT", fnum=lv * 100 + j + 1,
                                          ltype="L" if with_level else "", lnum=lv if with_level else None,
                                          xy=xy, gt=gt, shell=False, role="unit", legal=f"{lot}/{s.sp}",
                                          slip=not stacked))
            total += len(kids)
            parent = container or prec
            if parent:
                self.link(parent, kids, prob=0.7)
        shop_num, shop_last = (lo, hi) if range_present else (lo, None)
        for i in range(int(self.rng.integers(0, 5))):
            lot += 1
            self.addr(s, num=shop_num, last=shop_last, ftype="SHOP", fpre="G", fnum=i + 1, ltype="G",
                      xy=self.uc(s.rect(5, 2, s.w - 5, 7)), gt="UC", shell=False, role="shop", legal=f"{lot}/{s.sp}")
        if self.rng.random() < 0.2:
            for c in range(int(self.rng.integers(10, 21))):
                lot += 1
                self.addr(s, num=shop_num, last=shop_last, ftype="CARS", fnum=c + 1, xy=range_xy, gt="PC",
                          shell=False, role="carspace", legal=f"{lot}/{s.sp}")
        self.strata_counts(s, vertical=lot)

    def sc_range_townhouses(self, s: Site):
        nums = s.numbers
        k = len(nums)
        slot = s.w / k
        s.sp = self.plan("SP")
        self.addr(s, num=nums[0], last=nums[-1], xy=s.pc(), gt="PC", shell=True, role="range_shell",
                  legal=f"CP/{s.sp}" if self.rng.random() < 0.5 else "")
        for i, num in enumerate(nums):
            b = s.rect(i * slot + 1, 6, (i + 1) * slot - 1, 25)
            self.building(s, b, 2)
            s.strata.append((s.rect(i * slot, 0, (i + 1) * slot, s.d), "Horizontal"))
            self.addr(s, num=num, xy=self.bc(b), gt="BC", shell=False, role="dwelling", legal=f"{i + 1}/{s.sp}")
        self.strata_counts(s, horizontal=k, populated_prob=0.5)

    def sc_retirement(self, s: Site):
        n = s.numbers[0]
        community = s.rect(s.w / 2 - 12, 4, s.w / 2 + 12, 14)
        self.building(s, community, 1, use="Commercial")
        base = self.addr(s, num=n, xy=s.fcs(), gt="FCS", shell=True, role="village_shell", legal=f"{s.lot_no}/{s.dp}")
        kids, i = [], 0
        for row in range(2):
            for col in range(int(s.w // 11)):
                if self.rng.random() < 0.15:
                    continue
                i += 1
                b = s.rect(col * 11 + 1, 17 + row * 14, col * 11 + 10, 17 + row * 14 + 11)
                self.building(s, b, 1)
                kids.append(self.addr(s, num=n, ftype="VLLA", fnum=i, xy=self.bc(b), gt="BC", shell=False, role="villa"))
        self.link(base, kids, prob=0.5)

    def sc_corner(self, s: Site):
        n = s.numbers[0]
        self.building(s, s.rect(4, 7, s.w - 3, 25), self.rng.choice([1, 2]))
        main = self.addr(s, num=n, xy=s.fcs(), gt="FCS", shell=False, role="house", legal=f"{s.lot_no}/{s.dp}")
        cross_street, cross_num = s.cross
        is_alias = self.rng.random() < 0.5
        alias = self.addr(s, num=cross_num, street=cross_street, xy=s.xy(6, s.d / 2), gt="FCS", shell=False,
                          role="corner_alias" if is_alias else "corner_second_frontage",
                          alias="A" if is_alias else "P", legal=f"{s.lot_no}/{s.dp}")
        if is_alias:
            self.aliases.append((main, alias))

    def sc_split_precinct(self, s: Site):
        n = s.numbers[0]
        split = 20.0
        s.parcel_polys = [s.rect(0, 0, split, s.d), s.rect(split, 0, s.w, s.d)]
        b = s.rect(5, 6, s.w - 3, 38)
        self.building(s, b, 4)
        base = self.addr(s, num=n, xy=s.xy(split / 2, s.d / 2), gt="PC", shell=True, role="street_shell")
        kids = [self.addr(s, num=n, ftype="UNIT", fnum=i + 1, xy=self.uc(s.rect(split + 1, 6, s.w - 3, 38)), gt="UC",
                          shell=False, role="unit") for i in range(int(self.rng.integers(8, 21)))]
        self.link(base, kids, prob=0.7)

    # ------------------------------------------------------------ layout
    def layout(self, n_parcels: int) -> None:
        rng = self.rng
        n_blocks = max(1, math.ceil(n_parcels / 26))
        cols = max(1, math.ceil(math.sqrt(n_blocks * 1.6)))
        rows = math.ceil(n_blocks / cols)
        x_origin, y_origin = ORIGIN
        localities = [Locality(f"loc{h:012x}"[:15], name, pc)
                      for h, (name, pc) in zip((0x5b1e0c3a77, 0x6c2f1d4b88, 0x7d302e5c99), LOCALITIES)]
        self.localities = localities

        def locality_for(col: int) -> Locality:
            return localities[min(len(localities) - 1, col * len(localities) // cols)]

        ew = [Street(EW_NAMES[i % len(EW_NAMES)], EW_TYPES[i % len(EW_TYPES)]) for i in range(rows + 1)]
        ns = [Street(NS_NAMES[j % len(NS_NAMES)], "STREET" if j % 3 else "ROAD", "N" if j == 2 else "")
              for j in range(cols + 1)]
        self.streets = ew + ns
        counters = {}
        ns_counter = {j: 1 for j in range(cols + 1)}
        names, weights = zip(*[(k, v[0]) for k, v in SCENARIOS.items()])
        weights = np.array(weights) / sum(weights)

        # road parcels
        y_top, y_bot = y_origin, y_origin - rows * ROW_PITCH
        x_left, x_right = x_origin, x_origin + cols * COL_PITCH
        for i in range(rows + 1):
            y = y_origin - i * ROW_PITCH
            self._road(box(x_left - ROAD_HALF, y - ROAD_HALF, x_right + ROAD_HALF, y + ROAD_HALF))
        for j in range(cols + 1):
            x = x_origin + j * COL_PITCH
            self._road(box(x - ROAD_HALF, y_bot - ROAD_HALF, x + ROAD_HALF, y_top + ROAD_HALF))

        sid = itertools.count()
        for i in range(rows):
            for j in range(cols):
                loc = locality_for(j)
                bx0 = x_origin + j * COL_PITCH + ROAD_HALF
                strips = [  # (street, y_front, facing, parity)
                    (ew[i], y_origin - i * ROW_PITCH - ROAD_HALF, -1, 1),
                    (ew[i + 1], y_origin - (i + 1) * ROW_PITCH + ROAD_HALF, +1, 0),
                ]
                for street, y_front, facing, parity in strips:
                    street.by_locality.setdefault(loc.name, None)
                    dp = self.plan("DP")
                    u, lot_no = 0.0, 0
                    while u < BLOCK_LEN - 0.01:
                        remaining = BLOCK_LEN - u
                        if remaining < 12:
                            self._reserve(Site(next(sid), bx0 + u, remaining, y_front, LOT_DEPTH, facing, street,
                                               loc, []), dp)
                            break
                        for _ in range(20):
                            name = str(rng.choice(names, p=weights))
                            if u == 0 and rng.random() < 0.3:
                                name = "corner"
                            wmin, wmax = SCENARIOS[name][1]
                            if name == "corner" and u > 0:
                                continue
                            if wmin <= remaining:
                                break
                        else:
                            name, wmin, wmax = "house", min(remaining, 14), min(remaining, 18)
                        width = float(min(rng.uniform(wmin, wmax), remaining))
                        if remaining - width < 12:
                            width = remaining
                        if name == "range_townhouses":
                            n_nums = int(max(3, min(5, width // 9)))
                        else:
                            n_nums = SCENARIOS[name][2]
                        key = (id(street), parity)
                        start = counters.get(key, 1 if parity else 2)
                        numbers = [start + 2 * k for k in range(n_nums)]
                        counters[key] = start + 2 * n_nums + 2 * int(rng.random() < 0.1)
                        lot_no += 1
                        s = Site(next(sid), bx0 + u, width, y_front, LOT_DEPTH, facing, street, loc, numbers,
                                 scenario=name, lot_no=lot_no, dp=dp)
                        if name == "corner":
                            cross = ns[j]
                            cross.by_locality.setdefault(loc.name, None)
                            s.cross = (cross, ns_counter[j])
                            ns_counter[j] += 2 + 2 * int(rng.integers(0, 3))
                        getattr(self, f"sc_{name}")(s)
                        self._finish_site(s)
                        u += width

        # street-locality pids
        seq = itertools.count(2_800_001)
        for st in self.streets:
            for lname in list(st.by_locality):
                st.by_locality[lname] = f"{STATE}{next(seq)}"

    def _road(self, poly) -> None:
        self.parcels.append({"geometry": poly, "parcel_type": "Road", "base_parcel": None, "lot": None,
                             "plan": None, "strata_type": None, "counts": {}})

    def _reserve(self, s: Site, dp: str) -> None:
        self.parcels.append({"geometry": s.polygon, "parcel_type": "Lot", "base_parcel": "Yes",
                             "lot": "RES", "plan": dp, "strata_type": None, "counts": {}, "site": s.sid})

    def _finish_site(self, s: Site) -> None:
        self.sites.append(s)
        polys = s.parcel_polys or [s.polygon]
        for k, poly in enumerate(polys):
            self.parcels.append({
                "geometry": poly, "parcel_type": "Lot", "base_parcel": "Yes", "lot": str(s.lot_no + k * 1000),
                "plan": s.dp, "strata_type": None, "counts": s.counts if k == 0 else {}, "site": s.sid,
            })
        for poly, stype in s.strata:
            self.parcels.append({"geometry": poly, "parcel_type": "Strata", "base_parcel": None, "lot": None,
                                 "plan": s.sp, "strata_type": stype, "counts": {}, "site": s.sid})

    # ------------------------------------------------------------ outputs
    def write(self, out_dir: Path) -> dict:
        out_dir = Path(out_dir)
        gnaf_dir = out_dir / "G-NAF" / RELEASE / "Standard"
        auth_dir = out_dir / "G-NAF" / RELEASE / "Authority Code"
        for d in (gnaf_dir, auth_dir, out_dir / "Cadastre", out_dir / "Buildings", out_dir / "truth", out_dir / "text"):
            d.mkdir(parents=True, exist_ok=True)

        addr = pd.DataFrame(self.addresses)
        # retired twins (must be ignored by the loader)
        retired = addr.sample(frac=0.01, random_state=int(self.rng.integers(1 << 30))).copy()
        retired["pid"] = [f"GA{STATE}{next(self._addr_seq)}" for _ in range(len(retired))]
        retired["retired"] = True
        addr["retired"] = False
        all_addr = pd.concat([addr, retired], ignore_index=True)

        pts = gpd.GeoSeries([Point(x, y) for x, y in zip(all_addr["x"], all_addr["y"])], crs=LOCAL_CRS).to_crs(OUT_CRS)
        all_addr["lon"], all_addr["lat"] = pts.x.round(8), pts.y.round(8)
        self._write_gnaf(all_addr, gnaf_dir, auth_dir)
        parcels_gdf = self._write_cadastre(out_dir / "Cadastre" / f"{STATE}_CADASTRE.gpkg")
        self._write_buildings(addr, parcels_gdf, out_dir / "Buildings" / f"{STATE}_BUILDINGS.gpkg")

        truth = addr[["pid", "is_shell", "role", "scenario", "site"]].rename(columns={"pid": "address_detail_pid"})
        truth["is_shell"] = truth["is_shell"].astype(int)
        truth.to_csv(out_dir / "truth" / "shell_truth.csv", index=False)
        self._write_text(addr, all_addr, out_dir / "text" / "addresses_free_text.csv")
        return {
            "addresses": int(len(addr)), "shells": int(addr["is_shell"].sum()),
            "parcels": int(sum(1 for p in self.parcels if p["base_parcel"] == "Yes")),
            "buildings": int(len(self.buildings)), "scenarios": addr.groupby("scenario")["site"].nunique().to_dict(),
        }

    def _write_gnaf(self, a: pd.DataFrame, gnaf_dir: Path, auth_dir: Path) -> None:
        rng = self.rng
        n = len(a)

        def s(v):
            return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)

        created = pd.to_datetime("2004-01-01") + pd.to_timedelta(rng.integers(0, 7000, n), unit="D")
        detail = pd.DataFrame({
            "ADDRESS_DETAIL_PID": a["pid"],
            "DATE_CREATED": created.strftime("%Y-%m-%d"),
            "DATE_LAST_MODIFIED": (created + pd.to_timedelta(rng.integers(0, 900, n), unit="D")).strftime("%Y-%m-%d"),
            "DATE_RETIRED": np.where(a["retired"], "2025-11-20", ""),
            "BUILDING_NAME": a["building_name"],
            "LOT_NUMBER_PREFIX": "", "LOT_NUMBER": "", "LOT_NUMBER_SUFFIX": "",
            "FLAT_TYPE_CODE": a["flat_type"],
            "FLAT_NUMBER_PREFIX": a["flat_prefix"],
            "FLAT_NUMBER": a["flat_number"].map(s),
            "FLAT_NUMBER_SUFFIX": a["flat_suffix"],
            "LEVEL_TYPE_CODE": a["level_type"],
            "LEVEL_NUMBER_PREFIX": "",
            "LEVEL_NUMBER": a["level_number"].map(s),
            "LEVEL_NUMBER_SUFFIX": "",
            "NUMBER_FIRST_PREFIX": "",
            "NUMBER_FIRST": a["number_first"].map(s),
            "NUMBER_FIRST_SUFFIX": a["number_first_suffix"],
            "NUMBER_LAST_PREFIX": "",
            "NUMBER_LAST": a["number_last"].map(s),
            "NUMBER_LAST_SUFFIX": "",
            "STREET_LOCALITY_PID": [st.by_locality[loc.name] for st, loc in zip(a["street"], a["locality"])],
            "LOCATION_DESCRIPTION": "",
            "LOCALITY_PID": [loc.pid for loc in a["locality"]],
            "ALIAS_PRINCIPAL": a["alias"],
            "POSTCODE": [loc.postcode for loc in a["locality"]],
            "PRIVATE_STREET": "",
            "LEGAL_PARCEL_ID": a["legal"],
            "CONFIDENCE": rng.choice(["0", "1", "2"], n, p=[0.2, 0.5, 0.3]),
            "ADDRESS_SITE_PID": a["address_site_pid"],
            "LEVEL_GEOCODED_CODE": "7",
            "PROPERTY_PID": "",
            "GNAF_PROPERTY_PID": "",
            "PRIMARY_SECONDARY": a["ps"],
        })
        for col in ("FLAT_NUMBER", "LEVEL_NUMBER", "NUMBER_FIRST", "NUMBER_LAST"):
            detail[col] = detail[col].str.replace(r"\.0$", "", regex=True)
        detail.to_csv(gnaf_dir / f"{STATE}_ADDRESS_DETAIL_psv.psv", sep="|", index=False)

        geo = pd.DataFrame({
            "ADDRESS_DEFAULT_GEOCODE_PID": np.arange(15_000_001, 15_000_001 + n).astype(str),
            "DATE_CREATED": detail["DATE_CREATED"], "DATE_RETIRED": detail["DATE_RETIRED"],
            "ADDRESS_DETAIL_PID": a["pid"], "GEOCODE_TYPE_CODE": a["geocode_type"],
            "LONGITUDE": a["lon"].map(lambda v: f"{v:.8f}"), "LATITUDE": a["lat"].map(lambda v: f"{v:.8f}"),
        })
        geo.to_csv(gnaf_dir / f"{STATE}_ADDRESS_DEFAULT_GEOCODE_psv.psv", sep="|", index=False)

        street_rows = []
        for st in self.streets:
            for lname, pid in st.by_locality.items():
                loc = next(l for l in self.localities if l.name == lname)
                street_rows.append({
                    "STREET_LOCALITY_PID": pid, "DATE_CREATED": "2004-04-29", "DATE_RETIRED": "",
                    "STREET_CLASS_CODE": "C", "STREET_NAME": st.name, "STREET_TYPE_CODE": st.type_code,
                    "STREET_SUFFIX_CODE": st.suffix, "LOCALITY_PID": loc.pid, "GNAF_STREET_PID": pid[3:],
                    "GNAF_STREET_CONFIDENCE": "2", "GNAF_RELIABILITY_CODE": "4",
                })
        pd.DataFrame(street_rows).to_csv(gnaf_dir / f"{STATE}_STREET_LOCALITY_psv.psv", sep="|", index=False)

        pd.DataFrame([{
            "LOCALITY_PID": l.pid, "DATE_CREATED": "2004-04-29", "DATE_RETIRED": "", "LOCALITY_NAME": l.name,
            "PRIMARY_POSTCODE": l.postcode, "LOCALITY_CLASS_CODE": "G", "STATE_PID": aut.STATE_PIDS[STATE],
            "GNAF_LOCALITY_PID": str(500_000 + k), "GNAF_RELIABILITY_CODE": "5",
        } for k, l in enumerate(self.localities)]).to_csv(gnaf_dir / f"{STATE}_LOCALITY_psv.psv", sep="|", index=False)

        pd.DataFrame([{"STATE_PID": aut.STATE_PIDS[STATE], "DATE_CREATED": "2004-04-29", "DATE_RETIRED": "",
                       "STATE_NAME": aut.STATES[STATE], "STATE_ABBREVIATION": STATE}]
                     ).to_csv(gnaf_dir / f"{STATE}_STATE_psv.psv", sep="|", index=False)

        pd.DataFrame([{
            "PRIMARY_SECONDARY_PID": str(9_000_001 + k), "PRIMARY_PID": p, "SECONDARY_PID": sec,
            "DATE_CREATED": "2016-02-17", "DATE_RETIRED": "", "PS_JOIN_TYPE_CODE": "2" if k % 5 else "1",
            "PS_JOIN_COMMENT": "",
        } for k, (p, sec) in enumerate(self.ps_links)],
            columns=["PRIMARY_SECONDARY_PID", "PRIMARY_PID", "SECONDARY_PID", "DATE_CREATED", "DATE_RETIRED",
                     "PS_JOIN_TYPE_CODE", "PS_JOIN_COMMENT"],
        ).to_csv(gnaf_dir / f"{STATE}_PRIMARY_SECONDARY_psv.psv", sep="|", index=False)

        pd.DataFrame([{
            "ADDRESS_ALIAS_PID": str(8_000_001 + k), "DATE_CREATED": "2012-08-01", "DATE_RETIRED": "",
            "PRINCIPAL_PID": p, "ALIAS_PID": al, "ALIAS_TYPE_CODE": "SYN", "ALIAS_COMMENT": "",
        } for k, (p, al) in enumerate(self.aliases)],
            columns=["ADDRESS_ALIAS_PID", "DATE_CREATED", "DATE_RETIRED", "PRINCIPAL_PID", "ALIAS_PID",
                     "ALIAS_TYPE_CODE", "ALIAS_COMMENT"],
        ).to_csv(gnaf_dir / f"{STATE}_ADDRESS_ALIAS_psv.psv", sep="|", index=False)

        def aut_file(name, rows):
            pd.DataFrame(rows, columns=["CODE", "NAME", "DESCRIPTION"]).to_csv(
                auth_dir / f"Authority_Code_{name}_psv.psv", sep="|", index=False)

        aut_file("FLAT_TYPE_AUT", [(c, n_, n_) for c, (n_, _) in aut.FLAT_TYPES.items()])
        aut_file("LEVEL_TYPE_AUT", [(c, n_, n_) for c, n_ in aut.LEVEL_TYPES.items()])
        aut_file("STREET_TYPE_AUT", [(c, ab, c) for c, ab in aut.STREET_TYPES.items()])
        aut_file("STREET_SUFFIX_AUT", [(c, n_, n_) for c, n_ in aut.STREET_SUFFIXES.items()])
        aut_file("GEOCODE_TYPE_AUT", [(c, n_, n_) for c, (n_, _) in aut.GEOCODE_TYPES.items()])
        aut_file("ALIAS_TYPE_AUT", [(c, n_, n_) for c, n_ in aut.ALIAS_TYPES.items()])
        aut_file("PS_JOIN_TYPE_AUT", [(c, n_, n_) for c, n_ in aut.PS_JOIN_TYPES.items()])
        aut_file("STREET_CLASS_AUT", [("C", "CONFIRMED", "CONFIRMED"), ("U", "UNCONFIRMED", "UNCONFIRMED")])

    def _write_cadastre(self, path: Path) -> gpd.GeoDataFrame:
        p = pd.DataFrame(self.parcels)
        gdf = gpd.GeoDataFrame(p.drop(columns=["geometry", "counts"]), geometry=list(p["geometry"]), crs=LOCAL_CRS)
        n = len(gdf)
        gdf["cadastre_polygon_pid"] = [self.hexpid("cdp") for _ in range(n)]
        gdf["cadastre_pid"] = [self.hexpid("cad") for _ in range(n)]
        lot = gdf["lot"].fillna("")
        plan = gdf["plan"].fillna("")
        gdf["contributor_id"] = np.where(plan != "", lot + "//" + plan, "")
        gdf["date_created"] = "2021-05-14"
        gdf["date_modified"] = "2026-08-01"
        gdf["state"] = STATE
        gdf["section"] = ""
        gdf["parcel_id"] = gdf["contributor_id"]
        gdf["source"] = STATE
        gdf["contributor_status"] = np.where(gdf["parcel_type"] == "Road", "Gazetted", "Registered")
        gdf["title_status"] = np.where(gdf["parcel_type"] == "Road", "Not Titled", "Titled")
        counts = p["counts"]
        gdf["total_strata_count"] = [c.get("total") if c else None for c in counts]
        gdf["horizontal_strata_count"] = [c.get("horizontal") if c else None for c in counts]
        gdf["vertical_strata_count"] = [c.get("vertical") if c else None for c in counts]
        gdf["area"] = gdf.to_crs(AREA_CRS).area.round(2)
        cols = ["cadastre_polygon_pid", "cadastre_pid", "contributor_id", "date_created", "date_modified", "state",
                "lot", "plan", "section", "parcel_id", "source", "contributor_status", "title_status", "parcel_type",
                "base_parcel", "strata_type", "total_strata_count", "horizontal_strata_count",
                "vertical_strata_count", "area"]
        keep = gdf[cols + ["site", "geometry"]].copy()
        out = keep[cols + ["geometry"]].to_crs(OUT_CRS)
        for col in ("total_strata_count", "horizontal_strata_count", "vertical_strata_count"):
            out[col] = out[col].astype("Int64")
        out.to_file(path, layer="cadastre", driver="GPKG", engine="pyogrio")
        return keep

    def _write_buildings(self, addr: pd.DataFrame, parcels: gpd.GeoDataFrame, path: Path) -> None:
        b = pd.DataFrame(self.buildings)
        b = b[b["captured"]].reset_index(drop=True)
        gdf = gpd.GeoDataFrame(b.drop(columns=["geometry"]), geometry=list(b["geometry"]), crs=LOCAL_CRS)
        rng = self.rng
        n = len(gdf)
        area = gdf.area
        site_loc = {s.sid: s.locality for s in self.sites}

        # building -> base parcel by largest overlap
        base = parcels[parcels["base_parcel"] == "Yes"]
        inter = gpd.overlay(gdf[["pid", "geometry"]], base[["cadastre_pid", "geometry"]], how="intersection",
                            keep_geom_type=True)
        inter["_a"] = inter.area
        bcad = inter.sort_values("_a", ascending=False).drop_duplicates("pid")

        # building_address: active principal addresses
        active = addr[addr["alias"] == "P"].reset_index(drop=True)
        apts = gpd.GeoDataFrame(active[["pid", "site"]], geometry=gpd.points_from_xy(active["x"], active["y"]),
                                crs=LOCAL_CRS)
        hit = gpd.sjoin(apts, gdf[["pid", "geometry"]].rename(columns={"pid": "bpid"}), predicate="within")
        hit = hit.drop_duplicates("pid")
        links = [(r.pid, r.bpid, 95) for r in hit.itertuples()]
        linked = set(hit["pid"])
        largest = (gdf.assign(_a=area).sort_values("_a", ascending=False).drop_duplicates("site")
                   .set_index("site")["pid"])
        for r in active[~active["pid"].isin(linked)].itertuples():
            if r.site in largest.index:
                links.append((r.pid, largest[r.site], 80))
        label = pd.Series(_labels(addr).to_numpy(), index=addr["pid"].to_numpy())
        ba = pd.DataFrame(links, columns=["address_detail_pid", "building_pid_internal", "relationship_confidence"])
        ba["building_address_pid"] = [self.hexpid("bla") for _ in range(len(ba))]
        ba["building_pid"] = ba["building_pid_internal"]
        ba["address_pid"] = ["GAP" + p[2:] for p in ba["address_detail_pid"]]
        ba["date_created"] = "2024-02-01"
        ba["date_modified"] = "2026-08-01"
        ba["address"] = ba["address_detail_pid"].map(label)
        ba["state"] = STATE
        ba["primary_building"] = "Yes"
        ba["primary_building_confidence"] = ba["relationship_confidence"]
        ba = ba[["building_address_pid", "building_pid", "address_pid", "address_detail_pid", "date_created",
                 "date_modified", "address", "state", "relationship_confidence", "primary_building",
                 "primary_building_confidence"]]

        counts = ba.groupby("building_pid").size()
        cent = gdf.geometry.centroid.to_crs(OUT_CRS)
        out = gpd.GeoDataFrame({
            "building_pid": gdf["pid"],
            "date_created": "2020-03-01",
            "date_modified": "2026-08-01",
            "date_first_captured": "2019-11-12",
            "date_last_captured": "2025-09-30",
            "capture_resolution": "0.10",
            "capture_method": "Aerial",
            "locality_pid": [site_loc[s].pid for s in gdf["site"]],
            "locality_name": [site_loc[s].name for s in gdf["site"]],
            "lga_pid": "lgasynth0000001",
            "lga_name": "SYNTHETIC CITY COUNCIL",
            "mesh_block_code": [f"1{rng.integers(10**9, 10**10)}" for _ in range(n)],
            "sa1_code": [f"1{rng.integers(10**9, 10**10)}" for _ in range(n)],
            "state": STATE,
            "building_use": gdf["use"],
            "building_use_source": "Planning Zone",
            "address_count": gdf["pid"].map(counts).fillna(0).astype(int),
            "primary_building_flag": np.where(gdf["pid"].isin(largest.values), "Yes", "No"),
            "ground_elevation": np.round(rng.uniform(20, 60, n), 1),
            "ground_elevation_resolution": "1.0",
            "building_area": area.round(2),
            "total_floor_area": (area * gdf["levels"] * rng.uniform(0.92, 1.0, n)).round(2),
            "number_vertices": 4,
            "centroid_longitude": cent.x.round(8),
            "centroid_latitude": cent.y.round(8),
        }, geometry=gdf.geometry, crs=LOCAL_CRS).to_crs(OUT_CRS)
        out.to_file(path, layer="buildings", driver="GPKG", engine="pyogrio")
        pyogrio.write_dataframe(ba, path, layer="building_address", driver="GPKG", append=False)

        bc = pd.DataFrame({
            "building_cadastre_pid": [self.hexpid("blc") for _ in range(len(bcad))],
            "building_pid": bcad["pid"].to_numpy(),
            "cadastre_pid": bcad["cadastre_pid"].to_numpy(),
            "date_created": "2024-02-01", "date_modified": "2026-08-01", "state": STATE,
            "relationship_confidence": 90,
        })
        pyogrio.write_dataframe(bc, path, layer="building_cadastre", driver="GPKG", append=False)

    def _write_text(self, addr: pd.DataFrame, all_addr: pd.DataFrame, path: Path) -> None:
        """Free-text rendering of each active address in a mix of real-world styles."""
        rng = self.rng
        a = addr.merge(all_addr[["pid", "lon", "lat"]], on="pid")
        rows = []
        for r in a.itertuples():
            rows.append({"address_pid": r.pid, "address": _free_text(r, rng), "longitude": r.lon, "latitude": r.lat})
        pd.DataFrame(rows).to_csv(path, index=False)


def _num(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(int(v))


def _labels(addr: pd.DataFrame) -> pd.Series:
    out = []
    for r in addr.itertuples():
        out.append(_free_text(r, None))
    return pd.Series(out, index=addr.index)


def _free_text(r, rng) -> str:
    """Render one address; ``rng=None`` gives the canonical G-NAF-like label."""
    street_type = r.street.type_code
    style = "label" if rng is None else str(rng.choice(["label", "abbrev", "slash", "lower"], p=[0.35, 0.3, 0.2, 0.15]))
    if style in ("abbrev", "slash", "lower"):
        street_type = aut.STREET_TYPES.get(street_type, street_type)
    number = _num(r.number_first) + (r.number_first_suffix or "")
    if r.number_last is not None and not (isinstance(r.number_last, float) and math.isnan(r.number_last)):
        number += "-" + _num(r.number_last)
    flat_name = aut.FLAT_TYPES.get(r.flat_type, (r.flat_type,))[0] if r.flat_type else ""
    flat_id = f"{r.flat_prefix}{_num(r.flat_number)}{r.flat_suffix}"
    level_name = aut.LEVEL_TYPES.get(r.level_type, r.level_type) if r.level_type else ""
    level_id = _num(r.level_number)
    if style == "abbrev":
        flat_name = {"UNIT": "U", "APARTMENT": "APT", "SUITE": "STE", "TOWNHOUSE": "TH", "CARSPACE": "CS"}.get(flat_name, flat_name)
        level_name = {"LEVEL": "LVL", "GROUND FLOOR": "GF"}.get(level_name, level_name)
    street = " ".join(p for p in [r.street.name, street_type, r.street.suffix] if p)
    place = f"{r.locality.name} {STATE} {r.locality.postcode}"
    flat = " ".join(p for p in [flat_name, flat_id] if p)
    level = " ".join(p for p in [level_name, level_id] if p)
    if style == "slash" and flat_id and not level and not r.building_name and flat_name in ("UNIT", ""):
        text = f"{flat_id}/{number} {street} {place}"
    elif style == "abbrev" and flat_name in ("U", "APT", "TH") and flat_id and not r.building_name:
        text = ", ".join(p for p in [f"{flat_name}{flat_id}", level, f"{number} {street}", place] if p)
    else:
        sep = ", " if style == "label" else " "
        text = sep.join(p for p in [r.building_name, flat, level, f"{number} {street}", place] if p)
    if style == "lower":
        text = text.title()
    return text


def generate(out_dir: str | Path, n_parcels: int = 1500, seed: int = 42) -> dict:
    """Generate the synthetic datasets into ``out_dir`` and return a summary dict."""
    builder = Builder(seed=seed)
    builder.layout(n_parcels)
    return builder.write(Path(out_dir))

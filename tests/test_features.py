"""Behaviour on hand-made parcels covering the main shell patterns."""


from conftest import addr


def test_unit_block_bare_address_is_shell(case):
    case.parcel("P1", 0, 0, 30, 40).building("B1", 3, 5, 27, 35, levels=3)
    case.add(addr("BASE", 15, 20, num=100))
    case.add(*[addr(f"U{i}", 15, 20, num=100, flat_type="UNIT", flat=i) for i in range(1, 11)])
    case.run()
    assert case.p("BASE") >= 0.8
    assert all(case.p(f"U{i}") < 0.05 for i in range(1, 11))
    assert case.row("U3")["parent_pid"] == "BASE"
    assert "nested sub-address" in case.row("BASE")["reasons"]


def test_single_house_is_not_shell(case):
    case.parcel("P1", 0, 0, 15, 40).building("B1", 2, 7, 13, 25)
    case.add(addr("H", 7.5, 6, num=12, geocode="FCS")).run()
    assert case.p("H") < 0.02


def test_suffix_pair_with_two_buildings_base_is_shell(case):
    # 100A and 100B each have a building; 100 sits at the parcel centroid between them.
    case.parcel("P1", 0, 0, 20, 45).building("BA", 1.5, 7, 9.25, 26).building("BB", 10.75, 7, 18.5, 26)
    case.add(addr("BASE", 10, 22.5, num=100), addr("A", 5, 15, num=100, sfx="A", geocode="BC"),
             addr("B", 15, 15, num=100, sfx="B", geocode="BC")).run()
    assert case.p("BASE") >= 0.6
    assert case.p("A") < 0.1 and case.p("B") < 0.1


def test_suffix_triplet_with_own_building_base_is_dwelling(case):
    case.parcel("P1", 0, 0, 24, 45)
    for i, pid in enumerate(["B0", "BA", "BB"]):
        case.building(pid, 1 + i * 7.5 + 0.5, 7, 1 + (i + 1) * 7.5 - 0.5, 25)
    case.add(addr("BASE", 4.75, 16, num=100, geocode="BC"), addr("A", 12.25, 16, num=100, sfx="A", geocode="BC"),
             addr("B", 19.75, 16, num=100, sfx="B", geocode="BC")).run()
    assert case.p("BASE") < 0.2


def test_suffix_ambiguity_resolved_by_strata_count(case):
    case.parcel("P1", 0, 0, 20, 45, total_strata_count=3)
    case.add(addr("BASE", 10, 22.5, num=100), addr("A", 5, 15, num=100, sfx="A"),
             addr("B", 15, 15, num=100, sfx="B")).run()
    with_three_lots = case.p("BASE")
    case.parcels[0]["total_strata_count"] = 2
    case.run()
    assert case.p("BASE") > with_three_lots + 0.3


def test_range_address_covering_numbers_is_shell(case):
    case.parcel("P1", 0, 0, 40, 45)
    case.add(addr("RANGE", 20, 22, num=2, last=8))
    for i, n in enumerate([2, 4, 6, 8]):
        case.building(f"B{n}", i * 10 + 1, 6, i * 10 + 9, 25)
        case.add(addr(f"N{n}", i * 10 + 5, 15, num=n, geocode="BC"))
    case.run()
    assert case.p("RANGE") >= 0.8
    assert all(case.p(f"N{n}") < 0.1 for n in [2, 4, 6, 8])
    assert case.row("N4")["parent_pid"] == "RANGE"


def test_level_with_suites_is_shell_but_whole_floor_is_not(case):
    case.parcel("P1", 0, 0, 30, 40).building("B1", 2, 4, 28, 36, levels=8)
    case.add(addr("BASE", 15, 6, num=25),
             addr("L3", 15, 6, num=25, level_type="L", level=3),
             addr("S301", 15, 6, num=25, level_type="L", level=3, flat_type="SE", flat=301),
             addr("S302", 15, 6, num=25, level_type="L", level=3, flat_type="SE", flat=302),
             addr("L4", 15, 6, num=25, level_type="L", level=4)).run()
    assert case.p("BASE") >= 0.8
    assert case.p("L3") >= 0.8
    assert case.p("L4") < 0.1
    assert case.row("S301")["parent_pid"] == "L3"


def test_tower_building_shells_in_precinct(case):
    case.parcel("P1", 0, 0, 90, 45).building("TA", 5, 8, 27, 30, levels=12).building("TB", 50, 8, 72, 30, levels=12)
    case.add(addr("PRECINCT", 45, 22, num=1, last=15),
             addr("TOWER_A", 16, 19, num=1, last=15, building="TOWER A"),
             addr("TOWER_B", 61, 19, num=1, last=15, building="TOWER B"))
    for t, x in (("A", 16), ("B", 61)):
        case.add(*[addr(f"{t}{u}", x, 19, num=1, last=15, building=f"TOWER {t}", flat_type="UNIT", flat=u,
                        level_type="L", level=u // 100) for u in (101, 102, 201, 202, 301)])
    case.run()
    for pid in ("PRECINCT", "TOWER_A", "TOWER_B"):
        assert case.p(pid) >= 0.8, pid
    assert case.row("A201")["parent_pid"] == "TOWER_A"
    assert case.row("TOWER_A")["parent_pid"] == "PRECINCT"
    assert case.p("A201") < 0.05


def test_vanity_names_become_buildings_when_several_share_a_number(case):
    case.parcel("P1", 0, 0, 90, 45).building("T1", 5, 8, 27, 30, levels=10).building("T2", 50, 8, 72, 30, levels=10)
    case.add(addr("ASH", 16, 19, num=1, last=15, building="THE ASH"),
             addr("BIRCH", 61, 19, num=1, last=15, building="THE BIRCH"))
    for name, x in (("THE ASH", 16), ("THE BIRCH", 61)):
        case.add(*[addr(f"{name}{u}", x, 19, num=1, last=15, building=name, flat_type="UNIT", flat=u)
                   for u in range(1, 6)])
    case.run()
    assert case.p("ASH") >= 0.8 and case.p("BIRCH") >= 0.8
    assert case.row("THE BIRCH3")["parent_pid"] == "BIRCH"


def test_single_vanity_name_is_just_a_label(case):
    case.parcel("P1", 0, 0, 30, 40).building("B1", 3, 5, 27, 35, levels=4)
    case.add(addr("BASE", 15, 20, num=20, building="THE ARCADIA"))
    case.add(*[addr(f"U{i}", 15, 20, num=20, flat_type="UNIT", flat=i, building="THE ARCADIA" if i % 2 else "")
               for i in range(1, 7)])
    case.run()
    assert case.p("BASE") >= 0.8
    assert case.row("U2")["parent_pid"] == "BASE" and case.row("U3")["parent_pid"] == "BASE"


def test_corner_lot_with_two_frontages_is_not_shell(case):
    case.parcel("P1", 0, 0, 20, 45).building("B1", 4, 7, 17, 25)
    case.add(addr("MAIN", 10, 6, num=2, geocode="FCS"),
             addr("SIDE", 6, 22, street="JONES", num=45, geocode="FCS")).run()
    assert case.p("MAIN") < 0.05 and case.p("SIDE") < 0.05


def test_alias_does_not_create_children(case):
    case.parcel("P1", 0, 0, 20, 45).building("B1", 4, 7, 17, 25)
    case.add(addr("MAIN", 10, 15, num=10),
             addr("ALIAS", 10, 15, num=10, flat_type="UNIT", flat=1, alias="A")).run()
    assert case.p("MAIN") < 0.05
    assert case.row("ALIAS")["f_alias"] == 1


def test_rear_dwelling_does_not_make_house_a_shell(case):
    case.parcel("P1", 0, 0, 18, 45).building("HOUSE", 2, 7, 16, 24).building("REAR", 8, 34, 16, 42)
    case.add(addr("HOUSE_ADDR", 9, 15, num=10), addr("REAR_ADDR", 12, 38, num=10, flat_type="REAR", geocode="BC")).run()
    assert case.p("HOUSE_ADDR") < 0.1


def test_children_geocoded_on_neighbouring_parcel_still_count(case):
    case.parcel("PA", 0, 0, 20, 45).parcel("PB", 20, 0, 50, 45).building("B1", 5, 6, 47, 38, levels=4)
    case.add(addr("BASE", 10, 22.5, num=40))
    case.add(*[addr(f"U{i}", 22 + 2 * i, 20, num=40, flat_type="UNIT", flat=i, geocode="UC") for i in range(1, 11)])
    case.run()
    assert case.row("BASE")["n_offparcel_desc"] == 10
    assert case.p("BASE") >= 0.6


def test_gnaf_primary_link_raises_score(case):
    case.parcel("P1", 0, 0, 20, 45).building("B1", 2, 7, 18, 26)
    case.add(addr("BASE", 10, 22, num=7), addr("U1", 10, 22, num=7, flat_type="UNIT", flat=1))
    case.run()
    without = case.p("BASE")
    case.addresses[0]["primary_secondary"] = "P"
    case.addresses[0]["n_ps_secondaries"] = 1
    case.run()
    assert case.p("BASE") > without

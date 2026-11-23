import pytest

from shell_identifier.components import split_building_name
from shell_identifier.parser import parse_address

CASES = [
    ("UNIT 1205, LEVEL 12, TOWER B, THE QUAY, 1-15 SMITH STREET SOUTHBANK VIC 3006",
     dict(flat_type_code="UNIT", flat_number="1205", level_type_code="L", level_number="12",
          building_name="TOWER B THE QUAY", number_first="1", number_last="15", street_name="SMITH",
          street_type_code="STREET", locality_name="SOUTHBANK", state="VIC", postcode="3006")),
    ("3/25 Elizabeth St, Sydney NSW 2000",
     dict(flat_number="3", number_first="25", street_name="ELIZABETH", street_type_code="STREET",
          locality_name="SYDNEY", state="NSW", postcode="2000")),
    ("G01/25-27 ELIZABETH ST", dict(flat_number_prefix="G", flat_number="01", number_first="25", number_last="27")),
    ("Shop 2 Ground Floor 25 Elizabeth Street", dict(flat_type_code="SHOP", flat_number="2", level_type_code="G")),
    ("APT 3.04 25 SMITH ST", dict(flat_type_code="APT", flat_number="3.04", number_first="25")),
    ("U5 100 TEST ST", dict(flat_type_code="UNIT", flat_number="5", number_first="100")),
    ("SHOP 5/25 HIGH ST", dict(flat_type_code="SHOP", flat_number="5", number_first="25")),
    ("LOT 12 SMITH RD", dict(lot_number="12", number_first="", street_type_code="ROAD")),
    ("REAR 10 SMITH ST", dict(flat_type_code="REAR", flat_number="", number_first="10")),
    ("100A TEST ST", dict(number_first="100", number_first_suffix="A")),
    ("25 SMITH ST ST KILDA VIC 3182", dict(street_name="SMITH", street_type_code="STREET", locality_name="ST KILDA")),
    ("10 GROVE HILL RD TESTWOOD", dict(street_name="GROVE HILL", street_type_code="ROAD", locality_name="TESTWOOD")),
    ("10 ST JOHNS AVE", dict(street_name="ST JOHNS", street_type_code="AVENUE")),
    ("NORTH TOWER 1 SMITH ST", dict(building_name="TOWER NORTH", number_first="1")),
    ("A BLOCK, 1 SMITH ST", dict(building_name="BLOCK A")),
    ("THE ARCADIA, 100 TEST ST", dict(building_name="THE ARCADIA", number_first="100")),
    ("1205 THE QUAY, 1-15 SMITH ST", dict(flat_number="1205", building_name="THE QUAY", number_first="1", number_last="15")),
    ("SUITE 301 LEVEL 3 25 ELIZABETH ST", dict(flat_type_code="SE", flat_number="301", level_number="3")),
    ("1ST FLOOR 25 ELIZABETH ST", dict(level_type_code="FL", level_number="1")),
    ("CAR SPACE 12 100 TEST ST", dict(flat_type_code="CARS", flat_number="12")),
    ("10 SMITH STREET NORTH, TESTWOOD", dict(street_suffix_code="N", locality_name="TESTWOOD")),
    ("10 SMITH ST NORTH SYDNEY NSW", dict(street_suffix_code="", locality_name="NORTH SYDNEY")),
    ("PENTHOUSE 2 100 TEST ST", dict(flat_type_code="PTHS", flat_number="2")),
    ("L2/100 TEST ST", dict(level_type_code="L", level_number="2", flat_number="")),
    ("BUILDING C, UNIT 5, 100 TEST ST", dict(building_name="BUILDING C", flat_number="5")),
    ("25 - 27 SMITH ST", dict(number_first="25", number_last="27")),
    ("Unit 5 / 25 O'Connell Street, Mockton Park NSW 2997",
     dict(flat_number="5", number_first="25", street_name="O'CONNELL", locality_name="MOCKTON PARK")),
]


@pytest.mark.parametrize("text,expected", CASES, ids=[c[0][:40] for c in CASES])
def test_parse(text, expected):
    res = parse_address(text)
    for key, value in expected.items():
        assert res[key] == value, f"{key}: {res[key]!r} != {value!r}"


def test_gazetteer_resolves_locality_that_looks_like_a_street_type():
    no_gaz = parse_address("10 HIGH ST PARK RIDGE QLD")
    with_gaz = parse_address("10 HIGH ST PARK RIDGE QLD", localities={"PARK RIDGE"})
    assert with_gaz["street_name"] == "HIGH" and with_gaz["locality_name"] == "PARK RIDGE"
    assert no_gaz["street_name"] == "HIGH"  # the scoring heuristic also gets this one


def test_no_number_warns():
    res = parse_address("THE ESPLANADE")
    assert "no street number found" in res["parse_warnings"]


@pytest.mark.parametrize("name,expected", [
    ("THE QUAY TOWER A", ("TOWER A", "THE QUAY")),
    ("NORTH TOWER", ("TOWER NORTH", "")),
    ("BLDG 2", ("BUILDING 2", "")),
    ("THE ARCADIA", ("", "THE ARCADIA")),
    ("TOWER THE PINNACLE", ("", "TOWER THE PINNACLE")),
])
def test_split_building_name(name, expected):
    assert split_building_name(name) == expected

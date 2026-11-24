"""Authority code tables modelled on the G-NAF ``*_AUT`` tables (AS4590 based).

Each flat type carries a *category* used when weighing nested sub-addresses:

* ``dwelling``   - a self-contained residence (UNIT, APT, TNHS, VLLA ...)
* ``commercial`` - a business tenancy (SHOP, SUITE, OFFICE ...)
* ``ancillary``  - not an occupancy in its own right (CARSPACE, STORE, ATM ...)
* ``secondary``  - a secondary dwelling that implies the main dwelling is still
                   occupied (REAR, COTTAGE, BUNGALOW)
* ``structure``  - a building inside a complex (TOWER, BUILDING, BLOCK); these are
                   treated as a building component, not a flat
"""

FLAT_TYPES: dict[str, tuple[str, str]] = {
    "ANT": ("ANTENNA", "ancillary"),
    "APT": ("APARTMENT", "dwelling"),
    "ATM": ("AUTOMATED TELLER MACHINE", "ancillary"),
    "BBQ": ("BARBECUE", "ancillary"),
    "BLCK": ("BLOCK", "structure"),
    "BLDG": ("BUILDING", "structure"),
    "BNGW": ("BUNGALOW", "secondary"),
    "BTSD": ("BOATSHED", "ancillary"),
    "CAGE": ("CAGE", "ancillary"),
    "CARP": ("CARPARK", "ancillary"),
    "CARS": ("CARSPACE", "ancillary"),
    "CLUB": ("CLUB", "commercial"),
    "COOL": ("COOLROOM", "ancillary"),
    "CTGE": ("COTTAGE", "secondary"),
    "DUPL": ("DUPLEX", "dwelling"),
    "FCTY": ("FACTORY", "commercial"),
    "FLAT": ("FLAT", "dwelling"),
    "GRGE": ("GARAGE", "ancillary"),
    "HALL": ("HALL", "commercial"),
    "HSE": ("HOUSE", "dwelling"),
    "KSK": ("KIOSK", "commercial"),
    "LBBY": ("LOBBY", "ancillary"),
    "LOFT": ("LOFT", "dwelling"),
    "LOT": ("LOT", "dwelling"),
    "LSE": ("LEASE", "commercial"),
    "MBTH": ("MARINE BERTH", "ancillary"),
    "MSNT": ("MAISONETTE", "dwelling"),
    "OFFC": ("OFFICE", "commercial"),
    "PTHS": ("PENTHOUSE", "dwelling"),
    "REAR": ("REAR", "secondary"),
    "RESV": ("RESERVE", "ancillary"),
    "ROOM": ("ROOM", "dwelling"),
    "SE": ("SUITE", "commercial"),
    "SEC": ("SECTION", "dwelling"),
    "SHED": ("SHED", "ancillary"),
    "SHOP": ("SHOP", "commercial"),
    "SHRM": ("SHOWROOM", "commercial"),
    "SIGN": ("SIGN", "ancillary"),
    "SITE": ("SITE", "dwelling"),
    "STLL": ("STALL", "commercial"),
    "STOR": ("STORE", "ancillary"),
    "STR": ("STRATA UNIT", "dwelling"),
    "STU": ("STUDIO", "dwelling"),
    "SUBS": ("SUBSTATION", "ancillary"),
    "TNCY": ("TENANCY", "commercial"),
    "TNHS": ("TOWNHOUSE", "dwelling"),
    "TWR": ("TOWER", "structure"),
    "UNIT": ("UNIT", "dwelling"),
    "VLLA": ("VILLA", "dwelling"),
    "VLT": ("VAULT", "ancillary"),
    "WARD": ("WARD", "dwelling"),
    "WHSE": ("WAREHOUSE", "commercial"),
    "WKSH": ("WORKSHOP", "commercial"),
}

# Free-text spellings -> flat type code.  Codes and names map to themselves.
FLAT_TYPE_ALIASES: dict[str, str] = {
    **{code: code for code in FLAT_TYPES},
    **{name: code for code, (name, _) in FLAT_TYPES.items()},
    "U": "UNIT", "UNT": "UNIT", "APPT": "APT", "APTMT": "APT", "FLT": "FLAT",
    "SH": "SHOP", "SHP": "SHOP", "SUITE": "SE", "STE": "SE", "SUIT": "SE",
    "TH": "TNHS", "TWNHS": "TNHS", "TOWNHSE": "TNHS", "VILL": "VLLA",
    "CS": "CARS", "CAR SPACE": "CARS", "CARSPACE": "CARS", "CAR PARK": "CARP",
    "PARKING SPACE": "CARS", "OFF": "OFFC", "PH": "PTHS", "RM": "ROOM",
    "STORAGE": "STOR", "STORE": "STOR", "BERTH": "MBTH", "MARINA BERTH": "MBTH",
    "GRANNY FLAT": "REAR", "STUDIO": "STU", "TENANCY": "TNCY", "TEN": "TNCY",
    "WAREHSE": "WHSE", "FACTORY": "FCTY", "FAC": "FCTY", "KIOSK": "KSK",
    "TOWER": "TWR", "BUILDING": "BLDG", "BLD": "BLDG", "BLK": "BLCK",
}

# Words that name a building inside a complex.  Canonical form on the right.
STRUCTURE_WORDS: dict[str, str] = {
    "TOWER": "TOWER", "TWR": "TOWER",
    "BUILDING": "BUILDING", "BLDG": "BUILDING", "BLD": "BUILDING",
    "BLOCK": "BLOCK", "BLCK": "BLOCK", "BLK": "BLOCK",
    "WING": "WING",
}

LEVEL_TYPES: dict[str, str] = {
    "B": "BASEMENT",
    "FL": "FLOOR",
    "G": "GROUND FLOOR",
    "L": "LEVEL",
    "LG": "LOWER GROUND FLOOR",
    "LL": "LOWER LEVEL",
    "M": "MEZZANINE",
    "OD": "OBSERVATION DECK",
    "P": "PARKING",
    "PDM": "PODIUM",
    "PLF": "PLATFORM",
    "PTHS": "PENTHOUSE",
    "RT": "ROOFTOP",
    "SB": "SUB-BASEMENT",
    "UG": "UPPER GROUND FLOOR",
    "UL": "UPPER LEVEL",
}

# Level types that are commonly written without a number.
LEVEL_TYPES_NUMBERLESS = {"G", "LG", "UG", "M", "RT", "PDM", "OD", "PTHS"}

LEVEL_TYPE_ALIASES: dict[str, str] = {
    **{code: code for code in LEVEL_TYPES},
    **{name: code for code, name in LEVEL_TYPES.items()},
    "LEVEL": "L", "LVL": "L", "LEV": "L", "LV": "L",
    "FLOOR": "FL", "FLR": "FL",
    "GROUND": "G", "GF": "G", "GRD": "G", "GROUND LEVEL": "G",
    "BASEMENT": "B", "BSMT": "B",
    "LOWER GROUND": "LG", "UPPER GROUND": "UG",
    "MEZZANINE": "M", "MEZZ": "M", "MEZ": "M",
    "PODIUM": "PDM", "PARKING": "P", "ROOFTOP": "RT", "ROOF": "RT",
    "SUB BASEMENT": "SB", "LOWER LEVEL": "LL", "UPPER LEVEL": "UL",
}
# Single letters are too ambiguous to treat as level types in free text, and
# "PENTHOUSE 2" is far more often a flat than a level.
for _ambiguous in ("B", "G", "M", "P", "PTHS", "PENTHOUSE"):
    LEVEL_TYPE_ALIASES.pop(_ambiguous, None)

# G-NAF STREET_TYPE_AUT: CODE (full word) -> NAME (abbreviation)
STREET_TYPES: dict[str, str] = {
    "ACCESS": "ACCS", "ALLEY": "ALLY", "ALLEYWAY": "ALWY", "AMBLE": "AMBL",
    "APPROACH": "APP", "ARCADE": "ARC", "ARTERIAL": "ARTL", "AVENUE": "AV",
    "BANK": "BANK", "BAY": "BAY", "BEACH": "BCH", "BEND": "BEND",
    "BOULEVARD": "BVD", "BOULEVARDE": "BVDE", "BRACE": "BRCE", "BRAE": "BRAE",
    "BREAK": "BRK", "BRIDGE": "BDGE", "BROADWAY": "BDWY", "BROW": "BROW",
    "BYPASS": "BYPA", "BYWAY": "BYWY", "CAUSEWAY": "CAUS", "CENTRE": "CTR",
    "CENTREWAY": "CNWY", "CHASE": "CH", "CIRCLE": "CIR", "CIRCLET": "CLT",
    "CIRCUIT": "CCT", "CIRCUS": "CRCS", "CLOSE": "CL", "COMMON": "CMMN",
    "CONCOURSE": "CON", "COPSE": "CPS", "CORNER": "CNR", "CORSO": "CSO",
    "COURT": "CT", "COURTYARD": "CTYD", "COVE": "COVE", "CRESCENT": "CR",
    "CREST": "CRST", "CROSS": "CRSS", "CROSSING": "CRSG", "CUL-DE-SAC": "CDS",
    "DALE": "DALE", "DELL": "DELL", "DEVIATION": "DEVN", "DIP": "DIP",
    "DRIVE": "DR", "DRIVEWAY": "DRWY", "EDGE": "EDGE", "ELBOW": "ELB",
    "END": "END", "ENTRANCE": "ENT", "ESPLANADE": "ESP", "ESTATE": "EST",
    "EXPRESSWAY": "EXP", "EXTENSION": "EXTN", "FAIRWAY": "FAWY",
    "FIRETRAIL": "FTRL", "FOLLOW": "FOLW", "FORMATION": "FTN", "FREEWAY": "FWY",
    "FRONT": "FRNT", "FRONTAGE": "FRTG", "GAP": "GAP", "GARDEN": "GDN",
    "GARDENS": "GDNS", "GATE": "GTE", "GATES": "GTES", "GLADE": "GLD",
    "GLEN": "GLEN", "GRANGE": "GRA", "GREEN": "GRN", "GROVE": "GR",
    "GULLY": "GLY", "HEATH": "HTH", "HEIGHTS": "HTS", "HIGHROAD": "HRD",
    "HIGHWAY": "HWY", "HILL": "HILL", "INTERCHANGE": "INTG", "JUNCTION": "JNC",
    "KEY": "KEY", "LANDING": "LDG", "LANE": "LANE", "LANEWAY": "LNWY",
    "LINE": "LINE", "LINK": "LINK", "LOOKOUT": "LKT", "LOOP": "LOOP",
    "MALL": "MALL", "MEANDER": "MNDR", "MEWS": "MEWS", "MOTORWAY": "MWY",
    "MOUNT": "MT", "NOOK": "NOOK", "OUTLOOK": "OTLK", "PARADE": "PDE",
    "PARK": "PARK", "PARKWAY": "PKWY", "PASS": "PASS", "PATH": "PATH",
    "PATHWAY": "PHWY", "PIAZZA": "PIAZ", "PLACE": "PL", "PLAZA": "PLZA",
    "POCKET": "PKT", "POINT": "PNT", "PORT": "PORT", "PROMENADE": "PROM",
    "QUAD": "QUAD", "QUADRANT": "QDRT", "QUAY": "QY", "QUAYS": "QYS",
    "RAMBLE": "RMBL", "RAMP": "RAMP", "RANGE": "RNGE", "REACH": "RCH",
    "RESERVE": "RES", "REST": "REST", "RETREAT": "RTT", "RIDGE": "RDGE",
    "RIDGEWAY": "RGWY", "RING": "RING", "RISE": "RISE", "RIVER": "RVR",
    "ROAD": "RD", "ROADWAY": "RDWY", "ROTARY": "RTY", "ROUND": "RND",
    "ROUTE": "RTE", "ROW": "ROW", "RUE": "RUE", "RUN": "RUN",
    "SERVICEWAY": "SVWY", "SIDING": "SDNG", "SLOPE": "SLPE", "SPUR": "SPUR",
    "SQUARE": "SQ", "STEPS": "STPS", "STRAND": "STRA", "STREET": "ST",
    "STRIP": "STRP", "SUBWAY": "SBWY", "TARN": "TARN", "TERRACE": "TCE",
    "THOROUGHFARE": "THOR", "TOLLWAY": "TLWY", "TOP": "TOP", "TOR": "TOR",
    "TRACK": "TRK", "TRAIL": "TRL", "TURN": "TURN", "UNDERPASS": "UPAS",
    "VALE": "VALE", "VIADUCT": "VDCT", "VIEW": "VIEW", "VISTA": "VSTA",
    "WALK": "WALK", "WALKWAY": "WKWY", "WAY": "WAY", "WHARF": "WHRF",
    "WYND": "WYND",
}

STREET_TYPE_ALIASES: dict[str, str] = {
    **{code: code for code in STREET_TYPES},
    **{abbr: code for code, abbr in STREET_TYPES.items()},
    "AVE": "AVENUE", "AVN": "AVENUE", "BLVD": "BOULEVARD", "BVDE": "BOULEVARDE",
    "CRES": "CRESCENT", "CRS": "CRESCENT", "CRESC": "CRESCENT", "CIRC": "CIRCUIT",
    "DRV": "DRIVE", "DRIV": "DRIVE", "ESPL": "ESPLANADE", "GRV": "GROVE",
    "GDNS": "GARDENS", "HWAY": "HIGHWAY", "LN": "LANE", "PDE": "PARADE",
    "PRDE": "PARADE", "PLC": "PLACE", "PKWY": "PARKWAY", "RD": "ROAD",
    "SQR": "SQUARE", "STR": "STREET", "STRT": "STREET", "TER": "TERRACE",
    "TERR": "TERRACE", "WY": "WAY", "WLK": "WALK", "CT": "COURT", "CRT": "COURT",
}

# G-NAF STREET_SUFFIX_AUT: CODE -> NAME
STREET_SUFFIXES: dict[str, str] = {
    "CN": "CENTRAL", "DE": "DEVIATION", "E": "EAST", "EX": "EXTENSION",
    "IN": "INNER", "LR": "LOWER", "ML": "MALL", "N": "NORTH", "NE": "NORTH EAST",
    "NW": "NORTH WEST", "OP": "OVERPASS", "S": "SOUTH", "SE": "SOUTH EAST",
    "SW": "SOUTH WEST", "UP": "UPPER", "W": "WEST",
}
STREET_SUFFIX_ABBREV_ALIASES: dict[str, str] = {
    **{code: code for code in STREET_SUFFIXES},
    "NTH": "N", "STH": "S",
}
STREET_SUFFIX_WORD_ALIASES: dict[str, str] = {
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "CENTRAL": "CN",
    "EXTENSION": "EX", "LOWER": "LR", "UPPER": "UP", "INNER": "IN",
}

STATES: dict[str, str] = {
    "NSW": "NEW SOUTH WALES", "VIC": "VICTORIA", "QLD": "QUEENSLAND",
    "SA": "SOUTH AUSTRALIA", "WA": "WESTERN AUSTRALIA", "TAS": "TASMANIA",
    "NT": "NORTHERN TERRITORY", "ACT": "AUSTRALIAN CAPITAL TERRITORY",
    "OT": "OTHER TERRITORIES",
}
STATE_PIDS: dict[str, str] = {
    "NSW": "1", "VIC": "2", "QLD": "3", "SA": "4", "WA": "5", "TAS": "6",
    "NT": "7", "ACT": "8", "OT": "9",
}

# G-NAF GEOCODE_TYPE_AUT (subset): CODE -> (NAME, placement category)
GEOCODE_TYPES: dict[str, tuple[str, str]] = {
    "BAP": ("BUILDING ACCESS POINT", "building"),
    "BC": ("BUILDING CENTROID", "building"),
    "CDF": ("CENTRE-LINE DROPPED FRONTAGE", "parcel"),
    "DF": ("DRIVEWAY FRONTAGE", "parcel"),
    "EA": ("EMERGENCY ACCESS", "other"),
    "EAS": ("EMERGENCY ACCESS SECONDARY", "other"),
    "FC": ("FRONTAGE CENTRE", "parcel"),
    "FCS": ("FRONTAGE CENTRE SETBACK", "parcel"),
    "GG": ("GAP GEOCODE", "other"),
    "LB": ("LETTERBOX", "other"),
    "PAP": ("PROPERTY ACCESS POINT", "parcel"),
    "PAPS": ("PROPERTY ACCESS POINT SETBACK", "parcel"),
    "PC": ("PROPERTY CENTROID", "parcel"),
    "PCM": ("PROPERTY CENTROID MANUAL", "parcel"),
    "UC": ("UNIT CENTROID", "unit"),
    "UCM": ("UNIT CENTROID MANUAL", "unit"),
    "WCP": ("WATER CONNECTION", "other"),
    "WM": ("WATER METER", "other"),
}

ALIAS_TYPES: dict[str, str] = {
    "AL": "ALTERNATIVE LOCALITY",
    "CD": "CONTRIBUTOR DEFINED",
    "LD": "LEVEL DUPLICATION",
    "MR": "MAINTENANCE REFERENCE",
    "RA": "RANGED ADDRESS",
    "SYN": "SYNONYM",
}

PS_JOIN_TYPES: dict[str, str] = {"1": "AUTO", "2": "MANUAL"}


def flat_category(code: str) -> str:
    """Category of a flat type code ('' for no flat; unknown types count as dwellings)."""
    if not code:
        return ""
    return FLAT_TYPES.get(code, ("", "dwelling"))[1]


def normalise_flat_type(value: str) -> str:
    v = (value or "").strip().upper()
    return FLAT_TYPE_ALIASES.get(v, v)


def normalise_level_type(value: str) -> str:
    v = (value or "").strip().upper()
    if v in LEVEL_TYPES:
        return v
    return LEVEL_TYPE_ALIASES.get(v, v)


def normalise_street_type(value: str) -> str:
    v = (value or "").strip().upper()
    return STREET_TYPE_ALIASES.get(v, v)


def normalise_street_suffix(value: str) -> str:
    v = (value or "").strip().upper()
    return STREET_SUFFIX_ABBREV_ALIASES.get(v) or STREET_SUFFIX_WORD_ALIASES.get(v) or v


_GEOCODE_BY_NAME = {name: code for code, (name, _) in GEOCODE_TYPES.items()}


def normalise_geocode_type(value: str) -> str:
    """Geocode type code from a code or its description ("PROPERTY CENTROID" -> "PC")."""
    v = " ".join((value or "").strip().upper().split())
    return v if v in GEOCODE_TYPES else _GEOCODE_BY_NAME.get(v, v)


def geocode_category(code: str) -> str:
    """parcel / building / unit / other ('' when unknown)."""
    return GEOCODE_TYPES.get(normalise_geocode_type(code), ("", ""))[1]


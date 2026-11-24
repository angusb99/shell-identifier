# Shell Identifier

Scores every address on a cadastral parcel for the likelihood that it is a **shell address**: a parent or container address (a complex, building or floor) that other addresses nest under, rather than an occupiable address in its own right.

```
1-15 SMITH ST                         <- precinct shell      p = 1.00
├── TOWER A, 1-15 SMITH ST            <- building shell      p = 1.00
│   ├── LEVEL 3, TOWER A, 1-15 ...    <- level shell (if suites exist on it)
│   │   └── UNIT 301, LEVEL 3, ...    <- occupiable          p = 0.00
│   └── UNIT 1205, TOWER A, ...       <- occupiable
├── SHOP G01, 1-15 SMITH ST           <- occupiable
└── 3 SMITH ST                        <- inside the range: child of the precinct
```

There are two approaches:

- **Shell scoring** (`shell-identifier run`): an expert-weighted model that needs no labels.
- **Dwelling model** (`shell-identifier dwellings …`): learns from PD address labels (every other address is not PD) whose per-parcel count is right but whose choice of address may be wrong. It decides which addresses are the dwellings, moves misplaced labels, and names the primary shell. See [Dwelling model from labelled data](#dwelling-model-from-labelled-data).

It uses three data products: **G-NAF**, **Geoscape Cadastre** and **Geoscape Buildings**, plus the PD labels for the dwelling model. The inputs can be passed as polars LazyFrames (see [From polars LazyFrames](#from-polars-lazyframes)), or located by path and table name in a small YAML file (see [Your datasets](#your-datasets)). All processing up to the model is done in **polars**, with vectorised shapely for the spatial joins. It can also read free-text address strings with a rule-based parser. There is no LLM and no libpostal. The scoring is a transparent log-odds model: every probability breaks down into named reasons, such as `12 nested sub-address(es) (12 dwelling) [+5.6]; G-NAF primary with 12 linked secondaries [+2.0]`.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,plot]"      # Windows; use .venv/bin/python elsewhere
```

```bash
# 1. synthetic G-NAF / Cadastre / Buildings with ground truth (also writes data/synthetic/assets/)
shell-identifier synth --out data/synthetic --parcels 1500 --seed 42
```

```bash
# 1b. the dwelling model on the four synthetic assets, located by a datasets YAML
shell-identifier dwellings verify --datasets data/synthetic/assets/datasets.yaml --out results/dwellings
```

```bash
# 2. score (G-NAF input)
shell-identifier run --gnaf data/synthetic/G-NAF --cadastre data/synthetic/Cadastre/NSW_CADASTRE.gpkg --buildings data/synthetic/Buildings/NSW_BUILDINGS.gpkg --out results/synthetic_gnaf
```

```bash
# 2b. or score free-text addresses (CSV with address, longitude, latitude)
shell-identifier run --text data/synthetic/text/addresses_free_text.csv --id-col address_pid --cadastre data/synthetic/Cadastre/NSW_CADASTRE.gpkg --buildings data/synthetic/Buildings/NSW_BUILDINGS.gpkg --out results/synthetic_text
```

```bash
# 3. evaluate against labels, then render review maps
shell-identifier evaluate --results results/synthetic_gnaf --truth data/synthetic/truth/shell_truth.csv --out results/synthetic_gnaf/evaluation.md
```

```bash
shell-identifier plot --results results/synthetic_gnaf --cadastre data/synthetic/Cadastre/NSW_CADASTRE.gpkg --buildings data/synthetic/Buildings/NSW_BUILDINGS.gpkg --pattern NEEDS_REVIEW --n 5
```

```bash
# parse a string
shell-identifier parse "UNIT 1205, LEVEL 12, TOWER B, THE QUAY, 1-15 SMITH STREET SOUTHBANK VIC 3006"
```

Run the tests with `.venv/Scripts/python -m pytest`.

## From polars LazyFrames

The most direct way in: read the four inputs yourself and pass the frames. Only the columns that are needed are collected. For a step-by-step walkthrough, from preparing the inputs to reading the results and finding insights, see the [run guide](docs/run_guide.md); for the mathematics behind the model, see the [technical report](docs/methodology.md).

```python
import polars as pl
from shell_identifier import verify_frames, check_frames

frames = dict(
    labelled_asset=pl.scan_parquet("D:/data/pd_labels.parquet"),   # ADDRESS_ID, PD_FLAG (the PD addresses)
    ds_asset=pl.scan_parquet("D:/data/addresses.parquet"),         # ADDRESS_ID, G-NAF fields, LATITUDE, LONGITUDE
    cad=pl.scan_csv("D:/data/cadastre_base.csv"),                  # cadastre_polygon_pid, parcel_id, WKT
    bld=pl.scan_csv("D:/data/buildings.csv"),                      # building_pid, WKT
)
print(check_frames(**frames))        # which column feeds which field, and what's missing
result = verify_frames(**frames, out_dir="results/dwellings")
result["assignments"]                # per address: dwelling / primary shell / sub-shell / out of scope
result["moves"]                      # PD labels moved to another address (the review list)
```

Defaults for this layout:
- **Keys:** `ADDRESS_ID` is the address key in both the labels and `ds_asset`.
- **Labels:** `labelled_asset` lists the PD addresses. Rows whose `PD_FLAG` is `TRUE/1/Y/YES/PD` (a boolean or 0/1 column works too) are dwellings. Rows with any other value are ignored, and so is the flag column itself if there isn't one (every row is then a PD). Every `ds_asset` address that isn't flagged is taken as not PD, so a parcel's dwelling count is its number of PD flags, which can be zero. `ds_asset` should therefore cover the same area as the labels: an unlabelled suburb would be read as having no dwellings.
- **Address fields:** G-NAF names are found case-insensitively (`FLAT_TYPE_CODE` or `FLAT_TYPE`, `NUMBER_FIRST`, `STREET_NAME`, `STATE_ABBREVIATION`, …), as are `LATITUDE`/`LONGITUDE` (or `LAT`/`LON`/`LONG`).
- **Parcels:** `parcel_id` is the parcel key, and polygons that share it are parts of one parcel. `cadastre_polygon_pid` is used where `parcel_id` is empty, and every polygon is treated as a base parcel.
- **Geometry:** WKT columns are found by name (`WKT`, `GEOM_WKT`, `SHAPE`, …) or by their content. EWKT (`SRID=…;`) and WKB also work.
- **CRS:** `EPSG:7844` (GDA2020 lon/lat) is the default. Pass `crs="EPSG:4283"` for GDA94, or `crs={"cad": "EPSG:7856"}` for projected WKT. Lon/lat swapped, or projected coordinates labelled as lon/lat, stop the run with an explanation.

Override any name with `columns={"ds_asset": {"latitude": "Y"}, "cad": {"geometry": "GEOM_TEXT"}}`, and the values that mean PD with `pd_values=`. Optional inputs are `counts=` (a frame of `parcel_id, n_dwellings` with trusted counts), and the `bld_address=` / `bld_cadastre=` link tables. Any extra columns the pipeline knows are used when present, such as `LEGAL_PARCEL_ID`, `PRIMARY_SECONDARY`, `GEOCODE_TYPE_CODE`, building `total_floor_area` / `roof_height` / `building_use`, and cadastre strata counts. Features with no data, or too little to use, are left out of the model and listed in the log. "Too little" means fewer than about 30 expected values in the 200,000-row sample that scikit-learn bins from, on runs with more rows than that.

`predict_frames(result["bundle"], ds_asset=..., cad=..., bld=...)` applies the fitted model to unlabelled areas, and `score_frames(...)` runs the label-free shell scorer. The same files can be named on the command line instead, for example `--labelled-asset pd_labels.parquet --ds-asset addresses.parquet --cad cadastre_base.csv --bld buildings.csv`.

## Your datasets

The pipeline reads four datasets. Their names are placeholders for your real table names:

| dataset | what it holds | key columns (defaults) |
|---|---|---|
| `labelled_asset` | the PD addresses (every other address is not PD) | `address_detail_pid`, optional `label` flag |
| `ds_asset` | every address, G-NAF fields, one row each | `address_detail_pid`, `longitude`, `latitude`, the G-NAF number / flat / level / street / locality columns, `alias_principal`, `primary_secondary`, `legal_parcel_id`, `geocode_type_code`, `date_retired` |
| `bld` | Geoscape Buildings footprints | `building_pid`, geometry, `building_area`, `total_floor_area`, `roof_height`, `building_use` |
| `cad` | Geoscape Cadastre parcels (base + strata) | `cadastre_pid`, geometry, `parcel_type`, `base_parcel`, `strata_type`, strata counts |

Optional extras: `bld_address` and `bld_cadastre` (the Geoscape link tables) and `parcel_counts` (trusted dwelling counts).

**1. Write the datasets file** and edit the paths, table names and any column names that differ:

```bash
shell-identifier datasets template --out datasets.yaml
```

A minimal file needs only the paths and the real table names:

```yaml
root: D:/data/address_project        # relative paths and unset paths resolve here
labelled_asset: {table: census_pd_labels}
ds_asset:       {table: gnaf_address_view}
bld:            {table: geoscape_buildings}
cad:            {table: geoscape_cadastre}
```

Each `path` can be a file (`.parquet`, `.csv`, `.psv`, `.arrow`, `.gpkg`, `.shp`, `.fgb`, …), a folder (the table is `<folder>/<table>.<ext>` or a folder of parquet parts `<folder>/<table>/`), a GeoPackage, FileGDB or SQLite database (the table is the layer), or a glob or object-store URI that polars can scan. Column mappings are `canonical: your_column`, matched case-insensitively. If a name isn't found, the usual G-NAF and Geoscape names and the 10-character shapefile aliases are tried. `filter:` takes an SQL predicate on the source columns, for example `"state_abbreviation = 'NSW'"`. Geometry can be WKB (binary or hex), WKT or GeoParquet. Flat, level and geocode types can be codes (`APT`, `L`, `PC`) or names (`APARTMENT`, `LEVEL`, `PROPERTY CENTROID`). For `labelled_asset`, set `pd_values` to the codes that mean PD. Rows with any other value are taken as not PD, and the run log lists which values it saw.

**2. Check it against the data.** This shows where each dataset resolved to, its row count and CRS, and which column each field maps to:

```bash
shell-identifier datasets check --datasets datasets.yaml
```

**3. Run.** Paths and table names on the command line override the file:

```bash
shell-identifier dwellings verify --datasets datasets.yaml --out results/dwellings
```

```bash
shell-identifier dwellings verify --labelled-asset D:/data --ds-asset D:/data --bld D:/data --cad D:/data --table ds_asset=gnaf_address_view --table cad=geoscape_cadastre --out results/dwellings
```

The same flags work for `run`, `dwellings predict` and `datasets check`. `ds_asset` can also be a G-NAF PSV release folder (detected automatically, or use `--gnaf`) or single-line address text (`format: text` or `--text`; parsed by the rule-based parser). In Python:

```python
from shell_identifier import datasets_config, load_inputs, prepare_context
from shell_identifier.dwellings import verify, write_outputs

ds = datasets_config("datasets.yaml", {"ds_asset": {"path": "D:/data/addresses.parquet"}})
inputs = load_inputs(ds, labels=True)
ctx = prepare_context(*inputs.pipeline_args())
write_outputs(verify(ctx, inputs.labels, counts=inputs.counts), "results/dwellings")
```

## How it works

```
G-NAF PSV ─┐                         ┌─ address → base parcel (point in polygon, snap ≤ 5 m)
or text ───┼─ components (street,    ├─ address → building (geocode in footprint, else building_address link)
           │  number, building,      ├─ building → parcel (building_cadastre link, else centroid)
Cadastre ──┤  level, flat)  ─────────┤
Buildings ─┘                         └─ per-parcel context features ─→ log-odds score ─→ class + reasons
```

### 1. Address components (`components.py`, `parser.py`)

Each address is reduced to a street key, a number key (`25A`, `1-15`) and up to three sub-address components, from least to most specific:

| component | from | example |
|---|---|---|
| `comp_structure` | structural building names (TOWER A, BUILDING 2, BLOCK C, NORTH WING) or flat types BLDG/TWR/BLCK | `TOWER A` |
| `comp_level` | LEVEL_TYPE_CODE + number | `L:3` |
| `comp_flat` | FLAT_TYPE_CODE + number | `UNIT:1205` |

**Nesting rule.** Address B nests under A when they share street and number, every component A has is equal in B, and B has at least one component A lacks. A flat is never a container.

Building names that only label a site ("THE ARCADIA") are ignored when comparing. If one street number carries several different names on its units ("THE ASH", "THE BIRCH"), those names are promoted to building identifiers.

The **free-text parser** handles commas or none, slash units (`5/25`, `G01/25-27`, `3.04/25`), attached forms (`U5`, `L2`, `APT12`), ordinal floors (`1ST FLOOR`), towers (`NORTH TOWER`, `A BLOCK`) and ranges. It also handles localities that look like street types (`ST KILDA`, `SAMPLE HILL`), and can use the G-NAF locality gazetteer.

### 2. Context features (`features.py`)

Each address is described against the other **principal** addresses on its parcel. Aliases are scored but never count as context. Everything is polars group-by / join work, so there are no per-parcel Python loops.

| feature | default weight | meaning |
|---|---:|---|
| `log_children` | +2.2 | log(1 + weighted nested sub-addresses). Ancillary (car space, store) counts 0.25 and secondary dwellings (REAR, COTTAGE) count 0.5 |
| `single_child` | −1.0 | only one nested address: often a granny flat, with the main address still occupied |
| `only_secondary_children` / `only_ancillary_children` | −2.0 / −2.0 | children are only REAR/COTTAGE, or only car spaces |
| `log_range_children` | +2.0 | a number range (`1-15`) covering other numbers on the parcel |
| `log_offparcel_children` | +1.2 | sub-addresses of the same street address geocoded onto nearby parcels (≤150 m) |
| `stacked_children` | +1.0 | share of children at exactly the same geocode (typical G-NAF unit stacking) |
| `suffix_base` / `suffix_multi_variant` / `suffix_single_variant` | +0.4 / +0.8 / −1.2 | `100` beside `100A`+`100B` (usually split) vs `100` beside only `100A` (usually both occupied) |
| `strata_matches_siblings` / `strata_exceeds_siblings` | +1.5 / −2.0 | cadastre strata count = number of sibling dwellings (no lot left for this address) vs one more |
| `building_deficit` / `building_surplus` | +0.8 / −0.8 | fewer dwelling-sized buildings than sibling dwellings, or enough for this address too |
| `own_building` | −2.0 | the geocode sits in a building that no other address uses |
| `outside_buildings` | +0.7 | geocode outside every footprint while its siblings are inside one |
| `multistorey` / `vertical_strata` | +1.0 / +0.8 | building of 3 or more storeys (from floor area ÷ footprint) / vertical strata scheme |
| `parcel_level_geocode` | +0.4 | property-centroid/frontage geocode while siblings have building/unit geocodes |
| `gnaf_primary_linked` / `gnaf_primary_flag_only` / `gnaf_secondary` | +2.0 / +0.5 / −1.5 | G-NAF PRIMARY_SECONDARY flag and table |
| `legal_common_property` / `legal_strata_lot` | +1.5 / −1.0 | LEGAL_PARCEL_ID like `CP/SP12345` vs `5/SP12345` (regexes in config) |
| `vanity_name` | +0.3 | a container that carries a complex name |
| `leaf_subaddress` / `leaf_street_address` | −4.0 / −3.0 | nothing nested, ranged or suffixed around it |
| `single_address_parcel` / `has_parent` / `alias` | −2.0 / −0.5 / −1.5 | |

The building and strata balance checks only apply to *small groups* (1–6 sibling dwellings), where one dwelling more or less changes the answer. Large complexes are decided by their children.

### 3. Scoring (`scoring.py`)

`p = sigmoid(intercept + Σ weight·feature)`, and then:

| p | class |
|---|---|
| ≥ 0.80 | `SHELL` |
| 0.60–0.80 | `LIKELY_SHELL` |
| 0.40–0.60 | `UNCERTAIN`: send to review |
| 0.20–0.40 | `LIKELY_NOT_SHELL` |
| < 0.20 | `NOT_SHELL` |

All weights, bands and thresholds live in `src/shell_identifier/default_config.yaml`; override any of them with `--config my.yaml`.

### 4. Calibrating on your data (`train.py`)

Label a sample of real addresses (`address_detail_pid,is_shell`), score them, then:

```bash
shell-identifier train --results results/my_run --labels my_labels.csv --l2 1.0 --out weights.json
```

```bash
shell-identifier run ... --weights weights.json
```

Training minimises `NLL + l2·‖w − w_default‖²`. Weights only move where your labels give evidence, so features that are rare in a few hundred labels keep their expert defaults. Over-sample the contested cases (suffix groups, single-child parcels, precincts) when labelling, because leaf addresses teach the model nothing.

## Outputs

| file | content |
|---|---|
| `address_scores.csv` / `.gpkg` (`--no-gpkg` to skip) | `address_pid, address_label, parcel_pid, building_pid, shell_probability, shell_class, parent_pid, parent_label, node_type, n_desc, n_range_desc, n_offparcel_desc, n_suffix_variants, reasons, …` |
| `address_features.csv` | every `f_*` model feature plus the logit, for training and auditing |
| `parcel_summary.csv` | per parcel: counts per class, top candidate, shell PIDs, building/strata context, and a `pattern` (`SINGLE_ADDRESS`, `MULTI_UNIT_WITH_SHELL`, `MULTI_UNIT_NO_SHELL`, `SUFFIX_GROUP_WITH_SHELL`, `SUFFIX_GROUP_ALL_OCCUPIABLE`, `RANGE_PRECINCT`, `MULTI_FRONTAGE`, `NEEDS_REVIEW`) |

`parent_pid` gives each sub-address its most specific container (unit → tower → precinct), which is useful for building the address hierarchy downstream.

## Input schemas

These are the product layouts the loaders know by default. Any other naming is mapped in the datasets YAML (see [Your datasets](#your-datasets)). **Check them against your copies**: names follow the public product guides, and loaders also accept the 10-character Shapefile aliases.

**G-NAF** as a flattened address table (`ds_asset`; one row per address, ADDRESS_VIEW-style names), or as the PSV release (found by recursive glob): `{STATE}_ADDRESS_DETAIL`, `_ADDRESS_DEFAULT_GEOCODE`, `_STREET_LOCALITY`, `_LOCALITY`, `_STATE`, `_PRIMARY_SECONDARY` (optional). Retired rows are dropped. GDA2020 (`--crs EPSG:7844`) by default; pass `--crs EPSG:4283` for GDA94.

**Geoscape Cadastre** (layer `cadastre`): `cadastre_pid`, `parcel_type` (Lot/Strata/Stratum/Road), `base_parcel` (Yes/NULL), `strata_type`, `total_strata_count`, `horizontal_strata_count`, `vertical_strata_count`, … Only base parcels (`base_parcel = "Yes"`, or else `parcel_type = "Lot"`) define the parcel context. Where strata counts are empty, they are derived from the strata polygons.

**Geoscape Buildings**: `buildings` (`building_pid`, `building_area`, `total_floor_area`, …), plus the optional aspatial `building_address` (`building_pid`, `address_detail_pid`, `relationship_confidence`, `primary_building`) and `building_cadastre` tables. When `bld` is a GeoPackage holding those layers they are read automatically; otherwise give them as `bld_address` / `bld_cadastre` (or `--bld-address` / `--bld-cadastre`). Without them, addresses and buildings are matched to parcels spatially, which works as well on the synthetic data. `height_table` / `--building-height` joins the Building Height insight pack (`roof_height`) when floor area is missing, and a missing `building_area` is measured from the footprint.

## Synthetic data

`shell-identifier synth` lays out a fictional suburb (TESTWOOD / SAMPLE HILL / MOCKTON PARK NSW) and fills each lot with one of 18 scenarios, each with ground truth in `truth/shell_truth.csv`:

house · vacant land · house + REAR/COTTAGE · house + UNIT 1 granny flat · 10 + 10A (both occupied) · duplex shell (`10` + `10A/10B`, or `10` + `1/10, 2/10`) · triplex `10/10A/10B` (all occupied) · shop-top `70` + `70A` · townhouses · walk-up flats (with optional levels, vanity names, car spaces) · units without a parent · mixed use (shops + units) · office tower (level shells vs whole-floor tenancies) · high-rise precinct (range + TOWER/BUILDING/BLOCK or vanity-named towers, or towers at separate numbers) · range townhouses (`2-8` + `2,4,6,8`) · retirement village · corner lot with an alias or second frontage · a precinct split across two lots

The data also includes realistic noise:
- primary/secondary links present on only about 60–70% of complexes, and some with flags but no link rows
- strata counts populated on about 60% of parcels
- stacked vs spread unit geocodes
- 2% of unit geocodes slipped onto the neighbouring lot
- 3% of buildings not captured
- 30% of attached dwellings captured as one merged footprint
- 20% of suffix groups all geocoded at the property centroid
- retired records

It also writes the same addresses as messy free text: abbreviations, slash units, title case and missing commas.

`assets/` holds the four pipeline inputs as standalone tables in the shape of a typical export:
- `ds_asset.parquet` is a flattened address view, with upper-case names, flat / level / geocode type *names*, `PRIMARY_PID` and retired rows.
- `bld.parquet` and `cad.parquet` are GeoParquet.
- `labelled_asset.parquet` holds the noisy PD labels.
- `datasets.yaml` points at the four tables.

### Results on synthetic data

| | addresses | ROC AUC | contested F1 @0.5 | errors @0.5 | UNCERTAIN |
|---|---:|---:|---:|---:|---:|
| seed 42, 1,500 parcels (G-NAF input) | 8,179 | 1.000 | 0.994 | 8 FP / 0 FN | 6 |
| same, free-text input | 8,179 | 1.000 | 0.996 | 5 FP / 0 FN | 5 |
| seed 11, 15,000 parcels (5 s; 23 s before the polars port) | 89,823 | 1.000 | 0.997 | 34 FP / 6 FN | — |

The parser recovered 100% of street, number, building and level components from the free text. Every flat number was recovered too; the only differences are flat types missing in slash forms like `5/25`. All remaining errors are houses with a secondary dwelling where one of the two buildings was not captured.

**Read these numbers with care.** The generator and the features encode the same understanding of how shells look, so synthetic accuracy is an upper bound. It shows the mechanics work; it does not prove real-world accuracy. Real G-NAF will contain patterns not simulated here, so label a real sample and run `evaluate` / `train` before relying on the scores.

Ablation (seed 42, removing one input at a time):

| inputs | contested F1 | FP / FN | hard-case accuracy* | UNCERTAIN |
|---|---:|---:|---:|---:|
| all | 0.994 | 8 / 0 | 0.980 | 6 |
| no buildings | 1.000 | 0 / 0 | 1.000 | 63 |
| no strata counts | 0.986 | 18 / 0 | 0.955 | 16 |
| no G-NAF primary/secondary | 0.994 | 8 / 0 | 0.980 | 3 |
| address components + parcels only | 0.969 | 41 / 0 | 0.897 | 116 |

\*bare addresses in duplex, triplex, 10/10A, shop-top and granny-flat parcels.

Buildings turn about 60 UNCERTAIN suffix and duplex cases into confident answers. The cost is a few confident errors where a building is missing from the capture. If your buildings coverage is patchy, lower `own_building` / `building_deficit` in the config.

## Dwelling model from labelled data

Use this when you have a list of **PD** (private dwelling) addresses, with every other address out of scope, where each parcel's number of PDs is right but a label is sometimes on the wrong address. Typical mistakes are `20 X ST` instead of `UNIT 5, 20 X ST`, or `10` instead of `10B`.

```bash
shell-identifier dwellings verify --datasets datasets.yaml --out results/dwellings
```

```bash
# apply the trained model to unlabelled areas (optionally with known counts: parcel_counts / --counts)
shell-identifier dwellings predict --model results/dwellings/dwelling_model.joblib --datasets datasets.yaml --out results/predicted
```

`labelled_asset` needs an address id, and optionally a PD flag. The defaults are `address_detail_pid` and `label`; change them in the datasets YAML, or with `--labels-id` / `--label-col`. Rows flagged `PD/1/Y/YES/TRUE` are dwellings; set other codes with `pd_values` (or `--pd-values`). Boolean and 0/1 columns work as they are. Rows with any other value are ignored. Without a flag column, every row is a PD. Every address not flagged PD is out of scope, so every parcel is checked: its dwelling count is its number of PD labels, which may be zero, unless you give trusted counts as `parcel_counts` (`--counts`).

### How it works

**Features** (`dwellings/features.py`). Each address gets about 110 features, all relative to its parcel:

| family | examples |
|---|---|
| address text | flat type category (dwelling / commercial / ancillary / secondary), level, building and vanity names, `101`-on-level-1 numbering, `G01` prefixes, number suffix, ranges |
| geocode placement | geocode type, distance to the parcel centroid and boundary, how many addresses share the point, inside / near a building |
| building composition | footprint and storeys of its building, addresses per building, buildings / floor area / commercial share / strata counts on the parcel |
| nesting context | everything from the shell scorer (children, ranges, suffix siblings, G-NAF primary/secondary, legal parcel id) plus its shell logit |

**Model** (`dwellings/countem.py`). The true dwelling status of each address is unknown, with the constraint that exactly K addresses on a parcel are dwellings.
- A gradient-boosted classifier `f(x)` learns what dwelling addresses look like across all parcels.
- The labels enter through a *misplacement* model: a parcel has m PD labels on the wrong address, with a learned prior P(m), and every way of misplacing m labels is equally likely. This encodes "right count, wrong address" directly.
- EM alternates exact posterior marginals per parcel (a truncated Poisson-binomial forward/backward recursion, O(n·m)) with refitting the classifier on those soft targets.
- The classifier is cross-fitted by parcel, so a parcel is always judged by a model that never saw its own labels.
- The final answer is the exact most-probable configuration: K dwellings per parcel.

**Roles** (`dwellings/roles.py`).
- `DWELLING`.
- `PRIMARY_SHELL`: the top-most non-dwelling that the dwellings nest under. Nesting covers unit → tower → precinct, `100A`/`100B` → `100`, ranges, and sub-addresses geocoded onto a neighbouring parcel.
- `SUB_SHELL`: a tower or level shell inside a primary shell.
- `OUT_OF_SCOPE`: everything else.

A non-dwelling above a single dwelling only counts as a shell if it also looks like one on the shell score, so the shop in a `70` / `70A` shop-top stays out of scope.

### Outputs

| file | content |
|---|---|
| `dwelling_assignments.csv` / `.parquet` / `.gpkg` | per address: input label, cross-fitted `dwelling_prob`, `posterior`, rank in parcel, `role`, `label_change` (`PD->OOS`, `OOS->PD`), primary shell, `evidence` (what the model saw) and `decision` |
| `label_moves.csv` | each moved label: `from` address → `to` address with posteriors and a confidence; this is the review list |
| `parcel_dwellings.csv` | per parcel: labelled vs assigned dwellings, expected dwellings, primary shell(s), `status` (`AGREES_WITH_LABELS` / `LABELS_MOVED` / `PREDICTED`), and `margin` (gap between the weakest dwelling and the strongest non-dwelling); `review` is set when the margin is < 0.3 |
| `dwelling_model.joblib`, `model_report.md` | the fitted model for `predict`, EM history, estimated misplacement rate, feature importances |

### Results on synthetic data

The generator writes `truth/pd_labels.csv`: the PD count is right on every site, but on 30% of eligible sites one PD (sometimes two) is placed on a non-dwelling, usually the shell.

| | result |
|---|---|
| `verify`, seed 42 (8,179 addresses, 360 misplaced labels) | address accuracy 0.956 → **1.000**; **360 / 360** misplaced labels moved to the right address, **0** correct labels broken; shells precision / recall **1.000 / 1.000**; misplacement rate estimated 0.30 (true 0.30); 5–6 EM iterations, ~80 s |
| same, from the four `assets/` tables only (no Geoscape link tables) | identical: **360 / 360** fixed, 0 broken, shells 1.000 / 1.000 |
| `predict` on a different suburb (seed 11), no labels, no counts | address accuracy **0.997**; 99.1% of parcels exactly right |
| same, with known parcel counts (`--counts`) | address accuracy **1.000** |

**These numbers are a best case.** The synthetic misplacements follow exactly the uniform-swap story the model assumes, and dwelling flat types (UNIT/APT/VILLA) are always dwellings there, which is why `t_node_dwelling` dominates the feature importances and posteriors sit at 0 or 1. On real data, expect softer posteriors, some genuinely ambiguous parcels (`review = True`), and more weight on building and geocode features. Read `label_moves.csv` from the highest confidence down, and spot-check a sample before accepting the moves wholesale.

Runtime is dominated by the classifier fits (folds × iterations). For large extracts use `--folds 3 --iterations 4`; single-address parcels are already subsampled for training.

## Limitations and assumptions

- **Parcel context is the base parcel.** Where a complex spans several unconsolidated lots, children on other lots are only picked up when they share the exact street address (`log_offparcel_children`). Range or suffix relationships across parcels are not.
- **The suffix case is inherently ambiguous** without buildings, strata counts or legal parcel ids. Expect `UNCERTAIN` there; that is the honest answer.
- **The legal parcel regexes are NSW-flavoured** (`CP/SP…`, `n/SP…`). Adjust `legal_*_patterns` for other jurisdictions.
- **Strata counts may include car-space and storage lots**, which is why those checks only apply to small groups.
- **The parser is rule-based.** It covers common Australian formats, but unusual strings (corner descriptions, `RMB`, `CNR X & Y`) will produce warnings in `parse_warnings`. Pass a locality list (`--localities`) for best street/locality splitting.
- **Scale.** Loading and features are polars: the shell scorer handles the 8k-address synthetic suburb in under 0.5 s, and the polars port reproduces the earlier pandas results exactly. The dwelling model's time is in the classifier fits (folds × iterations). For a state-sized extract, run state by state, use `--bbox` windows with some overlap, or a `filter:` in the datasets YAML.

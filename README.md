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
- **Dwelling model** (`shell-identifier dwellings …`): learns from PD / out-of-scope labels whose per-parcel count is right but whose choice of address may be wrong. It decides which addresses are the dwellings, moves misplaced labels, and names the primary shell. See [Dwelling model from labelled data](#dwelling-model-from-labelled-data).

It uses three data products: **G-NAF**, **Geoscape Cadastre** and **Geoscape Buildings**. It can also read free-text address strings with a rule-based parser. There is no LLM and no libpostal. The scoring is a transparent log-odds model: every probability breaks down into named reasons, such as `12 nested sub-address(es) (12 dwelling) [+5.6]; G-NAF primary with 12 linked secondaries [+2.0]`.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,plot]"      # Windows; use .venv/bin/python elsewhere
```

```bash
# 1. synthetic G-NAF / Cadastre / Buildings with ground truth
shell-identifier synth --out data/synthetic --parcels 1500 --seed 42
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

Each address is described against the other **principal** addresses on its parcel. Aliases are scored but never count as context. Everything is groupby/merge based, so there are no per-parcel Python loops.

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
| `address_scores.csv` / `.gpkg` | `address_pid, address_label, parcel_pid, building_pid, shell_probability, shell_class, parent_pid, parent_label, node_type, n_desc, n_range_desc, n_offparcel_desc, n_suffix_variants, reasons, …` |
| `address_features.csv` | every `f_*` model feature plus the logit, for training and auditing |
| `parcel_summary.csv` | per parcel: counts per class, top candidate, shell PIDs, building/strata context, and a `pattern` (`SINGLE_ADDRESS`, `MULTI_UNIT_WITH_SHELL`, `MULTI_UNIT_NO_SHELL`, `SUFFIX_GROUP_WITH_SHELL`, `SUFFIX_GROUP_ALL_OCCUPIABLE`, `RANGE_PRECINCT`, `MULTI_FRONTAGE`, `NEEDS_REVIEW`) |

`parent_pid` gives each sub-address its most specific container (unit → tower → precinct), which is useful for building the address hierarchy downstream.

## Input schemas

The synthetic data follows these schemas and the loaders read them. **Check them against your copies**: names follow the public product guides, and loaders also accept the 10-character Shapefile aliases.

**G-NAF** (PSV release, found by recursive glob under `--gnaf`): `{STATE}_ADDRESS_DETAIL`, `_ADDRESS_DEFAULT_GEOCODE`, `_STREET_LOCALITY`, `_LOCALITY`, `_STATE`, `_PRIMARY_SECONDARY` (optional). Retired rows are dropped. GDA2020 (`--crs EPSG:7844`) by default; pass `--crs EPSG:4283` for GDA94.

**Geoscape Cadastre** (layer `cadastre`): `cadastre_pid`, `parcel_type` (Lot/Strata/Stratum/Road), `base_parcel` (Yes/NULL), `strata_type`, `total_strata_count`, `horizontal_strata_count`, `vertical_strata_count`, … Only base parcels (`base_parcel = "Yes"`, or else `parcel_type = "Lot"`) define the parcel context. Where strata counts are empty, they are derived from the strata polygons.

**Geoscape Buildings**: `buildings` (`building_pid`, `building_area`, `total_floor_area`, …), plus the aspatial `building_address` (`building_pid`, `address_detail_pid`, `relationship_confidence`, `primary_building`) and `building_cadastre` tables. These are read automatically from the same GeoPackage, or given with `--building-address` / `--building-cadastre`. `--building-height` joins the Building Height insight pack (`roof_height`) when floor area is missing.

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

### Results on synthetic data

| | addresses | ROC AUC | contested F1 @0.5 | errors @0.5 | UNCERTAIN |
|---|---:|---:|---:|---:|---:|
| seed 42, 1,500 parcels (G-NAF input) | 8,179 | 1.000 | 0.994 | 8 FP / 0 FN | 6 |
| same, free-text input | 8,179 | 1.000 | 0.996 | 5 FP / 0 FN | 5 |
| seed 11, 15,000 parcels (23 s) | 89,823 | 1.000 | 0.997 | 34 FP / 6 FN | — |

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

Use this when you have address labels of **PD** (private dwelling) or **out of scope**, where each parcel's number of PDs is right but a label is sometimes on the wrong address. Typical mistakes are `20 X ST` instead of `UNIT 5, 20 X ST`, or `10` instead of `10B`.

```bash
shell-identifier dwellings verify --gnaf <G-NAF dir> --cadastre <cadastre.gpkg> --buildings <buildings.gpkg> --labels labels.csv --out results/dwellings
```

```bash
# apply the trained model to unlabelled areas (optionally with known counts: --counts parcel_pid,n_dwellings)
shell-identifier dwellings predict --model results/dwellings/dwelling_model.joblib --gnaf <G-NAF dir> --cadastre <cadastre.gpkg> --buildings <buildings.gpkg> --out results/predicted
```

`labels.csv` needs an address id and a label. The defaults are `address_detail_pid` and `label`; change them with `--labels-id` / `--label-col`. Values `PD/1/Y/YES/TRUE` mean dwelling, and `OOS/0/N/NO/FALSE` mean out of scope; override them with `--pd-values` / `--oos-values`. Addresses missing from the file are treated as unlabelled candidates. A parcel's dwelling count is its number of PD labels, unless you pass trusted counts with `--counts`.

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
| `dwelling_assignments.csv` / `.gpkg` | per address: input label, cross-fitted `dwelling_prob`, `posterior`, rank in parcel, `role`, `label_change` (`PD->OOS`, `OOS->PD`), primary shell, `evidence` (what the model saw) and `decision` |
| `label_moves.csv` | each moved label: `from` address → `to` address with posteriors and a confidence; this is the review list |
| `parcel_dwellings.csv` | per parcel: labelled vs assigned dwellings, expected dwellings, primary shell(s), `status` (`AGREES_WITH_LABELS` / `LABELS_MOVED` / `PREDICTED`), and `margin` (gap between the weakest dwelling and the strongest non-dwelling); `review` is set when the margin is < 0.3 |
| `dwelling_model.joblib`, `model_report.md` | the fitted model for `predict`, EM history, estimated misplacement rate, feature importances |

### Results on synthetic data

The generator writes `truth/pd_labels.csv`: the PD count is right on every site, but on 30% of eligible sites one PD (sometimes two) is placed on a non-dwelling, usually the shell.

| | result |
|---|---|
| `verify`, seed 42 (8,179 addresses, 360 misplaced labels) | address accuracy 0.956 → **1.000**; **360 / 360** misplaced labels moved to the right address, **0** correct labels broken; shells precision / recall **1.000 / 1.000**; misplacement rate estimated 0.30 (true 0.30); 6 EM iterations, ~90 s |
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
- **Scale.** Throughput is about 4k addresses/s on one core. Run state by state, or use `--bbox` windows with some overlap, for very large extracts.

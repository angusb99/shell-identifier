# Dwelling model: run guide

How to run the dwelling model from a notebook on the four inputs, read every output, and turn the results into findings you can act on.

1. [What the model does](#1-what-the-model-does)
2. [Set up](#2-set-up)
3. [Prepare the four inputs](#3-prepare-the-four-inputs)
4. [Check the column mapping](#4-check-the-column-mapping)
5. [Trial run on a small area](#5-trial-run-on-a-small-area)
6. [Full run and its log](#6-full-run-and-its-log)
7. [What you get back](#7-what-you-get-back)
8. [Reading the results](#8-reading-the-results)
9. [Finding insights](#9-finding-insights)
10. [Maps](#10-maps)
11. [Reusing the model](#11-reusing-the-model)
12. [Settings](#12-settings)
13. [Troubleshooting](#13-troubleshooting)
14. [Assumptions and limits](#14-assumptions-and-limits)

The code blocks run top to bottom in one notebook. Replace the `D:/data/...` paths with your own.

---

## 1. What the model does

You have a list of PD (private dwelling) addresses. Every other address is not a PD. The **number** of PD labels on each parcel is assumed right. The **choice of address** is sometimes wrong, and the label sits on another address of the same parcel:

| Labelled PD | Should be |
|---|---|
| `20 X ST` (the unit block's street address) | `UNIT 5, 20 X ST` |
| `10 X ST` | `10B X ST` |
| `THE ELMS, 84 X CRES` (the building name) | `THE ELMS, UNIT 201, 84 X CRES` |

The model works in two steps:

1. **It learns what a dwelling address looks like, from every parcel at once.** It uses the address text (flat and level parts, suffixes, ranges), where the geocode sits (inside a building, stacked with others, near the parcel edge), the buildings on the parcel, and how addresses nest under each other (units under a street address). Each parcel's probabilities come from a model trained on the other parcels, so a parcel's own labels never vouch for themselves.
2. **It keeps each parcel's count and picks the addresses.** A parcel with *k* PD labels gets exactly *k* dwellings. A label moves only when the evidence beats the model's estimate of how often labels are misplaced. That estimate is learned from your data and reported (see [8.7](#87-the-model-report)).

Then it names each address's **role**:

- `DWELLING`: chosen as a dwelling.
- `PRIMARY_SHELL`: the top address that dwellings nest under, such as a unit block's street address.
- `SUB_SHELL`: a shell inside a primary shell, such as a tower or level in a complex.
- `OUT_OF_SCOPE`: everything else, such as shops, car spaces and empty parcels.

What it does **not** do:

- **Fix wrong counts.** If a parcel has a missing or extra PD label, it keeps that count. It does point you to such parcels ([9.4](#94-parcels-whose-count-looks-wrong)).
- **Move a label to another parcel.**
- **Know storeys without height data** ([3](#3-prepare-the-four-inputs), `bld`).

---

## 2. Set up

Once per environment, in a notebook cell:

```python
%pip install -e "F:/Claude Code Projects/Shell Identifier"
```

`-e` links the package to the repo folder, so after a `git pull` you only **restart the kernel**; there's nothing to reinstall. A notebook keeps the code it imported until the kernel restarts. To pick up changes without restarting, put this at the top of the notebook:

```python
%load_ext autoreload
%autoreload 2
```

The imports used throughout this guide:

```python
import polars as pl

from shell_identifier import check_frames, load_config, predict_frames, verify_frames
from shell_identifier.dwellings import EMSettings

pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(60)
```

---

## 3. Prepare the four inputs

| Input | Required columns | Optional columns that improve the model |
|---|---|---|
| `labelled_asset` | `ADDRESS_ID` | `PD_FLAG`: rows that are true (`TRUE`/`1`/`Y`/`YES`/`PD`, boolean or 0/1) are PD. Other rows are ignored. Without the column, every row is a PD. |
| `ds_asset` | `ADDRESS_ID`, `LATITUDE`, `LONGITUDE`, and the G-NAF address fields (`NUMBER_FIRST`, `STREET_NAME` at least) | `FLAT_TYPE_CODE`, `FLAT_NUMBER`, `LEVEL_*`, `NUMBER_LAST`, suffixes, `BUILDING_NAME`, `LOCALITY_NAME`, `POSTCODE`, `PRIMARY_SECONDARY`, `LEGAL_PARCEL_ID`, `GEOCODE_TYPE_CODE`, `DATE_RETIRED` (rows with a value are dropped) |
| `cad` | `parcel_id`, a WKT polygon column | `cadastre_polygon_pid`, strata counts (`total_strata_count`, …). Base parcels only. |
| `bld` | `building_pid`, a WKT polygon column | `roof_height` / `ROOF_HGT`, `total_floor_area` / `TOT_FLR_AR`, `building_use` / `BLD_USE`, `building_area` |

Column names are matched case-insensitively, and G-NAF and Geoscape names are recognised. WKT columns are found by name (`WKT`, `GEOM_WKT`, `SHAPE`, …) or by their contents. EWKT (`SRID=7844;POLYGON…`) and WKB work too.

Before you run, check these rules:

1. **`ds_asset` must cover the labelled area and nothing more.** Any address without a PD flag is read as "not PD". An unlabelled suburb would look like a suburb with no dwellings, and the model would learn from it as if that were true. Filter `ds_asset` to the area the labels cover.
2. **`parcel_id` must be unique.** Polygons that share a `parcel_id` are treated as one parcel. In a multi-state extract, the same lot/plan id can exist in two states. Either run one state at a time, or add the state to the id before passing `cad` in.
3. **Coordinates are GDA2020 lon/lat by default (`EPSG:7844`).** Pass `crs="EPSG:4283"` for GDA94, or a per-table value such as `crs={"cad": "EPSG:7856"}` for projected WKT. The run stops with an explanation if latitude and longitude look swapped, or if the numbers aren't lon/lat.
4. **Building heights are optional but useful.** With only footprints, the model can't tell a 1-storey building from a 10-storey one. If you have Geoscape roof heights or floor areas, join them onto `bld` by `building_pid` first.

Read the inputs as LazyFrames. Only the columns the model needs are read.

```python
frames = dict(
    labelled_asset=pl.scan_parquet("D:/data/labelled_asset.parquet"),  # ADDRESS_ID, PD_FLAG
    ds_asset=pl.scan_parquet("D:/data/ds_asset.parquet"),              # ADDRESS_ID, G-NAF fields, LATITUDE, LONGITUDE
    cad=pl.scan_parquet("D:/data/cad.parquet"),                        # cadastre_polygon_pid, parcel_id, WKT
    bld=pl.scan_parquet("D:/data/bld.parquet"),                        # building_pid, WKT
)
```

`pl.scan_csv` works the same way. To limit the run to an area, filter `ds_asset` before passing it in, for example `.filter(pl.col("STATE_ABBREVIATION") == "NSW")`.

---

## 4. Check the column mapping

This is fast: it reads a sample, not the data.

```python
print(check_frames(**frames))
```

An abridged example:

```
labelled_asset (PD addresses; every other address is not PD): polars LazyFrame labelled_asset, 3,197 rows
  address_pid              <- ADDRESS_ID
  label                    <- PD_FLAG

ds_asset (addresses): polars LazyFrame ds_asset, 3,985 rows
  geometry: longitude / latitude; crs: EPSG:7844
  address_pid              <- ADDRESS_ID
  longitude                <- LONGITUDE
  latitude                 <- LATITUDE
  flat_type_code           <- FLAT_TYPE_CODE
  ...
  primary_secondary           (not found, optional)
  geocode_type                (not found, optional)

bld (building footprints): polars LazyFrame bld, 1,164 rows
  geometry: WKT; crs: EPSG:7844
  building_pid             <- building_pid
  roof_height                 (not found, optional)

cad (cadastre parcels): polars LazyFrame cad, 702 rows
  geometry: WKT; crs: EPSG:7844
  parcel_pid               <- parcel_id
  polygon_pid              <- cadastre_polygon_pid
```

How to read it:

- `field <- COLUMN` means your column feeds that field.
- `(NOT FOUND - required)` means the run will fail. Map the column with `columns=` (below).
- `(not found, optional)` means the model runs without it. The features that need it are left out and listed in the run log.
- `label (not found: every row is a PD)` means there's no flag column, so every `labelled_asset` row counts as a PD.
- `geometry:` names the WKT column it found. `NONE FOUND` means you need to name it.
- `unused columns:` lists columns it didn't recognise. Check there's nothing there you meant to use.

If a column has a different name, map it. Pass the same `columns=` to every call that follows (`verify_frames`, `predict_frames`):

```python
columns = {
    "ds_asset": {"latitude": "LATITUDE", "longitude": "LONGITUDE"},   # e.g. {"latitude": "LAT_GDA2020"}
    "cad": {"geometry": "WKT"},                                         # e.g. {"geometry": "GEOM_TEXT"}
}
print(check_frames(**frames, columns=columns))
```

---

## 5. Trial run on a small area

Start with a few suburbs. This shows how long a run takes, and catches data problems early.

```python
trial = verify_frames(
    **frames,
    bbox=(151.00, -33.90, 151.10, -33.80),        # lon/lat box: min lon, min lat, max lon, max lat
    settings=EMSettings(folds=3, max_iter=4),     # faster than the defaults (5 folds, 8 iterations)
    out_dir="results/trial",
)
```

`bbox` cuts all three spatial tables to the box. Labels outside the box don't match any address and are reported in the log, which is expected here.

Things to check in the trial:

- **Labels matched:** the log says how many PD labels matched an address.
- **Parcels matched:** few addresses fall outside every parcel.
- **Labels moved:** the number of moved labels looks plausible, and moved labels at the top of `trial["moves"]` look right when you read them (see [8.6](#86-the-moves-table)).

---

## 6. Full run and its log

```python
result = verify_frames(**frames, out_dir="results/dwellings")
```

Progress prints in the cell as it runs. To add timestamps, pass your own logger: `log=lambda m: print(time.strftime("%H:%M:%S"), m)`, with `import time`. Runs are deterministic: the same inputs give the same results.

An example log (synthetic data):

```
ds_asset: polars LazyFrame ds_asset: 3,985 addresses
cad: polars LazyFrame cad: 702 base parcel polygons, 0 strata polygons
bld: polars LazyFrame bld: 1,164 footprints
labelled_asset: polars LazyFrame labelled_asset: 3,197 PD addresses (PD_FLAG true)
3,985 addresses on 702 parcels (655 with a PD label); 3,197 PD labels, 0 of them on addresses outside every parcel; all other addresses are taken as not PD
8 features have no data (or too little to use) and are left out: b_levels, b_use_residential, ...
EM iteration 1: 108 assignments changed, 54 PD labels moved, P(parcel has a misplaced label) = 0.220
EM iteration 2: 54 assignments changed, 79 PD labels moved, P(parcel has a misplaced label) = 0.262
EM iteration 3: 20 assignments changed, 87 PD labels moved, P(parcel has a misplaced label) = 0.297
EM iteration 4: 6 assignments changed, 90 PD labels moved, P(parcel has a misplaced label) = 0.305
```

| Log line | What it means | What to check |
|---|---|---|
| `ds_asset: … addresses` | Addresses read, after dropping retired rows | Matches your expected count |
| `cad: … base parcel polygons` | Parcel polygons read | Not zero; strata polygons only appear if `cad` has them |
| `labelled_asset: … PD addresses (PD_FLAG true); N rows with other values ignored` | PD rows kept, plus the values of the rows dropped | The ignored values really mean "not PD" (otherwise set `pd_values=`) |
| `N PD labels have an address id that is not in ds_asset` | Labels whose `ADDRESS_ID` isn't an address in this run | Small, or explained by a filter or `bbox` |
| `… addresses on … parcels (… with a PD label); … of them on addresses outside every parcel` | How the counts were set up | Few labels outside parcels; those can't be checked against a count |
| `N features have no data (or too little to use)` | Features your inputs can't supply | Expected without heights, building use or strata ([3](#3-prepare-the-four-inputs)) |
| `EM iteration i: …` | One round of refit and reassign | `assignments changed` should drop quickly, then stay small. It stops early when fewer than 0.05% of addresses change. A handful that keep flipping between rounds are near-ties, and that's normal. |

---

## 7. What you get back

`result` is a dict:

| Key | What it is |
|---|---|
| `result["assignments"]` | One row per address: the decision, the reason, and every address field and feature behind it |
| `result["parcels"]` | One row per parcel: counts, roles, status, review flag |
| `result["moves"]` | One row per moved label: from which address, to which, and how confidently |
| `result["bundle"]` | The fitted model, to reuse with `predict_frames` |
| `result["features"]` | The feature matrix the model was fitted on |
| `result["paths"]` | The files written to `out_dir` |

The files in `out_dir`:

| File | Contents |
|---|---|
| `dwelling_assignments.csv` / `.parquet` | Per address: the main columns of `result["assignments"]` (below) |
| `dwelling_assignments.gpkg` | The same, as points, for QGIS |
| `parcel_dwellings.csv` | `result["parcels"]` |
| `label_moves.csv` | `result["moves"]` |
| `model_report.md` | Headline numbers, EM iterations, most influential features |
| `dwelling_model.joblib` | The fitted model |

The files hold the main columns only. In memory, `result["assignments"]` also has the address fields (`locality_name`, `postcode`, `flat_type_code`, …), `node_type`, coordinates `x`/`y`, and the label-free shell score. To get those from the files, join back to `ds_asset` on `ADDRESS_ID`.

---

## 8. Reading the results

### 8.1 Per-address columns

| Column | Meaning |
|---|---|
| `address_label` | The address as text |
| `parcel_pid` | The parcel it falls in, or the nearest parcel within 5 m. Empty if neither. |
| `mode` | `verified`: on a parcel, checked against the parcel's count. `predicted`: outside every parcel, so the model alone decides. `counted`: from `predict_frames` with given counts. |
| `input_label` | `PD` if it had a PD flag, else `OOS` |
| `dwelling_prob` | The model's probability that this is a dwelling, from features alone. It ignores the labels and the count, and comes from a model that never saw this parcel. |
| `posterior` | The probability after combining `dwelling_prob` with the parcel's count and its labels. This drives the decision. |
| `rank_in_parcel` | 1 = the most dwelling-like address on the parcel |
| `assigned_dwelling` | The decision: true/false |
| `role` | `DWELLING`, `PRIMARY_SHELL`, `SUB_SHELL` or `OUT_OF_SCOPE` (see 8.2) |
| `label_change` | `PD->OOS`: a PD label was moved away from this address. `OOS->PD`: this address received a moved label. Empty: no change. |
| `primary_shell_pid` / `_label` | The primary shell above this address (for a shell, itself) |
| `dwellings_below` | Number of dwellings nested beneath this address |
| `shell_probability` | The label-free expert score of how shell-like the address is (0–1) |
| `parent_pid` | The address this one nests under (e.g. unit → street address) |
| `evidence` | What the model saw, in words |
| `decision` | Why it was assigned, in words |

An example of the last two:

```
evidence: street address; geocode outside buildings; 8 nested beneath; parcel 9 address(es), 1 dwelling-size building(s)
decision: parcel has 8 dwelling(s); ranked 9/9, posterior 0.0
```

This is the street address of an 8-unit block. It was labelled PD, ranked last of 9, and its label moved to a unit.

### 8.2 Roles

A typical unit block on one parcel:

```
20 X ST                    PRIMARY_SHELL   dwellings_below = 6
├── UNIT 1, 20 X ST        DWELLING        primary_shell = 20 X ST
├── ...
└── UNIT 6, 20 X ST        DWELLING
SHOP 1, 20 X ST            OUT_OF_SCOPE    (not a dwelling, nothing beneath it)
```

- A non-dwelling counts as a shell only if it has **two or more** dwellings beneath it, or scores as shell-like on its own. A shop with one flat beside it (`70` / `70A`) stays `OUT_OF_SCOPE`: it's an occupancy, not a container.
- A `SUB_SHELL` is a shell inside a primary shell: a tower or level address in a large complex.
- On a parcel with no PD labels, every address is `OUT_OF_SCOPE`.

### 8.3 Label changes

Within a parcel the count is fixed, so changes come in pairs. Every `PD->OOS` on a parcel has a matching `OOS->PD` on the same parcel. `result["moves"]` pairs them up for you (8.6).

### 8.4 `dwelling_prob` vs `posterior`

| `dwelling_prob` | `posterior` | Reading |
|---|---|---|
| high | high | Looks like a dwelling, and the count agrees |
| low | low | Looks like a non-dwelling, and the count agrees |
| low | high | Kept as a dwelling **because the count needs it**. If this happens often on a parcel, its count may be too high (9.4). |
| high | low | Looks like a dwelling but the count is used up. The count may be too low. |

### 8.5 Parcel summary (`result["parcels"]`)

| Column | Meaning |
|---|---|
| `n_addresses` | Addresses on the parcel |
| `labelled_pd` | PD labels on it: the count the model kept |
| `assigned_dwellings` | Dwellings chosen. Equals `labelled_pd` for every `verified` parcel, unless `counts=` gives the parcel another count. |
| `expected_dwellings` | Sum of `dwelling_prob`: how many dwellings the model expects from features alone, ignoring the count |
| `n_label_changes` | PD labels moved away |
| `n_primary_shells`, `n_sub_shells`, `n_out_of_scope` | Role counts |
| `margin` | The weakest chosen address's `posterior` minus the strongest unchosen one's. Near 1 is clear-cut; near 0 or negative is a near tie. |
| `status` | `AGREES_WITH_LABELS`, `LABELS_MOVED` or `PREDICTED` (no count) |
| `review` | `margin < 0.3`: the choice on this parcel is close |

### 8.6 The moves table

| Column | Meaning |
|---|---|
| `from_pid`, `from_label`, `from_posterior` | The address that had the PD label |
| `to_pid`, `to_label`, `to_posterior` | The address that gets it |
| `confidence` | `to_posterior − from_posterior`. 1.0 is a clear swap; values near 0 are coin flips. |

Sorted most confident first. With `counts=` (section 11), a parcel's count can differ from its labels, so one side of a row can be empty.

### 8.7 The model report

`model_report.md` has three parts.

- **Headline numbers:** parcels checked, PD labels, labels moved, parcels flagged for review, and role counts.
- **`Estimated misplaced PD labels per parcel, P(m = 0, 1, 2, ...)`:** for parcels where a label *could* be misplaced, this is the estimated share with 0, 1, 2 and 3 misplaced labels. Those are parcels with both PD and non-PD addresses. `1 − P(m = 0)` is the model's estimate of the label error rate on such parcels. It's learned from your data, starting from a prior of 25%.
- **EM iterations:** `assignments_changed` should drop quickly, then stay small. A handful flipping back and forth are near-ties between addresses. If it's still a large share of addresses at the last iteration, raise `max_iter` (section 12).
- **Most influential features:** how much worse the model gets when each feature is scrambled. Feature-name prefixes:

  | Prefix | Family |
  |---|---|
  | `t_` | address text |
  | `g_` | geocode placement |
  | `b_` | the address's building |
  | `p_` | parcel buildings |
  | `c_` | parcel composition |
  | `x_` / `f_` | nesting and shell evidence |

  `x_shell_logit` is the expert shell score. On data without building heights, expect `t_` and `x_` features to dominate.

---

## 9. Finding insights

Short names for the three tables:

```python
a, p, m = result["assignments"], result["parcels"], result["moves"]
verified = p.filter(pl.col("mode") == "verified")
```

### 9.1 Headline numbers

```python
n_pd = int((a["pd_label"] == 1).sum())
n_moved = int((a["label_change"] == "PD->OOS").sum())
touched = int((verified["n_label_changes"] > 0).sum())
print(f"PD labels: {n_pd:,}")
print(f"Labels moved: {n_moved:,} ({n_moved / n_pd:.2%} of PD labels) on {touched:,} parcels "
      f"({touched / verified.height:.2%} of parcels)")
print(f"Parcels flagged for review: {int(verified['review'].sum()):,}")
print(a["role"].value_counts(sort=True))
```

These are the numbers to report: what share of labels were on the wrong address, and how many parcels it affects.

### 9.2 Where the errors are

Which kind of address the labels were on, and which kind they moved to:

```python
kind = a.select("address_pid", "node_type")
moves_typed = (
    m.join(kind.rename({"address_pid": "from_pid", "node_type": "from_type"}), on="from_pid", how="left")
     .join(kind.rename({"address_pid": "to_pid", "node_type": "to_type"}), on="to_pid", how="left")
)
moves_typed.group_by("from_type", "to_type").len().sort("len", descending=True)
```

How to read the common pairs:

- `street → dwelling`: labels were put on the street address of a unit block instead of a unit. Usually the most common pattern.
- `street → street`: usually `10` → `10B`. Suffixed numbers are street addresses too.
- `commercial`, `ancillary` or `structure → dwelling`: labels on a shop, car space or building name instead of a unit.

`node_type` values:

| Value | Meaning |
|---|---|
| `street` | a bare street address |
| `dwelling` | a unit or flat |
| `commercial` | a shop, office or suite |
| `ancillary` | car space, storage |
| `level` | a level address |
| `structure` | a building or tower name |
| `secondary` | other sub-address |

By locality: which areas have the most misplaced labels.

```python
by_area = (
    a.filter(pl.col("mode") == "verified")
     .group_by("locality_name")
     .agg(addresses=pl.len(),
          pd_labels=(pl.col("pd_label") == 1).sum(),
          moved=(pl.col("label_change") == "PD->OOS").sum())
     .with_columns(moved_share=(pl.col("moved") / pl.col("pd_labels")).round(4))
     .sort("moved", descending=True)
)
by_area.head(20)
```

A locality with a much higher `moved_share` than the rest often points to one labeller or one batch with a habit, such as always picking the street address.

### 9.3 A review queue

The least clear-cut moves first, so review time goes where the model is least sure:

```python
queue = (
    m.join(p.select("parcel_pid", "n_addresses", "margin"), on="parcel_pid", how="left")
     .sort("confidence")
)
queue.write_csv("results/dwellings/review_queue.csv")
queue.head(10)
```

Also check `p.filter(pl.col("review"))`: parcels where the choice was close, whether or not a label moved.

### 9.4 Parcels whose count looks wrong

The model keeps each parcel's count. `expected_dwellings` is what the features alone suggest. A large gap between the two points to a **missing or extra PD label**, which moving labels can't fix:

```python
count_suspects = (
    verified.with_columns(gap=(pl.col("expected_dwellings") - pl.col("labelled_pd")).round(2))
            .filter(pl.col("gap").abs() >= 1)
            .sort(pl.col("gap").abs(), descending=True)
            .select("parcel_pid", "n_addresses", "labelled_pd", "expected_dwellings", "gap")
)
count_suspects.head(20)
```

- **Positive gap:** more dwelling-like addresses than labels. These may be dwellings that were never labelled, for example a parcel of units with no PD flag at all.
- **Negative gap:** more labels than dwelling-like addresses, for example a commercial parcel labelled PD.

On real data, large complexes produce small gaps routinely. Look at the largest gaps first.

### 9.5 Shell inventory

```python
shells = (
    a.filter(pl.col("role") == "PRIMARY_SHELL")
     .select("address_pid", "address_label", "parcel_pid", "dwellings_below", "input_label")
     .sort("dwellings_below", descending=True)
)
print(shells.group_by("input_label").len())   # PD: shells that carried a PD label (now moved off)
shells.head(10)
```

### 9.6 How often are the moves right?

The model's confidence is not a measured accuracy. To measure it, check a sample by hand, stratified by confidence:

```python
banded = m.with_columns(
    band=pl.when(pl.col("confidence") >= 0.9).then(pl.lit("1 high (>= 0.9)"))
           .when(pl.col("confidence") >= 0.5).then(pl.lit("2 medium (0.5-0.9)"))
           .otherwise(pl.lit("3 low (< 0.5)"))
)
sample = pl.concat([g.sample(min(25, g.height), seed=0) for g in banded.partition_by("band")])
sample.with_columns(correct=pl.lit(None, dtype=pl.Boolean)).write_csv("results/dwellings/move_sample.csv")
```

Fill in `correct` (true/false) for each row in Excel, save, and read it back:

```python
checked = pl.read_csv("results/dwellings/move_sample.csv", infer_schema=False)
answer = pl.col("correct").fill_null("").str.strip_chars().str.to_uppercase()
checked = checked.with_columns(correct=pl.when(answer.is_in(["TRUE", "Y", "YES", "1"])).then(True)
                                         .when(answer.is_in(["FALSE", "N", "NO", "0"])).then(False))
checked.group_by("band").agg(sampled=pl.len(), checked=pl.col("correct").count(),
                             precision=pl.col("correct").mean()).sort("band")
```

Blank rows don't count, so you can read it back part-way through.

From the measured precision in each band you can decide which moves to accept automatically and which to review. For example, accept the high band and review the rest.

### 9.7 Data quality checks

```python
a.select(
    addresses=pl.len(),
    outside_parcels=(pl.col("parcel_pid") == "").sum(),
    snapped_to_parcel=(pl.col("parcel_match") == "snapped").sum(),
    outside_buildings=pl.col("building_pid").is_null().sum(),
    shared_geocode=(pl.len().over("parcel_pid", "xy_key") > 1).sum(),
)
```

- **`outside_parcels`:** these addresses are `mode = predicted`, with no count to check against. Many of them suggest coordinates and cadastre don't line up (wrong CRS, or `cad` missing an area).
- **`outside_buildings`:** high if `bld` has gaps, or geocodes sit at parcel centroids. That weakens the building features.
- **`shared_geocode`:** units often share one point. That's normal, but it means location can't tell them apart, only the text can.

### 9.8 Why did the model decide this?

```python
pid = m["to_pid"][0]                                # any address id
a.filter(pl.col("address_pid") == pid).select(
    "address_label", "input_label", "role", "label_change", "dwelling_prob", "posterior",
    "decision", "evidence", "reasons",
)
```

`reasons` lists the strongest signals from the expert shell score, each with its weight in brackets. Positive weights push towards "shell", negative ones away. For example, `sub-address with nothing nested under it [-4.0]` means the address looks like a unit, not a container.

---

## 10. Maps

**In QGIS:**
1. Open `dwelling_assignments.gpkg` from `out_dir`.
2. Style it by category on `role` or `label_change`.
3. Add your cadastre and buildings as layers.

This is the best way to browse a whole area.

**In the notebook**, one parcel at a time:

```python
import geopandas as gpd
import matplotlib.pyplot as plt

from shell_identifier.dwellings.pipeline import ASSIGNMENT_COLUMNS

points = gpd.GeoDataFrame(a.select(ASSIGNMENT_COLUMNS).to_pandas(),
                          geometry=gpd.points_from_xy(a["x"], a["y"]), crs=result["crs"])

def to_gdf(frame, wkt="WKT", crs="EPSG:7844"):
    df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    return gpd.GeoDataFrame(df.drop(wkt).to_pandas(), geometry=gpd.GeoSeries.from_wkt(df[wkt].to_list()), crs=crs)

parcels, buildings = to_gdf(frames["cad"]), to_gdf(frames["bld"])   # once; slow on millions of rows

def show_parcel(parcel_pid, pad=0.0005):
    p_geom = parcels[parcels["parcel_id"] == parcel_pid]
    x0, y0, x1, y1 = p_geom.total_bounds
    near = buildings.cx[x0 - pad:x1 + pad, y0 - pad:y1 + pad]
    pts = points[points["parcel_pid"] == parcel_pid]
    utm = p_geom.estimate_utm_crs()                  # metres, so shapes aren't squashed
    ax = p_geom.to_crs(utm).plot(figsize=(7, 7), facecolor="#e8f0fe", edgecolor="#1a73e8")
    near.to_crs(utm).plot(ax=ax, facecolor="#dadce0", edgecolor="#5f6368")
    pts.to_crs(utm).plot(ax=ax, column="role", categorical=True, legend=True, markersize=70,
                         edgecolor="black", zorder=3)
    ax.set_title(parcel_pid)
    ax.set_axis_off()
    plt.show()
    return (pts[["address_label", "input_label", "role", "label_change", "posterior", "evidence"]]
            .sort_values("posterior", ascending=False))

show_parcel(m["parcel_pid"][0])                     # the most confident move
```

Units often share one geocode, so their points sit on top of each other. The table under the map has the per-address detail. Good parcels to look at:

- `queue["parcel_pid"]`: least confident moves first.
- `p.filter(pl.col("review"))["parcel_pid"]`: close calls.
- `count_suspects["parcel_pid"]`: counts that look wrong.

For a map you can pan, run `%pip install folium mapclassify`, then `points.explore(column="role", tooltip=["address_label", "role", "label_change"])`.

---

## 11. Reusing the model

Apply the fitted model to an area without labels:

```python
pred = predict_frames("results/dwellings/dwelling_model.joblib",
                      ds_asset=frames["ds_asset"], cad=frames["cad"], bld=frames["bld"])
pred["assignments"]["mode"].value_counts()
```

`result["bundle"]` works in place of the file path.

- **Without counts:** every address is `predicted`, and is a dwelling when `dwelling_prob >= 0.5`.
- **With trusted counts:** add `counts=`, a frame of `parcel_id, n_dwellings`. Those parcels become `counted` and get exactly that many dwellings, the most dwelling-like first.

`counts=` also works in `verify_frames`, where it overrides the label count for the parcels it lists.

A model trained in one area transfers best to areas with similar housing and similar data. A model trained on a city may not suit a rural area.

---

## 12. Settings

**Model settings**, `settings=EMSettings(...)`:

| Setting | Default | Effect |
|---|---|---|
| `folds` | 5 | Cross-fitting folds. Fewer is faster; 3 is fine for trials. |
| `max_iter` | 8 | Most EM rounds. Raise it if `assignments_changed` is still large at the end. |
| `max_misplaced` | 3 | Most misplaced labels considered per parcel |
| `misplacement_prior` | `(0.75, 0.18, 0.05, 0.02)` | Starting guess for P(0, 1, 2, 3 misplaced labels) |
| `prior_strength` | 50 | How strongly the estimate is held to the starting guess |
| `estimate_prior` | `True` | Learn the misplacement rate from the data. `False` keeps the starting guess. |
| `clf_params` | gradient boosting, 150 trees | Passed to scikit-learn's `HistGradientBoostingClassifier` |

**Pipeline settings**, `cfg=`: start from the defaults and change what you need.

```python
cfg = load_config()
cfg["spatial"]["parcel_snap_distance_m"]   # 5.0: addresses within this distance of a parcel are snapped to it
cfg["spatial"]["storey_height_m"]          # 3.1: storeys = roof height / this, when only roof height is known
# then: verify_frames(**frames, cfg=cfg, ...)
```

---

## 13. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `no PD label matched an address` | `ADDRESS_ID` in the labels doesn't match `ADDRESS_ID` in `ds_asset` | Check both are the same id (type and format), and that `ds_asset` isn't filtered to another area |
| `labelled_asset: no row has a PD value in PD_FLAG` | The flag uses values the model doesn't recognise | `pd_values=["your value"]` |
| `latitude and longitude look swapped` | Lat and lon columns are the wrong way round | `columns={"ds_asset": {"latitude": ..., "longitude": ...}}` |
| `… which is not lon/lat in EPSG:7844` | Projected coordinates or WKT | `crs={"cad": "EPSG:78xx"}` etc. |
| `no address could be matched to a parcel` | Addresses and parcels don't overlap | Check the CRS of both, and that `cad` covers the area |
| `cad: no geometry column found` / `bld: …` | WKT column not recognised | `columns={"cad": {"geometry": "YOUR_WKT_COLUMN"}}` |
| `… no parcel id column` | `parcel_id` under another name | `columns={"cad": {"parcel_pid": "YOUR_ID"}}` |
| An error about a column that should exist (e.g. `b_levels`) | The notebook is running old code | Pull, then restart the kernel |
| Many `features have no data` | Inputs lack heights, building use, strata | Expected; add the columns if you have them ([3](#3-prepare-the-four-inputs)) |
| Run is slow or runs out of memory | Very large area | Try `folds=3`; run one region (state, LGA) at a time; turning WKT into shapes for millions of buildings takes a while |

---

## 14. Assumptions and limits

- **Each parcel's PD count is right.** Wrong counts are kept, but you can find them ([9.4](#94-parcels-whose-count-looks-wrong)).
- **Every address without a PD flag is not PD.** `ds_asset` must match the labelled area.
- **Labels move only within a parcel,** and at most `max_misplaced` (3) per parcel.
- **Addresses outside every parcel** are decided by the model alone (`mode = predicted`). Their label changes have no count behind them, so treat them separately.
- **Building storeys and use** are only known if `bld` carries heights, floor areas or use. Without them, the model leans on the address text and how addresses nest.
- **Parcels are keyed by `parcel_id`.** It must be unique within a run.

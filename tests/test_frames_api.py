"""The LazyFrame API, with inputs in this layout:

labelled_asset: ADDRESS_ID, PD_FLAG (true on every row: the PD addresses; the rest are not PD)
ds_asset:       ADDRESS_ID, G-NAF address fields, LATITUDE, LONGITUDE
cad:            cadastre_polygon_pid, parcel_id, WKT polygon (base parcels only)
bld:            building_pid, WKT polygon
"""

import numpy as np
import polars as pl
import pytest
import shapely

from shell_identifier import check_frames, predict_frames, score_frames, verify_frames
from shell_identifier.api import frame_datasets, load_frames
from shell_identifier.assets import load_label_table
from shell_identifier.dwellings import EMSettings
from shell_identifier.dwellings.evaluate import evaluate_dwellings
from shell_identifier.dwellings.pipeline import evidence
from shell_identifier.loaders import load_gnaf

FIELDS = ["building_name", "lot_number_prefix", "lot_number", "lot_number_suffix", "flat_type_code",
          "flat_number_prefix", "flat_number", "flat_number_suffix", "level_type_code", "level_number_prefix",
          "level_number", "level_number_suffix", "number_first_prefix", "number_first", "number_first_suffix",
          "number_last_prefix", "number_last", "number_last_suffix", "street_name", "street_type_code",
          "street_suffix_code", "locality_name", "postcode", "alias_principal", "primary_secondary", "legal_parcel_id"]
QUIET = {"log": lambda m: None}
FAST = EMSettings(folds=3, max_iter=4)


def _wkt(parquet) -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_parquet(parquet)
    return df, shapely.to_wkt(shapely.from_wkb(df["geometry"].ext.storage().to_numpy()),
                              rounding_precision=9).tolist()


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from shell_identifier.synthetic import generate

    out = tmp_path_factory.mktemp("frames")
    generate(out, n_parcels=300, seed=8)
    return out


@pytest.fixture(scope="module")
def frames(synth):
    g = load_gnaf(synth / "G-NAF").df
    ds = g.select(pl.col("address_pid").alias("ADDRESS_ID"), *[pl.col(f).alias(f.upper()) for f in FIELDS],
                  pl.col("state").alias("STATE_ABBREVIATION"), pl.col("geocode_type").alias("GEOCODE_TYPE_CODE"),
                  pl.col("y").alias("LATITUDE"), pl.col("x").alias("LONGITUDE"))
    lab = pl.read_csv(synth / "truth" / "pd_labels.csv").filter(pl.col("label") == "PD").select(
        pl.col("address_detail_pid").alias("ADDRESS_ID"), pl.lit(True).alias("PD_FLAG"))
    cad, wkt = _wkt(synth / "assets" / "cad.parquet")
    cad = (cad.with_columns(geom_wkt=pl.Series(wkt)).filter(pl.col("base_parcel") == "Yes")
           .select("cadastre_polygon_pid", "parcel_id", "geom_wkt"))
    bld, wkt = _wkt(synth / "assets" / "bld.parquet")
    bld = bld.select("building_pid").with_columns(WKT=pl.Series(wkt))
    return {"labelled_asset": lab.lazy(), "ds_asset": ds.lazy(), "cad": cad.lazy(), "bld": bld.lazy()}


def _truth(synth):
    return pl.read_csv(synth / "truth" / "shell_truth.csv", infer_schema=False)


def test_check_frames_maps_the_layout(frames):
    report = check_frames(**frames)
    assert "address_pid              <- ADDRESS_ID" in report
    assert "label                    <- PD_FLAG" in report
    assert "parcel_pid               <- parcel_id" in report
    assert "geometry: geom_wkt" in report and "geometry: WKT" in report
    assert "NOT FOUND" not in report


def test_verify_frames_end_to_end(frames, synth, tmp_path):
    res = verify_frames(**frames, settings=FAST, out_dir=tmp_path / "out", gpkg=False, **QUIET)
    m, _ = evaluate_dwellings(res["assignments"].with_columns(pl.col("assigned_dwelling").cast(pl.Utf8)),
                              _truth(synth))
    assert m["wrong_labels"] > 0
    assert m["wrong_labels_fixed"] >= 0.8 * m["wrong_labels"]
    assert m["correct_labels_broken"] <= 0.01 * m["addresses"]
    assert m["shell_precision"] > 0.95 and m["shell_recall"] > 0.95
    assert (tmp_path / "out" / "dwelling_assignments.parquet").exists()
    assert "b_levels" not in res["bundle"]["features"]            # no data for it: left out, not crashed
    ev = res["assignments"]["evidence"]
    assert ev.str.contains("in a building with").any() and not ev.str.contains("storey").any()
    # the fitted features alone (no b_levels) are enough to rebuild the text
    assert evidence(res["assignments"], res["features"]).equals(ev, check_names=False)

    pred = predict_frames(res["bundle"], ds_asset=frames["ds_asset"], cad=frames["cad"], bld=frames["bld"], **QUIET)
    assert (pred["assignments"]["mode"] == "predicted").all()
    agree = (pred["assignments"]["assigned_dwelling"] == res["assignments"]["assigned_dwelling"]).mean()
    assert agree > 0.97


def test_label_layouts_agree(frames):
    """PD rows only, a Y/N flag on every address, and an id column alone give the same PD set."""
    pd_ids = sorted(frames["labelled_asset"].collect()["ADDRESS_ID"].to_list())
    yn = frames["ds_asset"].select("ADDRESS_ID", PD_FLAG=pl.when(pl.col("ADDRESS_ID").is_in(pd_ids))
                                   .then(pl.lit("Y")).otherwise(pl.lit("N")))
    for lab in (frames["labelled_asset"], yn, frames["labelled_asset"].select("ADDRESS_ID")):
        spec = frame_datasets(lab)["labelled_asset"]
        assert sorted(load_label_table(spec, **QUIET)["address_pid"].to_list()) == pd_ids


def test_eager_frames_numeric_flags_and_ewkt(frames, synth):
    lab = frames["labelled_asset"].with_columns(pl.lit(1, dtype=pl.Int8).alias("PD_FLAG")).collect()
    cad = frames["cad"].with_columns(("SRID=7844;" + pl.col("geom_wkt")).alias("SHAPE")).drop("geom_wkt").collect()
    inputs = load_frames(lab, frames["ds_asset"].collect(), cad, frames["bld"].collect(), **QUIET)
    assert inputs.labels.height == lab.height and (inputs.labels["y"] == 1).all()
    assert len(inputs.parcels) == cad.height and inputs.parcels.crs == "EPSG:7844"
    assert inputs.buildings.df["building_area"].min() > 0          # measured from the footprint


def test_column_overrides(frames):
    ds = frames["ds_asset"].rename({"ADDRESS_ID": "GNAF_KEY", "LATITUDE": "Y_COORD", "LONGITUDE": "X_COORD"})
    inputs = load_frames(None, ds, frames["cad"], None, columns={
        "ds_asset": {"address_pid": "GNAF_KEY", "latitude": "Y_COORD", "longitude": "X_COORD"}}, **QUIET)
    assert inputs.addresses.df["address_pid"].str.starts_with("GA").all()


def test_swapped_lat_long_is_caught(frames):
    ds = frames["ds_asset"].rename({"LATITUDE": "LONGITUDE", "LONGITUDE": "LATITUDE"})
    with pytest.raises(ValueError, match="swapped"):
        load_frames(None, ds, frames["cad"], None, **QUIET)


def test_projected_wkt_needs_its_crs(frames):
    cad = frames["cad"].collect()
    proj = shapely.transform(shapely.from_wkt(cad["geom_wkt"].to_numpy()),
                             lambda xy: np.column_stack([xy[:, 0] * 1e5, xy[:, 1] * 1e5]))
    bad = cad.with_columns(geom_wkt=pl.Series(shapely.to_wkt(proj)))
    with pytest.raises(ValueError, match="not lon/lat"):
        load_frames(None, frames["ds_asset"], bad, None, **QUIET)


def test_score_frames(frames):
    res = score_frames(ds_asset=frames["ds_asset"], cad=frames["cad"], bld=frames["bld"], **QUIET)
    assert res["addresses"].height == frames["ds_asset"].select(pl.len()).collect().item()
    assert (res["addresses"]["shell_class"] == "SHELL").sum() > 0

"""The datasets layer: locating labelled_asset / ds_asset / bld / cad by path + table and mapping columns."""

import numpy as np
import polars as pl
import pytest
import shapely
import yaml

from shell_identifier.assets import (check_datasets, datasets_config, load_addresses, load_inputs, load_label_table,
                                     load_parcels)
from shell_identifier.config import load_config
from shell_identifier.loaders import open_table
from shell_identifier.pipeline import prepare_context


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from shell_identifier.synthetic import generate

    out = tmp_path_factory.mktemp("assets")
    generate(out, n_parcels=150, seed=3)
    return out


def test_template_has_the_four_placeholders():
    cfg = datasets_config()
    for role in ("labelled_asset", "ds_asset", "bld", "cad"):
        assert cfg[role]["table"] == role
        assert cfg[role]["path"] is None


def test_paths_resolve_against_root_and_yaml_folder(tmp_path):
    (tmp_path / "conf").mkdir()
    y = tmp_path / "conf" / "datasets.yaml"
    y.write_text(yaml.safe_dump({"root": "../data", "cad": {"path": "cadastre.gpkg"}}), encoding="utf-8")
    cfg = datasets_config(y, {"bld": {"path": str(tmp_path / "elsewhere" / "b.parquet")}})
    data = (tmp_path / "conf" / ".." / "data")
    assert cfg["ds_asset"]["path"] == str(data)                      # unset -> root folder, table picks the file
    assert cfg["cad"]["path"] == str(data / "cadastre.gpkg")
    assert cfg["bld"]["path"] == str(tmp_path / "elsewhere" / "b.parquet")


def test_folder_plus_table_name_finds_the_file(synth):
    ts = open_table(synth / "assets", "DS_ASSET")        # case-insensitive stem match
    assert ts.kind == "parquet" and ts.find("address_detail_pid") == "ADDRESS_DETAIL_PID"
    with pytest.raises(FileNotFoundError, match="found"):
        open_table(synth / "assets", "no_such_table")


def test_geopackage_layer_by_table_name(synth):
    gpkg = synth / "Buildings" / "NSW_BUILDINGS.gpkg"
    assert open_table(gpkg, "building_address").layer == "building_address"
    with pytest.raises(FileNotFoundError):
        open_table(gpkg, "bld")                          # three layers, none called bld
    spec = datasets_config(None, {"bld": {"path": str(gpkg)}})["bld"]
    from shell_identifier.assets import load_building_layer

    assert len(load_building_layer(spec, log=lambda m: None)) > 0   # falls back to the 'buildings' layer


def test_renamed_columns_csv_with_wkt_geometry(synth, tmp_path):
    """A cadastre exported as CSV with its own column names and WKT geometry."""
    cad = pl.read_parquet(synth / "assets" / "cad.parquet")
    geom = shapely.from_wkb(cad["geometry"].ext.storage().to_numpy())
    custom = cad.select(PARCEL_KEY=pl.col("cadastre_pid"), PTYPE=pl.col("parcel_type"), IS_BASE=pl.col("base_parcel"),
                        STRATA_T=pl.col("strata_type"), shape_wkt=pl.Series(shapely.to_wkt(geom)))
    custom.write_csv(tmp_path / "my_cadastre.csv")
    spec = datasets_config(None, {"cad": {"path": str(tmp_path), "table": "my_cadastre", "crs": "EPSG:7844", "columns": {
        "parcel_pid": "PARCEL_KEY", "parcel_type": "PTYPE", "base_parcel": "IS_BASE", "strata_type": "STRATA_T",
        "geometry": "shape_wkt"}}})["cad"]
    base, strata = load_parcels(spec, log=lambda m: None)
    ref_base, ref_strata = load_parcels(datasets_config(None, {"cad": {"path": str(synth / "assets" / "cad.parquet")}})
                                        ["cad"], log=lambda m: None)
    assert len(base) == len(ref_base) and len(strata) == len(ref_strata)
    assert base.crs == "EPSG:7844"
    # strata counts are derived from the strata polygons when the attributes are absent
    assert base.df["total_strata_count"].drop_nulls().len() > 0


def test_label_values_and_numeric_labels(tmp_path):
    pl.DataFrame({"gnaf_id": ["A", "B", "C", "D", "E"],
                  "status": ["Private Dwelling", "Non-private", "1", "UNKNOWN", None]}).write_parquet(
        tmp_path / "labels.parquet")
    spec = datasets_config(None, {"labelled_asset": {
        "path": str(tmp_path / "labels.parquet"), "columns": {"address_pid": "gnaf_id", "label": "status"},
        "pd_values": ["private dwelling", "1"], "oos_values": ["non-private"]}})["labelled_asset"]
    lab = load_label_table(spec, log=lambda m: None)
    assert dict(zip(lab["address_pid"], lab["y"])) == {"A": 1, "B": 0, "C": 1, "D": -1, "E": -1}
    pl.DataFrame({"address_detail_pid": ["A", "B"], "is_pd": [True, False]}).write_parquet(tmp_path / "flags.parquet")
    spec = datasets_config(None, {"labelled_asset": {"path": str(tmp_path / "flags.parquet"),
                                                     "columns": {"label": "is_pd"}}})["labelled_asset"]
    assert load_label_table(spec, log=lambda m: None)["y"].to_list() == [1, 0]


def test_ds_asset_view_matches_the_gnaf_release(synth):
    """The flattened address view (names, upper case, retired rows) gives the same features as the PSV release."""
    cfg = load_config()
    quiet = lambda m: None  # noqa: E731
    via_assets = load_inputs(datasets_config(synth / "assets" / "datasets.yaml"), cfg, log=quiet)
    via_gnaf = load_inputs(datasets_config(synth / "assets" / "datasets.yaml",
                                           {"ds_asset": {"path": str(synth / "G-NAF")}}), cfg, log=quiet)
    assert len(via_assets.addresses) == len(via_gnaf.addresses)
    a = prepare_context(*via_assets.pipeline_args(), cfg)["scored"].sort("address_pid")
    g = prepare_context(*via_gnaf.pipeline_args(), cfg)["scored"].sort("address_pid")
    cols = [k for k in a.columns if k.startswith("f_")] + ["shell_probability"]
    assert np.allclose(a.select(cols).to_numpy(), g.select(cols).to_numpy())
    assert (a["node_type"] == g["node_type"]).all() and (a["parent_pid"] == g["parent_pid"]).all()
    assert (a["geocode_type"] == g["geocode_type"]).all()       # names normalised back to codes


def test_custom_table_names_from_the_command_line(synth, tmp_path):
    from shell_identifier.cli import main

    for src, dst in (("ds_asset", "address_2026"), ("cad", "cadastre_aug"), ("bld", "buildings_v3"),
                     ("labelled_asset", "census_pd")):
        (tmp_path / f"{dst}.parquet").write_bytes((synth / "assets" / f"{src}.parquet").read_bytes())
    out = tmp_path / "run"
    assert main(["run", "--ds-asset", str(tmp_path), "--cad", str(tmp_path), "--bld", str(tmp_path),
                 "--table", "ds_asset=address_2026", "--table", "cad=cadastre_aug", "--table", "bld=buildings_v3",
                 "--no-gpkg", "--out", str(out)]) == 0
    scores = pl.read_csv(out / "address_scores.csv")
    assert scores.height == pl.read_parquet(synth / "assets" / "ds_asset.parquet").filter(
        pl.col("DATE_RETIRED").is_null()).height


def test_datasets_check_and_template_commands(synth, tmp_path, capsys):
    from shell_identifier.cli import main

    assert main(["datasets", "template", "--out", str(tmp_path / "d.yaml")]) == 0
    assert yaml.safe_load((tmp_path / "d.yaml").read_text(encoding="utf-8"))["ds_asset"]["table"] == "ds_asset"
    report = check_datasets(datasets_config(synth / "assets" / "datasets.yaml"))
    assert "address_pid              <- ADDRESS_DETAIL_PID" in report
    assert "parcel_pid               <- cadastre_pid" in report
    assert "bld_address (building-address links): no path set" in report


def test_counts_table_with_unrecognised_names_uses_first_two_columns(synth, tmp_path):
    base, _ = load_parcels(datasets_config(None, {"cad": {"path": str(synth / "assets" / "cad.parquet")}})["cad"],
                           log=lambda m: None)
    pids = base.df["parcel_pid"].head(3).to_list()
    pl.DataFrame({"lot": pids, "how_many": [1, 2, 3]}).write_csv(tmp_path / "k.csv")
    ds = datasets_config(synth / "assets" / "datasets.yaml", {"parcel_counts": {"path": str(tmp_path / "k.csv")}})
    inputs = load_inputs(ds, load_config(), log=lambda m: None)
    assert inputs.counts["n_dwellings"].to_list() == [1, 2, 3]


def test_addresses_from_point_geometry_and_single_line_text(synth, tmp_path):
    view = pl.read_parquet(synth / "assets" / "ds_asset.parquet").filter(pl.col("DATE_RETIRED").is_null())
    pts = shapely.points(view["LONGITUDE"].to_numpy(), view["LATITUDE"].to_numpy())
    view.drop("LONGITUDE", "LATITUDE").with_columns(geom=pl.Series(shapely.to_wkb(pts))).write_parquet(
        tmp_path / "pts.parquet")
    layer = load_addresses(datasets_config(None, {"ds_asset": {"path": str(tmp_path / "pts.parquet"),
                                                               "columns": {"geometry": "geom"}}})["ds_asset"],
                           log=lambda m: None)
    assert np.allclose(layer.df["x"].to_numpy(), view["LONGITUDE"].to_numpy())
    text = pl.read_csv(synth / "text" / "addresses_free_text.csv")
    text.rename({"address": "FULL_ADDRESS"}).write_parquet(tmp_path / "text.parquet")
    layer = load_addresses(datasets_config(None, {"ds_asset": {"path": str(tmp_path / "text.parquet")}})["ds_asset"],
                           log=lambda m: None)
    assert layer.df.height == text.height and (layer.df["street_name"] != "").mean() > 0.95

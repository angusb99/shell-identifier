"""End-to-end: generate synthetic products, load them with the real loaders, score, evaluate, train."""

import pandas as pd
import pytest

from shell_identifier.cli import main
from shell_identifier.evaluate import evaluate, join_truth
from shell_identifier.loaders import (load_building_address, load_building_cadastre, load_buildings, load_cadastre,
                                      load_gnaf)
from shell_identifier.synthetic import generate


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    summary = generate(out, n_parcels=400, seed=7)
    return out, summary


def test_loaders_read_synthetic_products(synth):
    out, summary = synth
    gnaf = load_gnaf(out / "G-NAF")
    assert len(gnaf) == summary["addresses"]          # retired records are dropped
    assert gnaf.crs == "EPSG:7844"
    assert {"address_pid", "street_name", "flat_type_code", "geocode_type", "n_ps_secondaries", "x", "y"}         <= set(gnaf.df.columns)
    assert gnaf.meta["localities"] and gnaf.meta["localities"] <= {"TESTWOOD", "SAMPLE HILL", "MOCKTON PARK"}
    parcels, strata = load_cadastre(out / "Cadastre" / "NSW_CADASTRE.gpkg")
    assert (parcels.df["base_parcel"] == "Yes").all()
    assert len(strata) > 0 and parcels.geom is not None and len(parcels.geom) == len(parcels)
    b = load_buildings(out / "Buildings" / "NSW_BUILDINGS.gpkg")
    assert b.df["est_levels"].min() >= 1
    links = load_building_address(out / "Buildings" / "NSW_BUILDINGS.gpkg")
    assert links.height > 0 and links["address_pid"].str.starts_with("GA").all()   # G-NAF ids, not Geoscape's
    assert load_building_cadastre(out / "Buildings" / "NSW_BUILDINGS.gpkg").height > 0


@pytest.mark.parametrize("mode", ["gnaf", "text"])
def test_cli_run_and_evaluate(synth, tmp_path, mode):
    out, _ = synth
    src = ["--gnaf", str(out / "G-NAF")] if mode == "gnaf" else \
        ["--text", str(out / "text" / "addresses_free_text.csv"), "--id-col", "address_pid"]
    res = tmp_path / "res"
    assert main(["run", *src, "--cadastre", str(out / "Cadastre" / "NSW_CADASTRE.gpkg"),
                 "--buildings", str(out / "Buildings" / "NSW_BUILDINGS.gpkg"), "--out", str(res)]) == 0
    scores = pd.read_csv(res / "address_scores.csv")
    feats = pd.read_csv(res / "address_features.csv").drop(columns=["parcel_pid", "shell_probability"])
    merged = join_truth(scores.merge(feats, on="address_pid"), pd.read_csv(out / "truth" / "shell_truth.csv"))
    m = evaluate(merged)
    assert m["auc"] > 0.98
    assert m["contested"]["at_0.5"]["f1"] > 0.95
    assert (res / "parcel_summary.csv").exists() and (res / "address_scores.gpkg").exists()

    weights = tmp_path / "w.json"
    assert main(["train", "--results", str(res), "--labels", str(out / "truth" / "shell_truth.csv"),
                 "--out", str(weights)]) == 0
    assert weights.exists()

"""Count-constrained dwelling model: exact inference, roles and an end-to-end synthetic run."""

import itertools

import numpy as np
import polars as pl
import pytest
from scipy.special import comb

from conftest import addr
from shell_identifier.dwellings.countem import conditional_marginals, swap_map, swap_posterior
from shell_identifier.dwellings.roles import DWELLING, OOS, PRIMARY, SUB, assign_roles

PRIOR = np.log(np.array([0.7, 0.2, 0.07, 0.03]))


def _brute(logits, labelled, k):
    p = 1 / (1 + np.exp(-logits))
    n, L = len(logits), labelled.sum()
    total, marg, best = 0.0, np.zeros(n), (0.0, None)
    for z in itertools.product([0, 1], repeat=n):
        z = np.array(z, bool)
        if z.sum() != k:
            continue
        u, v = int((labelled & ~z).sum()), int((~labelled & z).sum())
        if min(u, v) >= len(PRIOR):
            continue
        w = np.prod(np.where(z, p, 1 - p)) * np.exp(PRIOR[min(u, v)]) / (comb(L, u) * comb(n - L, v))
        total += w
        marg += w * z
        if w > best[0]:
            best = (w, z)
    return marg / total, best[1]


@pytest.mark.parametrize("seed", range(25))
def test_swap_posterior_and_map_match_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 8))
    k = int(rng.integers(1, n))
    labelled = np.zeros(n, bool)
    labelled[rng.choice(n, k, replace=False)] = True
    logits = rng.normal(0, 2, n)
    marg, best = _brute(logits, labelled, k)
    q, post_m = swap_posterior(logits, labelled, k, PRIOR)
    assert np.allclose(q, marg, atol=1e-10)
    assert post_m.sum() == pytest.approx(1.0)
    assert (swap_map(logits, labelled, k, PRIOR) == best).all()


def test_conditional_marginals_sum_to_count():
    q = conditional_marginals(np.array([2.0, -1.0, 0.5, 3.0, -2.0]), 3)
    assert q.sum() == pytest.approx(3.0)


def test_label_on_shell_moves_to_unit():
    # 100 TEST ST (shell-like) labelled PD with UNIT 1; UNIT 2 unlabelled. Two dwellings on the parcel.
    logits = np.array([-3.0, 3.0, 3.0])
    labelled = np.array([True, True, False])
    z = swap_map(logits, labelled, 2, PRIOR)
    assert z.tolist() == [False, True, True]
    q, _ = swap_posterior(logits, labelled, 2, PRIOR)
    assert q[0] < 0.2 and q[2] > 0.8


def test_weak_evidence_keeps_the_label():
    logits = np.array([-0.3, 0.2, 0.3])
    labelled = np.array([True, True, False])
    assert swap_map(logits, labelled, 2, PRIOR).tolist() == [True, True, False]


def _roles(case, dwellings):
    case.run()
    s = case.result
    out = assign_roles(s, s["address_pid"].is_in(dwellings).to_numpy())
    return {r["address_pid"]: r for r in out.iter_rows(named=True)}


def test_roles_primary_and_sub_shells(case):
    case.parcel("P1", 0, 0, 90, 45).building("TA", 5, 8, 27, 30, levels=12)
    case.add(addr("PRECINCT", 45, 22, num=1, last=15),
             addr("TOWER_A", 16, 19, num=1, last=15, building="TOWER A"),
             addr("SHOP", 45, 3, num=1, last=15, flat_type="SHOP", flat=1))
    units = [f"U{u}" for u in (101, 102, 201)]
    case.add(*[addr(u, 16, 19, num=1, last=15, building="TOWER A", flat_type="UNIT", flat=int(u[1:])) for u in units])
    r = _roles(case, units)
    assert r["PRECINCT"]["role"] == PRIMARY
    assert r["TOWER_A"]["role"] == SUB
    assert r["SHOP"]["role"] == OOS
    assert all(r[u]["role"] == DWELLING for u in units)
    assert r["U101"]["primary_shell_pid"] == "PRECINCT"
    assert r["PRECINCT"]["dwellings_below"] == 3


def test_suffix_base_is_shell_but_shop_beside_one_flat_is_not(case):
    case.parcel("P1", 0, 0, 20, 45).building("BA", 1.5, 7, 9.25, 26).building("BB", 10.75, 7, 18.5, 26)
    case.add(addr("BASE", 10, 22.5, num=100), addr("A", 5, 15, num=100, sfx="A", geocode="BC"),
             addr("B", 15, 15, num=100, sfx="B", geocode="BC"))
    case.parcel("P2", 30, 0, 42, 45).building("SHOPB", 30.5, 0.5, 41.5, 20, levels=2)
    case.add(addr("SHOP", 36, 6, num=70, geocode="FCS"), addr("FLAT", 36, 6, num=70, sfx="A", geocode="FCS"))
    r = _roles(case, ["A", "B", "FLAT"])
    assert r["BASE"]["role"] == PRIMARY
    assert r["SHOP"]["role"] == OOS


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from shell_identifier.synthetic import generate

    out = tmp_path_factory.mktemp("dw")
    generate(out, n_parcels=500, seed=5)
    return out


def _check_verify(res, synth):
    from shell_identifier.dwellings.evaluate import evaluate_dwellings

    assign = pl.read_csv(res / "dwelling_assignments.csv", infer_schema=False)
    metrics, _ = evaluate_dwellings(assign, pl.read_csv(synth / "truth" / "shell_truth.csv", infer_schema=False))
    assert metrics["wrong_labels"] > 0
    assert metrics["model_accuracy"] > metrics["label_accuracy"]
    assert metrics["wrong_labels_fixed"] >= 0.6 * metrics["wrong_labels"]
    assert metrics["correct_labels_broken"] <= 0.02 * metrics["addresses"]
    # every verified parcel keeps its labelled dwelling count
    parcels = pl.read_csv(res / "parcel_dwellings.csv")
    v = parcels.filter(pl.col("mode") == "verified")
    assert (v["assigned_dwellings"] == v["labelled_pd"]).all()
    assert (res / "dwelling_model.joblib").exists() and (res / "model_report.md").exists()
    return metrics


def test_verify_fixes_misplaced_labels(synth, tmp_path):
    from shell_identifier.cli import main

    inputs = ["--gnaf", str(synth / "G-NAF"), "--cadastre", str(synth / "Cadastre" / "NSW_CADASTRE.gpkg"),
              "--buildings", str(synth / "Buildings" / "NSW_BUILDINGS.gpkg")]
    res = tmp_path / "verify"
    assert main(["dwellings", "verify", *inputs, "--labels", str(synth / "truth" / "pd_labels.csv"),
                 "--folds", "3", "--iterations", "4", "--out", str(res)]) == 0
    _check_verify(res, synth)

    pred = tmp_path / "predict"
    assert main(["dwellings", "predict", *inputs, "--model", str(res / "dwelling_model.joblib"),
                 "--out", str(pred)]) == 0
    p = pl.read_csv(pred / "dwelling_assignments.csv")
    assert (p["mode"] == "predicted").all()


def test_verify_from_the_four_assets(synth, tmp_path):
    """labelled_asset / ds_asset / bld / cad only (no link tables), located by a datasets YAML."""
    from shell_identifier.cli import main

    res = tmp_path / "verify_assets"
    assert main(["dwellings", "verify", "--datasets", str(synth / "assets" / "datasets.yaml"),
                 "--folds", "3", "--iterations", "4", "--no-gpkg", "--out", str(res)]) == 0
    m = _check_verify(res, synth)
    assert m["shell_precision"] > 0.95 and m["shell_recall"] > 0.95
    assert (res / "dwelling_assignments.parquet").exists()

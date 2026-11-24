"""Verify labelled dwellings (and find shells), or predict dwellings on unlabelled data."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .countem import CountConstrainedEM, EMSettings, _logit, conditional_marginals
from .features import dwelling_features
from .labels import parcel_counts
from .roles import DWELLING, OOS, PRIMARY, SUB, assign_roles, pair_moves

ASSIGNMENT_COLUMNS = [
    "address_pid", "address_label", "parcel_pid", "mode", "input_label", "dwelling_prob", "posterior",
    "rank_in_parcel", "assigned_dwelling", "role", "label_change", "primary_shell_pid", "primary_shell_label",
    "dwellings_below", "shell_probability", "parent_pid", "evidence", "decision",
]


def _join(parts: list[pd.Series], sep: str = "; ") -> pd.Series:
    out = pd.Series("", index=parts[0].index, dtype=object)
    for p in parts:
        p = p.fillna("").astype(str)
        out = pd.Series(np.where((out != "") & (p != ""), out + sep + p, np.where(out == "", p, out)), index=out.index)
    return out


def _txt(cond, text) -> pd.Series:
    return pd.Series(np.where(cond, text, ""), index=cond.index, dtype=object)


def evidence(s: pd.DataFrame, F: pd.DataFrame) -> pd.Series:
    """Short plain-text summary of what the model saw for each address."""
    i = lambda x: x.round(0).astype("Int64").astype(str)  # noqa: E731
    kind = np.select(
        [s["node_type"] == "street", s["node_type"] == "level", s["node_type"] == "structure"],
        ["street address", "level address", "building address"],
        s["flat_type_code"].replace("", "unit") + " (" + s["node_type"] + ")",
    )
    kind = pd.Series(kind, index=s.index)
    stack = np.expm1(F["g_stack_size"])
    levels = F["b_levels"]
    bld = _txt(s["building_pid"].notna(), "building " + i(levels.fillna(1)) + " storey(s), "
               + i(F["b_addr_count"]) + " address(es)")
    out_b = _txt(s["building_pid"].isna() & (s["n_dwelling_buildings"] > 0), "geocode outside buildings")
    stk = _txt(stack > 1, "geocode shared by " + i(stack) + " addresses")
    kids = _txt(s["n_desc"] > 0, i(s["n_desc"]) + " nested beneath")
    rng = _txt(s["n_range_desc"] > 0, "range covers " + i(s["n_range_numbers"]) + " number(s)")
    sfx = _txt(s["n_suffix_variants"] > 0, "suffix siblings " + s["suffix_list"].astype(str))
    n_parcel = s["n_parcel_others"] + s["in_ctx"].astype(int)
    strata = _txt(s["total_strata_count"].fillna(0) > 0, ", strata " + i(s["total_strata_count"].fillna(0)))
    parcel = "parcel " + i(n_parcel) + " address(es), " + i(s["n_dwelling_buildings"]) + " dwelling-size building(s)" + strata
    return _join([kind, bld, out_b, stk, kids, rng, sfx, parcel])


def _prepare(ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.DataFrame(ctx["scored"].drop(columns="geometry")).reset_index(drop=True)
    F = dwelling_features(s, ctx["parcels_m"], ctx["buildings_m"])
    return s, F


def verify(ctx: dict, labels: pd.DataFrame, settings: EMSettings | None = None,
           counts: pd.DataFrame | None = None, log=print) -> dict:
    """Fit the count-constrained model on labelled parcels and assign dwellings and shells.

    Parcels with labels get exactly their labelled number of dwellings (cross-fitted
    posteriors); parcels without labels are predicted by the final model.
    """
    s, F = _prepare(ctx)
    lab = labels.set_index("address_pid")
    s["y"] = s["address_pid"].map(lab["y"]).fillna(-1).astype(int)
    s["input_label"] = s["address_pid"].map(lab["label_raw"]).fillna("")
    k = parcel_counts(s["parcel_pid"], s["y"], counts)
    train = (s["parcel_pid"].isin(k.index) & (s["parcel_pid"] != "")).to_numpy()
    if not train.any():
        raise ValueError("no labelled address could be matched to a parcel; check the id column")
    log(f"{train.sum():,} addresses on {len(k):,} labelled parcels; {int((s['y'] == 1).sum()):,} PD labels")

    em = CountConstrainedEM(settings).fit(F.to_numpy()[train], s.loc[train, "parcel_pid"].to_numpy(), k,
                                          s.loc[train, "y"].to_numpy(), log=log)
    s["dwelling_prob"] = np.nan
    s["posterior"] = np.nan
    s["assigned_dwelling"] = False
    s.loc[train, "dwelling_prob"] = em.oof_prob
    s.loc[train, "posterior"] = em.posterior
    s.loc[train, "assigned_dwelling"] = em.assigned
    if (~train).any():
        prob = em.predict_proba(F.to_numpy()[~train])
        s.loc[~train, "dwelling_prob"] = prob
        s.loc[~train, "posterior"] = prob
        s.loc[~train, "assigned_dwelling"] = prob >= 0.5
    s["mode"] = np.where(train, "verified", "predicted")
    s["k_parcel"] = s["parcel_pid"].map(k)
    result = _finish(s, F)
    result.update(em=em, features=F, bundle={
        "model": em.model, "features": list(F.columns),
        "misplacement_prior": np.exp(em.log_prior_m).round(4).tolist(),
        "history": em.history, "settings": em.s.__dict__,
    })
    return result


def predict(ctx: dict, bundle: dict, counts: pd.DataFrame | None = None) -> dict:
    """Apply a saved model. With ``counts`` (parcel_pid, n_dwellings) those parcels get exactly that many."""
    s, F = _prepare(ctx)
    X = F.reindex(columns=bundle["features"]).to_numpy()
    prob = bundle["model"].predict_proba(X)[:, 1]
    s["y"] = -1
    s["input_label"] = ""
    s["dwelling_prob"] = prob
    s["posterior"] = prob
    s["assigned_dwelling"] = prob >= 0.5
    s["mode"] = "predicted"
    s["k_parcel"] = np.nan
    if counts is not None and not counts.empty:
        k = counts.set_index(counts.columns[0])[counts.columns[1]].astype(int)
        logits = _logit(prob)
        for pid, idx in s[s["parcel_pid"].isin(k.index)].groupby("parcel_pid").indices.items():
            kk = int(k[pid])
            q = conditional_marginals(logits[idx], kk)
            sel = np.zeros(len(idx), dtype=bool)
            sel[np.argsort(-logits[idx], kind="stable")[:kk]] = True
            s.loc[idx, "posterior"] = q
            s.loc[idx, "assigned_dwelling"] = sel
            s.loc[idx, "mode"] = "counted"
            s.loc[idx, "k_parcel"] = kk
    return _finish(s, F)


def _finish(s: pd.DataFrame, F: pd.DataFrame) -> dict:
    s["assigned_dwelling"] = s["assigned_dwelling"].astype(bool)
    roles = assign_roles(s, s["assigned_dwelling"].to_numpy())
    s = s.join(roles.drop(columns="address_pid"))
    labels = s.set_index("address_pid")["address_label"]
    s["primary_shell_label"] = s["primary_shell_pid"].map(labels).fillna("")
    s["rank_in_parcel"] = s.groupby("parcel_pid")["posterior"].rank(ascending=False, method="first")
    s.loc[s["parcel_pid"] == "", "rank_in_parcel"] = np.nan
    labelled_pd = s["y"] == 1
    s["label_change"] = np.select(
        [labelled_pd & ~s["assigned_dwelling"], (s["y"] == 0) & s["assigned_dwelling"],
         (s["y"] == -1) & s["assigned_dwelling"] & (s["mode"] == "verified")],
        ["PD->OOS", "OOS->PD", "unlabelled->PD"], "")
    n_parcel = s.groupby("parcel_pid")["address_pid"].transform("size")
    rank = s["rank_in_parcel"].round(0).astype("Int64").astype(str)
    s["decision"] = np.where(
        s["mode"].isin(["verified", "counted"]),
        "parcel has " + s["k_parcel"].round(0).astype("Int64").astype(str) + " dwelling(s); ranked " + rank + "/"
        + n_parcel.astype(str) + ", posterior " + s["posterior"].round(2).astype(str),
        "dwelling probability " + s["dwelling_prob"].round(2).astype(str),
    )
    s["evidence"] = evidence(s, F).to_numpy()

    verified = s[s["mode"] == "verified"]
    moves = pair_moves(verified[["parcel_pid", "address_pid", "address_label", "y", "assigned_dwelling", "posterior"]]
                       .rename(columns={"assigned_dwelling": "assigned"})) if len(verified) else pd.DataFrame()
    return {"assignments": s, "parcels": summarise(s), "moves": moves}


def summarise(s: pd.DataFrame) -> pd.DataFrame:
    p = s[s["parcel_pid"] != ""].assign(
        _pd=lambda d: d["y"] == 1, _moved=lambda d: d["label_change"] == "PD->OOS",
        _primary=lambda d: d["role"] == PRIMARY, _sub=lambda d: d["role"] == SUB, _oos=lambda d: d["role"] == OOS)
    g = p.groupby("parcel_pid")
    assigned_post = p["posterior"].where(p["assigned_dwelling"])
    other_post = p["posterior"].where(~p["assigned_dwelling"])
    out = pd.DataFrame({
        "mode": g["mode"].first(),
        "n_addresses": g.size(),
        "labelled_pd": g["_pd"].sum(),
        "assigned_dwellings": g["assigned_dwelling"].sum(),
        "expected_dwellings": g["dwelling_prob"].sum().round(2),
        "n_label_changes": g["_moved"].sum(),
        "n_primary_shells": g["_primary"].sum(),
        "n_sub_shells": g["_sub"].sum(),
        "n_out_of_scope": g["_oos"].sum(),
        "margin": (assigned_post.groupby(p["parcel_pid"]).min().fillna(1.0)
                   - other_post.groupby(p["parcel_pid"]).max().fillna(0.0)).round(3),
    })
    prim = p[p["role"] == PRIMARY].groupby("parcel_pid")
    out["primary_shell_pids"] = prim["address_pid"].agg(";".join).reindex(out.index).fillna("")
    out["primary_shell_labels"] = prim["address_label"].agg(" | ".join).reindex(out.index).fillna("")
    out["status"] = np.select(
        [out["mode"] != "verified", out["n_label_changes"] > 0],
        ["PREDICTED", "LABELS_MOVED"], "AGREES_WITH_LABELS")
    out["review"] = out["margin"] < 0.3
    return out.reset_index()


def write_outputs(result: dict, out_dir: str | Path, geometry: gpd.GeoSeries | None = None,
                  write_features: bool = False) -> dict[str, Path]:
    import joblib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = result["assignments"]
    paths = {"assignments_csv": out_dir / "dwelling_assignments.csv", "parcels_csv": out_dir / "parcel_dwellings.csv"}
    table = s[ASSIGNMENT_COLUMNS].copy()
    for c in ("dwelling_prob", "posterior"):
        table[c] = table[c].round(4)
    table.to_csv(paths["assignments_csv"], index=False)
    result["parcels"].to_csv(paths["parcels_csv"], index=False)
    if len(result.get("moves", [])):
        paths["moves_csv"] = out_dir / "label_moves.csv"
        result["moves"].to_csv(paths["moves_csv"], index=False)
    if geometry is not None:
        paths["assignments_gpkg"] = out_dir / "dwelling_assignments.gpkg"
        gpd.GeoDataFrame(table, geometry=geometry.to_numpy(), crs=geometry.crs).to_file(
            paths["assignments_gpkg"], layer="dwelling_assignments", driver="GPKG", engine="pyogrio")
    if write_features:
        paths["features_csv"] = out_dir / "dwelling_features.csv"
        result["features"].assign(address_pid=s["address_pid"].to_numpy()).to_csv(paths["features_csv"], index=False)
    if "bundle" in result:
        paths["model"] = out_dir / "dwelling_model.joblib"
        joblib.dump(result["bundle"], paths["model"])
        paths["report"] = out_dir / "model_report.md"
        paths["report"].write_text(model_report(result), encoding="utf-8")
    return paths


def feature_importance(result: dict, max_rows: int = 8000, seed: int = 0) -> pd.Series:
    """Permutation importance (drop in log-likelihood) of the final model on the assigned targets."""
    from sklearn.inspection import permutation_importance

    s, F = result["assignments"], result["features"]
    rows = np.flatnonzero((s["mode"] == "verified").to_numpy())
    rng = np.random.default_rng(seed)
    if len(rows) > max_rows:
        rows = rng.choice(rows, max_rows, replace=False)
    y = s["assigned_dwelling"].to_numpy()[rows].astype(int)
    if len(np.unique(y)) < 2:
        return pd.Series(dtype=float)
    imp = permutation_importance(result["bundle"]["model"], F.to_numpy()[rows], y, scoring="neg_log_loss",
                                 n_repeats=2, random_state=seed)
    return pd.Series(imp.importances_mean, index=F.columns).sort_values(ascending=False)


def model_report(result: dict) -> str:
    s, parcels, em = result["assignments"], result["parcels"], result["em"]
    v = parcels[parcels["mode"] == "verified"]
    hist = pd.DataFrame(em.history)
    lines = [
        "# Dwelling model report", "",
        f"* Labelled parcels: {len(v):,} ({int((v['n_addresses'] > 1).sum()):,} with more than one address)",
        f"* PD labels: {int((s['y'] == 1).sum()):,}; out of scope: {int((s['y'] == 0).sum()):,}; "
        f"unlabelled on labelled parcels: {int(((s['y'] == -1) & (s['mode'] == 'verified')).sum()):,}",
        f"* PD labels moved to another address: **{int((s['label_change'] == 'PD->OOS').sum()):,}** "
        f"on {int((v['n_label_changes'] > 0).sum()):,} parcels",
        f"* Parcels flagged for review (margin < 0.3): {int(v['review'].sum()):,}",
        "* Estimated misplaced PD labels per parcel, P(m = 0, 1, 2, ...): "
        + ", ".join(f"{v:.3f}" for v in np.exp(em.log_prior_m)),
        f"* Roles: " + ", ".join(f"{r} {int((s['role'] == r).sum()):,}" for r in (DWELLING, PRIMARY, SUB, OOS)),
        "", "## EM iterations", "", "| " + " | ".join(hist.columns) + " |", "|" + "---|" * len(hist.columns),
    ]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in hist.itertuples(index=False)]
    imp = feature_importance(result)
    if len(imp):
        lines += ["", "## Most influential features (permutation importance)", "",
                  "| feature | importance |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in imp.head(25).items()]
    return "\n".join(lines) + "\n"

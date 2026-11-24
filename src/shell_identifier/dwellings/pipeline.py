"""Verify labelled dwellings (and find shells), or predict dwellings on unlabelled data (polars)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..frames import c, int_text, join_nonempty, lookup, markdown_table
from .countem import (
    CountConstrainedEM,
    EMSettings,
    _logit,
    as_matrix,
    conditional_marginals,
    usable_columns,
)
from .features import dwelling_features
from .labels import parcel_counts
from .roles import DWELLING, OOS, PRIMARY, SUB, assign_roles, pair_moves

ASSIGNMENT_COLUMNS = [
    "address_pid", "address_label", "parcel_pid", "mode", "input_label", "dwelling_prob", "posterior",
    "rank_in_parcel", "assigned_dwelling", "role", "label_change", "primary_shell_pid", "primary_shell_label",
    "dwellings_below", "shell_probability", "parent_pid", "evidence", "decision",
]


def _txt(cond: pl.Expr, text: pl.Expr) -> pl.Expr:
    return pl.when(cond).then(text).otherwise(pl.lit(""))


def evidence(s: pl.DataFrame, F: pl.DataFrame) -> pl.Series:
    """Short plain-text summary of what the model saw for each address."""
    df = s.with_columns(_levels=F["b_levels"], _baddr=F["b_addr_count"],
                        _stack=pl.len().over(["parcel_pid", "xy_key"]))
    ftype = pl.when(c("flat_type_code") == "").then(pl.lit("unit")).otherwise(c("flat_type_code"))
    kind = (pl.when(c("node_type") == "street").then(pl.lit("street address"))
            .when(c("node_type") == "level").then(pl.lit("level address"))
            .when(c("node_type") == "structure").then(pl.lit("building address"))
            .otherwise(ftype + " (" + c("node_type") + ")"))
    strata = _txt(c("total_strata_count").fill_null(0) > 0, ", strata " + int_text(c("total_strata_count")))
    n_parcel = c("n_parcel_others") + c("in_ctx").cast(pl.Int64)
    expr = join_nonempty([
        kind,
        _txt(c("building_pid").is_not_null(), "building " + int_text(c("_levels").fill_null(1)) + " storey(s), "
             + int_text(c("_baddr")) + " address(es)"),
        _txt(c("building_pid").is_null() & (c("n_dwelling_buildings") > 0), pl.lit("geocode outside buildings")),
        _txt(c("_stack") > 1, "geocode shared by " + int_text(c("_stack")) + " addresses"),
        _txt(c("n_desc") > 0, int_text(c("n_desc")) + " nested beneath"),
        _txt(c("n_range_desc") > 0, "range covers " + int_text(c("n_range_numbers")) + " number(s)"),
        _txt(c("n_suffix_variants") > 0, "suffix siblings " + c("suffix_list")),
        "parcel " + int_text(n_parcel) + " address(es), " + int_text(c("n_dwelling_buildings"))
        + " dwelling-size building(s)" + strata,
    ], "; ")
    return df.select(expr.alias("evidence")).to_series()


def _prepare(ctx: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    s = ctx["scored"]
    return s, dwelling_features(s, ctx["parcels_m"], ctx["buildings_m"])


def _with_labels(s: pl.DataFrame, labels: pl.DataFrame | None) -> pl.DataFrame:
    """``y`` = 1 for addresses labelled PD and 0 for every other address (-1 throughout without labels)."""
    s = s.drop([k for k in ("y", "input_label") if k in s.columns])
    if labels is None:
        return s.with_columns(y=pl.lit(-1, dtype=pl.Int64), input_label=pl.lit(""))
    is_pd = c("address_pid").is_in(labels.filter(c("y") == 1)["address_pid"].cast(pl.Utf8).implode())
    return s.with_columns(y=is_pd.cast(pl.Int64), input_label=pl.when(is_pd).then(pl.lit("PD")).otherwise(pl.lit("OOS")))


def verify(ctx: dict, labels: pl.DataFrame, settings: EMSettings | None = None,
           counts: pl.DataFrame | None = None, log=print) -> dict:
    """Fit the count-constrained model and assign dwellings and shells.

    ``labels`` are the PD addresses; every other address is taken as not PD, so a
    parcel's dwelling count is its number of PD labels (or ``counts`` where given).
    Each parcel gets exactly that many dwellings (cross-fitted posteriors); addresses
    that matched no parcel are predicted by the final model.
    """
    s, F = _prepare(ctx)
    s = _with_labels(s, labels)
    n_pd = int((s["y"] == 1).sum())
    if not n_pd:
        raise ValueError("no PD label matched an address; check that the label ids are the ds_asset address ids")
    if labels.height > n_pd:
        log(f"{labels.height - n_pd:,} PD labels have an address id that is not in ds_asset and are ignored")
    k = parcel_counts(s, counts)
    k_map = dict(zip(k["parcel_pid"].to_list(), k["k"].to_list()))
    train = (s["parcel_pid"].is_in(k["parcel_pid"].implode()) & (s["parcel_pid"] != "")).to_numpy()
    if not train.any():
        raise ValueError("no address could be matched to a parcel; check the coordinates and the cadastre")
    n_pd_parcel = int((s["y"] == 1).filter(pl.Series(train)).sum())
    log(f"{int(train.sum()):,} addresses on {k.height:,} parcels ({int((k['k'] > 0).sum()):,} with a PD label); "
        f"{n_pd:,} PD labels, {n_pd - n_pd_parcel:,} of them on addresses outside every parcel; "
        f"all other addresses are taken as not PD")

    # Features with no value (or too few values) on the parcels, e.g. storeys without floor area or
    # roof height, carry no information and break histogram binning.
    X = as_matrix(F)
    ok = usable_columns(X[train])
    if not ok.all():
        dropped = [name for name, keep in zip(F.columns, ok) if not keep]
        log(f"{len(dropped)} features have no data (or too little to use) and are left out: {', '.join(dropped)}")
    F_all, F, X = F, F.select([name for name, keep in zip(F.columns, ok) if keep]), X[:, ok]
    parcel = s["parcel_pid"].to_numpy()
    y = s["y"].to_numpy()
    em = CountConstrainedEM(settings).fit(X[train], parcel[train], k_map, y[train], log=log)
    prob, post, assigned = np.full(s.height, np.nan), np.full(s.height, np.nan), np.zeros(s.height, dtype=bool)
    prob[train], post[train], assigned[train] = em.oof_prob, em.posterior, em.assigned
    if (~train).any():
        p = em.predict_proba(X[~train])
        prob[~train], post[~train], assigned[~train] = p, p, p >= 0.5
    s = s.with_columns(
        dwelling_prob=pl.Series(prob), posterior=pl.Series(post), assigned_dwelling=pl.Series(assigned),
        mode=pl.Series(np.where(train, "verified", "predicted")),
        k_parcel=lookup(c("parcel_pid"), k["parcel_pid"], k["k"].cast(pl.Float64)),
    )
    result = _finish(s, F_all)
    result.update(em=em, features=F, crs=ctx.get("crs"), bundle={
        "model": em.model, "features": F.columns,
        "misplacement_prior": np.exp(em.log_prior_m).round(4).tolist(),
        "history": em.history, "settings": em.s.__dict__,
    })
    return result


def predict(ctx: dict, bundle: dict, counts: pl.DataFrame | None = None) -> dict:
    """Apply a saved model. With ``counts`` (parcel_pid, n_dwellings) those parcels get exactly that many."""
    s, F = _prepare(ctx)
    X = as_matrix(F.select([c(k) if k in F.columns else pl.lit(None, dtype=pl.Float64).alias(k)
                            for k in bundle["features"]]))
    prob = bundle["model"].predict_proba(X)[:, 1]
    post, assigned = prob.copy(), prob >= 0.5
    mode = np.full(s.height, "predicted", dtype=object)
    k_parcel = np.full(s.height, np.nan)
    if counts is not None and counts.height:
        kc = counts.select(c(counts.columns[0]).cast(pl.Utf8).alias("parcel_pid"),
                           c(counts.columns[1]).cast(pl.Int64).alias("k"))
        k_map = dict(zip(kc["parcel_pid"].to_list(), kc["k"].to_list()))
        logits = _logit(prob)
        groups = (s.select("parcel_pid").with_row_index("_i").filter(c("parcel_pid").is_in(kc["parcel_pid"].implode()))
                  .group_by("parcel_pid").agg("_i"))
        for pid, idx in groups.iter_rows():
            idx = np.asarray(idx, dtype=np.int64)
            kk = int(k_map[pid])
            post[idx] = conditional_marginals(logits[idx], kk)
            sel = np.zeros(len(idx), dtype=bool)
            sel[np.argsort(-logits[idx], kind="stable")[:kk]] = True
            assigned[idx] = sel
            mode[idx] = "counted"
            k_parcel[idx] = kk
    s = _with_labels(s, None).with_columns(
        dwelling_prob=pl.Series(prob), posterior=pl.Series(post), assigned_dwelling=pl.Series(assigned),
        mode=pl.Series(mode.astype(str)), k_parcel=pl.Series(k_parcel).fill_nan(None))
    result = _finish(s, F)
    result.update(features=F, crs=ctx.get("crs"))
    return result


def _finish(s: pl.DataFrame, F: pl.DataFrame) -> dict:
    s = s.with_columns(c("assigned_dwelling").cast(pl.Boolean))
    roles = assign_roles(s, s["assigned_dwelling"].to_numpy())
    s = s.drop([k for k in roles.columns if k != "address_pid" and k in s.columns]).hstack(roles.drop("address_pid"))
    assigned, y = c("assigned_dwelling"), c("y")
    rank = pl.when(c("parcel_pid") != "").then(c("posterior").rank("ordinal", descending=True).over("parcel_pid"))
    s = s.with_columns(
        primary_shell_label=lookup(c("primary_shell_pid"), s["address_pid"], s["address_label"], default=""),
        rank_in_parcel=rank.cast(pl.Int64),
        label_change=pl.when((y == 1) & ~assigned).then(pl.lit("PD->OOS"))
        .when((y == 0) & assigned).then(pl.lit("OOS->PD"))
        .otherwise(pl.lit("")),
    )
    n_parcel = pl.len().over("parcel_pid").cast(pl.Utf8)
    s = s.with_columns(
        decision=pl.when(c("mode").is_in(["verified", "counted"]))
        .then("parcel has " + int_text(c("k_parcel")) + " dwelling(s); ranked " + int_text(c("rank_in_parcel"))
              + "/" + n_parcel + ", posterior " + c("posterior").round(2).cast(pl.Utf8))
        .otherwise("dwelling probability " + c("dwelling_prob").round(2).cast(pl.Utf8)),
    )
    s = s.with_columns(evidence(s, F))
    verified = s.filter(c("mode") == "verified")
    moves = (pair_moves(verified.select("parcel_pid", "address_pid", "address_label", "y", "posterior",
                                        assigned=c("assigned_dwelling")))
             if verified.height else pl.DataFrame())
    return {"assignments": s, "parcels": summarise(s), "moves": moves}


def summarise(s: pl.DataFrame) -> pl.DataFrame:
    p = s.filter(c("parcel_pid") != "")
    assigned, role = c("assigned_dwelling"), c("role")
    out = p.group_by("parcel_pid").agg(
        c("mode").first(),
        pl.len().alias("n_addresses"),
        (c("y") == 1).sum().alias("labelled_pd"),
        assigned.sum().alias("assigned_dwellings"),
        c("dwelling_prob").sum().round(2).alias("expected_dwellings"),
        (c("label_change") == "PD->OOS").sum().alias("n_label_changes"),
        (role == PRIMARY).sum().alias("n_primary_shells"),
        (role == SUB).sum().alias("n_sub_shells"),
        (role == OOS).sum().alias("n_out_of_scope"),
        c("posterior").filter(assigned).min().alias("_lo"),
        c("posterior").filter(~assigned).max().alias("_hi"),
        c("address_pid").filter(role == PRIMARY).str.join(";").alias("primary_shell_pids"),
        c("address_label").filter(role == PRIMARY).str.join(" | ").alias("primary_shell_labels"),
    ).sort("parcel_pid")
    out = out.with_columns(margin=(c("_lo").fill_null(1.0) - c("_hi").fill_null(0.0)).round(3)).drop("_lo", "_hi")
    order = ["parcel_pid", "mode", "n_addresses", "labelled_pd", "assigned_dwellings", "expected_dwellings",
             "n_label_changes", "n_primary_shells", "n_sub_shells", "n_out_of_scope", "margin",
             "primary_shell_pids", "primary_shell_labels"]
    return out.select(order).with_columns(
        status=pl.when(c("mode") != "verified").then(pl.lit("PREDICTED"))
        .when(c("n_label_changes") > 0).then(pl.lit("LABELS_MOVED")).otherwise(pl.lit("AGREES_WITH_LABELS")),
        review=c("margin") < 0.3,
    )


def write_outputs(result: dict, out_dir: str | Path, gpkg: bool = True, write_features: bool = False,
                  parquet: bool = True) -> dict[str, Path]:
    """CSV (and parquet) tables, an optional point GeoPackage, the model bundle and a report."""
    import joblib

    from ..pipeline import write_geopackage

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = result["assignments"]
    paths = {"assignments_csv": out_dir / "dwelling_assignments.csv", "parcels_csv": out_dir / "parcel_dwellings.csv"}
    table = s.select(ASSIGNMENT_COLUMNS).with_columns(c("dwelling_prob").round(4), c("posterior").round(4))
    table.write_csv(paths["assignments_csv"])
    if parquet:
        paths["assignments_parquet"] = out_dir / "dwelling_assignments.parquet"
        table.write_parquet(paths["assignments_parquet"])
    result["parcels"].write_csv(paths["parcels_csv"])
    if result.get("moves") is not None and result["moves"].height:
        paths["moves_csv"] = out_dir / "label_moves.csv"
        result["moves"].write_csv(paths["moves_csv"])
    if gpkg and {"x", "y"} <= set(s.columns):
        paths["assignments_gpkg"] = out_dir / "dwelling_assignments.gpkg"
        write_geopackage(table.with_columns(s.select("x", "y")), paths["assignments_gpkg"], "dwelling_assignments",
                         result.get("crs"))
    if write_features:
        feats = result["features"].with_columns(address_pid=s["address_pid"]).select(
            ["address_pid"] + result["features"].columns)
        paths["features_csv"] = out_dir / "dwelling_features.csv"
        feats.write_csv(paths["features_csv"])
        if parquet:
            paths["features_parquet"] = out_dir / "dwelling_features.parquet"
            feats.write_parquet(paths["features_parquet"])
    if "bundle" in result:
        paths["model"] = out_dir / "dwelling_model.joblib"
        joblib.dump(result["bundle"], paths["model"])
        paths["report"] = out_dir / "model_report.md"
        paths["report"].write_text(model_report(result), encoding="utf-8")
    return paths


def feature_importance(result: dict, max_rows: int = 8000, seed: int = 0) -> pl.DataFrame:
    """Permutation importance (drop in log-likelihood) of the final model on the assigned targets."""
    from sklearn.inspection import permutation_importance

    s, F = result["assignments"], result["features"]
    rows = np.flatnonzero((s["mode"] == "verified").to_numpy())
    rng = np.random.default_rng(seed)
    if len(rows) > max_rows:
        rows = rng.choice(rows, max_rows, replace=False)
    y = s["assigned_dwelling"].to_numpy()[rows].astype(int)
    if len(np.unique(y)) < 2:
        return pl.DataFrame({"feature": [], "importance": []}, schema={"feature": pl.Utf8, "importance": pl.Float64})
    imp = permutation_importance(result["bundle"]["model"], as_matrix(F)[rows], y, scoring="neg_log_loss",
                                 n_repeats=2, random_state=seed)
    return pl.DataFrame({"feature": F.columns, "importance": imp.importances_mean}).sort("importance", descending=True)


def model_report(result: dict) -> str:
    s, parcels, em = result["assignments"], result["parcels"], result["em"]
    v = parcels.filter(c("mode") == "verified")
    hist = pl.DataFrame(em.history)
    role_counts = ", ".join(f"{r} {int((s['role'] == r).sum()):,}" for r in (DWELLING, PRIMARY, SUB, OOS))
    lines = [
        "# Dwelling model report", "",
        (f"* Parcels checked against their PD count: {v.height:,} "
         f"({int((v['n_addresses'] > 1).sum()):,} with more than one address)"),
        f"* PD labels: {int((s['y'] == 1).sum()):,}; other addresses (taken as not PD): "
        f"{int((s['y'] == 0).sum()):,}",
        f"* PD labels moved to another address: **{int((s['label_change'] == 'PD->OOS').sum()):,}** "
        f"on {int((v['n_label_changes'] > 0).sum()):,} parcels",
        f"* Parcels flagged for review (margin < 0.3): {int(v['review'].sum()):,}",
        "* Estimated misplaced PD labels per parcel, P(m = 0, 1, 2, ...): "
        + ", ".join(f"{x:.3f}" for x in np.exp(em.log_prior_m)),
        f"* Roles: {role_counts}",
        "", "## EM iterations", "", markdown_table(hist),
    ]
    imp = feature_importance(result)
    if imp.height:
        lines += ["", "## Most influential features (permutation importance)", "",
                  markdown_table(imp.head(25).with_columns(c("importance").round(4)))]
    return "\n".join(lines) + "\n"

"""Command line interface: ``shell-identifier <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def _counts(series) -> str:
    return "\n".join(f"{k}: {v:,}" for k, v in series.value_counts(sort=True).iter_rows())


# ---------------------------------------------------------------------------------------------
# datasets: YAML + command-line overrides
# ---------------------------------------------------------------------------------------------

def _set(over: dict, role: str, key: str, value) -> None:
    if value is None:
        return
    spec = over.setdefault(role, {})
    if key.startswith("columns."):
        spec.setdefault("columns", {})[key.split(".", 1)[1]] = value
    else:
        spec[key] = value


def _datasets(args) -> dict:
    """The datasets YAML (or the template) with paths and names from the command line on top."""
    from .assets import ROLES, datasets_config

    over: dict = {}
    g = lambda name: getattr(args, name, None)  # noqa: E731
    _set(over, "ds_asset", "path", g("ds_asset"))
    if g("gnaf"):
        _set(over, "ds_asset", "path", g("gnaf"))
        _set(over, "ds_asset", "format", "gnaf")
    if g("text"):
        _set(over, "ds_asset", "path", g("text"))
        _set(over, "ds_asset", "format", "text")
        _set(over, "ds_asset", "columns.address_text", g("address_col") or "address")
    elif g("address_col"):
        _set(over, "ds_asset", "columns.address_text", g("address_col"))
    _set(over, "ds_asset", "columns.longitude", g("lon_col"))
    _set(over, "ds_asset", "columns.latitude", g("lat_col"))
    _set(over, "ds_asset", "columns.address_pid", g("id_col"))
    _set(over, "ds_asset", "crs", g("crs"))
    _set(over, "ds_asset", "states", g("states"))
    _set(over, "ds_asset", "localities", g("localities"))
    _set(over, "cad", "path", g("cad"))
    _set(over, "cad", "table", g("cadastre_layer"))
    _set(over, "bld", "path", g("bld"))
    _set(over, "bld", "table", g("buildings_layer"))
    _set(over, "bld", "height_table", g("building_height"))
    _set(over, "bld_address", "path", g("bld_address"))
    _set(over, "bld_cadastre", "path", g("bld_cadastre"))
    _set(over, "labelled_asset", "path", g("labelled_asset"))
    _set(over, "labelled_asset", "columns.address_pid", g("labels_id"))
    _set(over, "labelled_asset", "columns.label", g("label_col"))
    _set(over, "labelled_asset", "pd_values", g("pd_values"))
    _set(over, "parcel_counts", "path", g("counts"))
    for item in g("table") or []:
        role, _, name = item.partition("=")
        if role not in ROLES or not name:
            raise SystemExit(f"--table expects ROLE=NAME with ROLE one of {', '.join(ROLES)}; got {item!r}")
        _set(over, role, "table", name)
    return datasets_config(g("datasets"), over)


def _inputs(args, cfg, labels: bool = False):
    from .assets import load_inputs

    bbox = tuple(args.bbox) if getattr(args, "bbox", None) else None
    return load_inputs(_datasets(args), cfg, labels=labels, bbox=bbox, log=_log)


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------

def cmd_synth(args) -> int:
    from .synthetic import generate

    summary = generate(args.out, n_parcels=args.parcels, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_parse(args) -> int:
    from .parser import parse_address

    for text in args.text:
        res = parse_address(text)
        print(json.dumps({"input": text, **{k: v for k, v in res.items() if v}}, indent=2))
    return 0


def cmd_datasets_template(args) -> int:
    from .assets import template_text

    text = template_text()
    if args.out:
        out = Path(args.out)
        if out.exists() and not args.force:
            raise SystemExit(f"{out} exists; pass --force to overwrite")
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}; edit the paths, table names and columns, then run "
              f"`shell-identifier datasets check --datasets {out}`")
    else:
        print(text)
    return 0


def cmd_datasets_check(args) -> int:
    from .assets import check_datasets

    print(check_datasets(_datasets(args)))
    return 0


def cmd_run(args) -> int:
    from .config import load_config
    from .pipeline import run_pipeline, write_outputs

    cfg = load_config(args.config, args.weights)
    inputs = _inputs(args, cfg)
    t0 = time.time()
    result = run_pipeline(*inputs.pipeline_args(), cfg)
    paths = write_outputs(result, args.out)
    s = result["addresses"]
    _log(f"scored {s.height:,} addresses in {time.time() - t0:.1f}s")
    print(_counts(s["shell_class"]))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _settings(args):
    from .dwellings import EMSettings

    r, m = args.misplaced_rate, args.max_misplaced
    tail = [r * 0.25 ** (i - 1) for i in range(1, m + 1)]
    prior = tuple([1 - r] + [x * r / sum(tail) for x in tail])
    return EMSettings(folds=args.folds, max_iter=args.iterations, max_misplaced=m, misplacement_prior=prior)


def cmd_dwellings_verify(args) -> int:
    from .config import load_config
    from .dwellings import verify, write_outputs
    from .pipeline import prepare_context

    cfg = load_config(args.config, args.weights)
    t0 = time.time()
    inputs = _inputs(args, cfg, labels=True)
    ctx = prepare_context(*inputs.pipeline_args(), cfg)
    _log(f"context features for {ctx['scored'].height:,} addresses ({time.time() - t0:.1f}s)")
    result = verify(ctx, inputs.labels, _settings(args), counts=inputs.counts, log=_log)
    paths = write_outputs(result, args.out, gpkg=not args.no_gpkg, write_features=args.write_features)
    _log(f"done in {time.time() - t0:.1f}s")
    print(_counts(result["assignments"]["role"]))
    print(_counts(result["parcels"]["status"]))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def cmd_dwellings_predict(args) -> int:
    import joblib

    from .config import load_config
    from .dwellings import predict, write_outputs
    from .pipeline import prepare_context

    cfg = load_config(args.config, args.weights)
    bundle = joblib.load(args.model)
    inputs = _inputs(args, cfg)
    ctx = prepare_context(*inputs.pipeline_args(), cfg)
    result = predict(ctx, bundle, counts=inputs.counts)
    paths = write_outputs(result, args.out, gpkg=not args.no_gpkg)
    print(_counts(result["assignments"]["role"]))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def cmd_dwellings_evaluate(args) -> int:
    import polars as pl

    from .dwellings.evaluate import evaluate_dwellings

    assign = pl.read_csv(Path(args.results) / "dwelling_assignments.csv", infer_schema=False)
    _, text = evaluate_dwellings(assign, pl.read_csv(args.truth, infer_schema=False), truth_id=args.truth_id)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _load_results(results: str):
    import pandas as pd

    d = Path(results)
    scores = pd.read_csv(d / "address_scores.csv", dtype={"parcel_pid": str})
    feats = pd.read_csv(d / "address_features.csv", dtype={"parcel_pid": str})
    feats = feats.drop(columns=[k for k in ("parcel_pid", "shell_probability") if k in feats])
    return scores.merge(feats, on="address_pid", how="left")


def cmd_evaluate(args) -> int:
    import pandas as pd

    from .evaluate import breakdown, evaluate, join_truth, report_markdown

    merged = join_truth(_load_results(args.results), pd.read_csv(args.truth, dtype=str),
                        truth_id=args.truth_id, label=args.label)
    metrics = evaluate(merged, label=args.label)
    by = args.by if args.by in merged else "is_shell"
    errors = merged[(merged["shell_probability"] >= 0.5) != merged[args.label].astype(bool)]
    text = report_markdown(metrics, breakdown(merged, by=by, label=args.label), errors.head(args.max_errors))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        _log(f"report written to {args.out}")
    print(text)
    return 0


def cmd_train(args) -> int:
    import pandas as pd

    from .config import load_config
    from .evaluate import join_truth
    from .train import fit_weights, save_weights

    cfg = load_config(args.config)
    merged = join_truth(_load_results(args.results), pd.read_csv(args.labels, dtype=str),
                        truth_id=args.labels_id, label=args.label)
    result = fit_weights(merged, merged[args.label].to_numpy(), cfg, l2=args.l2)
    save_weights(result, args.out)
    moved = sorted(result["weights"], key=lambda k: -abs(result["weights"][k] - result["prior_weights"][k]))[:10]
    print(f"trained on {result['n_train']} labels ({result['n_positive']} shells); intercept {result['intercept']}")
    for k in moved:
        print(f"  {k:28s} {result['prior_weights'][k]:+6.2f} -> {result['weights'][k]:+6.2f}  (support {result['support'][k]})")
    print(f"weights written to {args.out}; use with: shell-identifier run ... --weights {args.out}")
    return 0


def cmd_plot(args) -> int:
    import geopandas as gpd
    import pandas as pd

    from .assets import load_building_layer, load_parcels
    from .plot import plot_parcel

    ds = _datasets(args)
    scored = gpd.read_file(Path(args.results) / "address_scores.gpkg", layer="address_scores")
    scored["parcel_pid"] = scored["parcel_pid"].fillna("")
    parcels = load_parcels(ds["cad"], log=_log)[0].to_geopandas()
    buildings = load_building_layer(ds["bld"], log=_log).to_geopandas() if ds["bld"].get("path") else None
    pids = list(args.parcel or [])
    if not pids:
        summary = pd.read_csv(Path(args.results) / "parcel_summary.csv", dtype={"parcel_pid": str})
        pool = summary[summary["pattern"] == args.pattern] if args.pattern else summary[summary["n_addresses"] > 1]
        pids = pool.sample(min(args.n, len(pool)), random_state=args.seed)["parcel_pid"].tolist()
    for pid in pids:
        print(plot_parcel(pid, scored, parcels, buildings, Path(args.out) / f"parcel_{pid}.png"))
    return 0


# ---------------------------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------------------------

def _dataset_args(p: argparse.ArgumentParser, labels: bool = False) -> None:
    g = p.add_argument_group("datasets (paths here override the datasets YAML)")
    g.add_argument("--datasets", help="YAML with each dataset's path, table name and columns "
                                      "(see `shell-identifier datasets template`)")
    g.add_argument("--ds-asset", dest="ds_asset", help="addresses: file, folder or database")
    g.add_argument("--cad", "--cadastre", dest="cad", help="cadastre parcels: file, folder or database")
    g.add_argument("--bld", "--buildings", dest="bld", help="building footprints: file, folder or database")
    g.add_argument("--bld-address", "--building-address", dest="bld_address", help="optional building_address links")
    g.add_argument("--bld-cadastre", "--building-cadastre", dest="bld_cadastre",
                   help="optional building_cadastre links")
    if labels:
        g.add_argument("--labelled-asset", "--labels", dest="labelled_asset",
                       help="address-level PD / out-of-scope labels")
    g.add_argument("--counts", help="optional parcel dwelling counts (parcel_pid, n_dwellings)")
    g.add_argument("--table", action="append", metavar="ROLE=NAME",
                   help="table / layer name for a dataset, e.g. --table ds_asset=address_2026 (repeatable)")
    g.add_argument("--gnaf", help="addresses from a G-NAF PSV release folder (instead of --ds-asset)")
    g.add_argument("--text", help="addresses as single-line text with coordinates (instead of --ds-asset)")
    o = p.add_argument_group("dataset options")
    o.add_argument("--states", nargs="+", help="G-NAF states to load (default: all found)")
    o.add_argument("--address-col", help="single-line address column (text addresses)")
    o.add_argument("--lon-col")
    o.add_argument("--lat-col")
    o.add_argument("--id-col", help="address id column of ds_asset")
    o.add_argument("--localities", help="text file of locality names (one per line) to help the parser")
    o.add_argument("--crs", help="CRS of the address coordinates (EPSG:7844 GDA2020, EPSG:4283 GDA94)")
    o.add_argument("--cadastre-layer", help="table / layer name of the cadastre")
    o.add_argument("--buildings-layer", help="table / layer name of the buildings")
    o.add_argument("--building-height", help="optional Building Height table (building_pid, roof_height)")
    o.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                   help="restrict to a lon/lat window")
    if labels:
        o.add_argument("--labels-id", help="address id column of labelled_asset")
        o.add_argument("--label-col", help="label column of labelled_asset")
        o.add_argument("--pd-values", nargs="+", help="label values meaning private dwelling (others: not PD)")


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML overriding default_config.yaml")
    p.add_argument("--weights", help="shell weights JSON from `train`")
    p.add_argument("--no-gpkg", action="store_true", help="skip the GeoPackage output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shell-identifier",
                                description="Shell addresses and private dwellings on cadastral parcels.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate synthetic G-NAF / Cadastre / Buildings data with ground truth")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--parcels", type=int, default=1500)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("parse", help="parse free-text addresses and print components")
    s.add_argument("text", nargs="+")
    s.set_defaults(func=cmd_parse)

    d = sub.add_parser("datasets", help="write or check the datasets YAML (paths, table names, columns)")
    dsub = d.add_subparsers(dest="datasets_command", required=True)
    s = dsub.add_parser("template", help="print (or write) the datasets YAML template")
    s.add_argument("--out", help="write the template here instead of printing it")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_datasets_template)
    s = dsub.add_parser("check", help="resolve every dataset and show the column mapping it will use")
    _dataset_args(s, labels=True)
    s.set_defaults(func=cmd_datasets_check)

    s = sub.add_parser("run", help="score addresses for shell likelihood")
    _dataset_args(s)
    _model_args(s)
    s.add_argument("--out", default="results")
    s.set_defaults(func=cmd_run)

    d = sub.add_parser("dwellings", help="count-constrained dwelling model trained on PD / out-of-scope labels")
    dsub = d.add_subparsers(dest="dwellings_command", required=True)
    s = dsub.add_parser("verify", help="fit on labelled parcels; move misplaced PD labels; find primary shells")
    _dataset_args(s, labels=True)
    _model_args(s)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--iterations", type=int, default=8)
    s.add_argument("--misplaced-rate", type=float, default=0.25,
                   help="prior share of multi-address parcels with at least one PD label on the wrong address")
    s.add_argument("--max-misplaced", type=int, default=3, help="most misplaced PD labels considered per parcel")
    s.add_argument("--write-features", action="store_true")
    s.add_argument("--out", default="results/dwellings")
    s.set_defaults(func=cmd_dwellings_verify)

    s = dsub.add_parser("predict", help="apply a model from `verify` to new addresses")
    _dataset_args(s)
    _model_args(s)
    s.add_argument("--model", required=True, help="dwelling_model.joblib written by `verify`")
    s.add_argument("--out", default="results/dwellings_predicted")
    s.set_defaults(func=cmd_dwellings_predict)

    s = dsub.add_parser("evaluate", help="compare dwelling assignments with truth (is_dwelling, is_shell)")
    s.add_argument("--results", required=True)
    s.add_argument("--truth", required=True)
    s.add_argument("--truth-id", default="address_detail_pid")
    s.add_argument("--out")
    s.set_defaults(func=cmd_dwellings_evaluate)

    s = sub.add_parser("evaluate", help="compare a shell-score run against labelled truth")
    s.add_argument("--results", required=True, help="output folder of `run`")
    s.add_argument("--truth", required=True, help="CSV with an id column and a 0/1 label")
    s.add_argument("--truth-id", default="address_detail_pid")
    s.add_argument("--label", default="is_shell")
    s.add_argument("--by", default="role", help="truth column to break results down by")
    s.add_argument("--max-errors", type=int, default=40)
    s.add_argument("--out", help="write the markdown report here")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("train", help="refit shell weights from labelled addresses (anchored to defaults)")
    s.add_argument("--results", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--labels-id", default="address_detail_pid")
    s.add_argument("--label", default="is_shell")
    s.add_argument("--l2", type=float, default=1.0)
    s.add_argument("--config")
    s.add_argument("--out", default="weights.json")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("plot", help="PNG review maps of parcels")
    s.add_argument("--results", required=True)
    s.add_argument("--datasets")
    s.add_argument("--cad", "--cadastre", dest="cad")
    s.add_argument("--bld", "--buildings", dest="bld")
    s.add_argument("--table", action="append", metavar="ROLE=NAME")
    s.add_argument("--parcel", nargs="+", help="parcel_pid(s) to plot")
    s.add_argument("--pattern", help="sample parcels with this pattern from parcel_summary.csv")
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/plots")
    s.set_defaults(func=cmd_plot)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

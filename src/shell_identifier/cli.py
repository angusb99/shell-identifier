"""Command line interface: ``shell-identifier <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def _gpkg_layers(path: str | None) -> set[str]:
    if not path or not str(path).lower().endswith(".gpkg"):
        return set()
    import pyogrio

    return {str(row[0]) for row in pyogrio.list_layers(path)}


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


def cmd_run(args) -> int:
    from .config import load_config
    from .loaders import (load_building_address, load_building_cadastre, load_buildings, load_cadastre,
                          load_gnaf, load_text_addresses)
    from .pipeline import run_pipeline, write_outputs

    cfg = load_config(args.config, args.weights)
    bbox = tuple(args.bbox) if args.bbox else None
    t0 = time.time()

    if args.gnaf:
        addresses = load_gnaf(args.gnaf, states=args.states, crs=args.crs)
        _log(f"G-NAF: {len(addresses):,} active addresses")
    else:
        localities = None
        if args.localities:
            localities = {l.strip().upper() for l in Path(args.localities).read_text(encoding="utf-8").splitlines() if l.strip()}
        addresses = load_text_addresses(args.text, address_col=args.address_col, lon_col=args.lon_col,
                                        lat_col=args.lat_col, id_col=args.id_col, crs=args.crs, localities=localities)
        _log(f"text: {len(addresses):,} addresses parsed")
    if bbox:
        addresses = addresses.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    parcels, strata = load_cadastre(args.cadastre, layer=args.cadastre_layer, bbox=bbox,
                                    use_base_parcels=cfg["cadastre"]["use_base_parcels"])
    _log(f"cadastre: {len(parcels):,} base parcel polygons, {len(strata):,} strata polygons")

    buildings = links = bcad = None
    if args.buildings:
        buildings = load_buildings(args.buildings, layer=args.buildings_layer, bbox=bbox,
                                   height_table=args.building_height,
                                   storey_height_m=cfg["spatial"]["storey_height_m"])
        layers = _gpkg_layers(args.buildings)
        ba_path = args.building_address or (args.buildings if "building_address" in layers else None)
        bc_path = args.building_cadastre or (args.buildings if "building_cadastre" in layers else None)
        links = load_building_address(ba_path) if ba_path else None
        bcad = load_building_cadastre(bc_path) if bc_path else None
        _log(f"buildings: {len(buildings):,} footprints, "
             f"{0 if links is None else len(links):,} address links, {0 if bcad is None else len(bcad):,} cadastre links")

    result = run_pipeline(addresses, parcels, buildings, links, bcad, cfg)
    paths = write_outputs(result, args.out)
    s = result["addresses"]
    _log(f"scored {len(s):,} addresses in {time.time() - t0:.1f}s")
    print(s["shell_class"].value_counts().to_string())
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _load_results(results: str) -> pd.DataFrame:
    d = Path(results)
    scores = pd.read_csv(d / "address_scores.csv", dtype={"parcel_pid": str})
    feats = pd.read_csv(d / "address_features.csv", dtype={"parcel_pid": str})
    feats = feats.drop(columns=[c for c in ("parcel_pid", "shell_probability") if c in feats])
    return scores.merge(feats, on="address_pid", how="left")


def cmd_evaluate(args) -> int:
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

    from .loaders import load_buildings, load_cadastre
    from .plot import plot_parcel

    scored = gpd.read_file(Path(args.results) / "address_scores.gpkg", layer="address_scores")
    scored["parcel_pid"] = scored["parcel_pid"].fillna("")
    parcels, _ = load_cadastre(args.cadastre)
    buildings = load_buildings(args.buildings) if args.buildings else None
    pids = list(args.parcel or [])
    if not pids:
        summary = pd.read_csv(Path(args.results) / "parcel_summary.csv", dtype={"parcel_pid": str})
        pool = summary[summary["pattern"] == args.pattern] if args.pattern else summary[summary["n_addresses"] > 1]
        pids = pool.sample(min(args.n, len(pool)), random_state=args.seed)["parcel_pid"].tolist()
    for pid in pids:
        print(plot_parcel(pid, scored, parcels, buildings, Path(args.out) / f"parcel_{pid}.png"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shell-identifier",
                                description="Score addresses on cadastral parcels for the likelihood of being shell addresses.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate synthetic G-NAF / Cadastre / Buildings data with ground truth")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--parcels", type=int, default=1500)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("parse", help="parse free-text addresses and print components")
    s.add_argument("text", nargs="+")
    s.set_defaults(func=cmd_parse)

    s = sub.add_parser("run", help="score addresses")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--gnaf", help="folder containing a G-NAF PSV release")
    src.add_argument("--text", help="CSV of free-text addresses with coordinates")
    s.add_argument("--states", nargs="+", help="G-NAF states to load (default: all found)")
    s.add_argument("--address-col", default="address")
    s.add_argument("--lon-col", default="longitude")
    s.add_argument("--lat-col", default="latitude")
    s.add_argument("--id-col", default=None)
    s.add_argument("--localities", help="text file of locality names (one per line) to help the parser")
    s.add_argument("--crs", default="EPSG:7844", help="CRS of address coordinates (EPSG:7844 GDA2020, EPSG:4283 GDA94)")
    s.add_argument("--cadastre", required=True, help="Geoscape Cadastre file (gpkg/gdb/shp/geojson)")
    s.add_argument("--cadastre-layer", default=None)
    s.add_argument("--buildings", help="Geoscape Buildings file")
    s.add_argument("--buildings-layer", default="buildings")
    s.add_argument("--building-address", help="building_address table (default: layer in --buildings gpkg)")
    s.add_argument("--building-cadastre", help="building_cadastre table (default: layer in --buildings gpkg)")
    s.add_argument("--building-height", help="optional Building Height insight pack table (building_pid, roof_height)")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                   help="restrict to a lon/lat window")
    s.add_argument("--config", help="YAML overriding default_config.yaml")
    s.add_argument("--weights", help="weights JSON from `train`")
    s.add_argument("--out", default="results")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("evaluate", help="compare a run against labelled truth")
    s.add_argument("--results", required=True, help="output folder of `run`")
    s.add_argument("--truth", required=True, help="CSV with an id column and a 0/1 label")
    s.add_argument("--truth-id", default="address_detail_pid")
    s.add_argument("--label", default="is_shell")
    s.add_argument("--by", default="role", help="truth column to break results down by")
    s.add_argument("--max-errors", type=int, default=40)
    s.add_argument("--out", help="write the markdown report here")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("train", help="refit weights from labelled addresses (anchored to defaults)")
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
    s.add_argument("--cadastre", required=True)
    s.add_argument("--buildings")
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

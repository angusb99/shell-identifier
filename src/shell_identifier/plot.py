"""Review plots: one PNG per parcel showing parcels, buildings and scored address points."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np


def local_mga(gdf: gpd.GeoDataFrame) -> str:
    """MGA zone of the data (north-up, true shape at parcel scale)."""
    if gdf.empty:
        raise KeyError("no geometry to choose an MGA zone from")
    lon = gdf.to_crs(4326).total_bounds[[0, 2]].mean()
    zone = int((lon + 180) // 6) + 1
    gda94 = gdf.crs is not None and gdf.crs.to_epsg() in (4283, 3577)
    return f"EPSG:{(28300 if gda94 else 7800) + zone}"


def plot_parcel(parcel_pid: str, scored: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame,
                buildings: gpd.GeoDataFrame | None, out_path: str | Path, metric_crs: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_crs = metric_crs or local_mga(parcels[parcels["parcel_pid"] == parcel_pid])
    parcels_m = parcels.to_crs(metric_crs)
    target = parcels_m[parcels_m["parcel_pid"] == parcel_pid]
    if target.empty:
        raise KeyError(f"parcel {parcel_pid} not found")
    minx, miny, maxx, maxy = target.total_bounds
    pad = max(maxx - minx, maxy - miny) * 0.35 + 5
    view = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    fig, ax = plt.subplots(figsize=(9, 9))
    near = parcels_m.cx[view[0]:view[2], view[1]:view[3]]
    near.boundary.plot(ax=ax, color="#9aa0a6", linewidth=0.6)
    target.plot(ax=ax, facecolor="#e8f0fe", edgecolor="#1a73e8", linewidth=1.6)
    if buildings is not None:
        b = buildings.to_crs(metric_crs).cx[view[0]:view[2], view[1]:view[3]]
        b.plot(ax=ax, facecolor="#dadce0", edgecolor="#5f6368", linewidth=0.6)

    s = scored[scored["parcel_pid"] == parcel_pid].to_crs(metric_crs)
    if not s.empty:
        # Fan out stacked points so every address is visible and labelled.
        key = s.geometry.x.round(1).astype(str) + "_" + s.geometry.y.round(1).astype(str)
        order = s.groupby(key).cumcount()
        size = s.groupby(key)["address_pid"].transform("size")
        ang = 2 * np.pi * order / size.clip(lower=1)
        r = np.where(size > 1, 1.2 + 0.08 * size, 0.0)
        xs, ys = s.geometry.x + r * np.cos(ang), s.geometry.y + r * np.sin(ang)
        sc = ax.scatter(xs, ys, c=s["shell_probability"], cmap="RdYlBu_r", vmin=0, vmax=1,
                        s=60 if len(s) <= 25 else 25, edgecolors="black", linewidths=0.6, zorder=5)
        ranked = s.sort_values("shell_probability", ascending=False)
        # Small parcels: label everything. Large complexes: label only the candidates.
        show = ranked.index if len(s) <= 12 else ranked[ranked["shell_probability"] >= 0.2].head(12).index

        def short(label: str) -> str:
            parts = str(label).split(",")
            return ", ".join(parts[:-1]) if len(parts) > 1 else parts[0]

        for i in show:
            ax.annotate(f"{short(s.at[i, 'address_label'])}  p={s.at[i, 'shell_probability']:.2f}", (xs[i], ys[i]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        counts = s["shell_class"].value_counts()
        summary = [f"{len(s)} addresses"] + [f"{counts.get(c, 0)} {c}" for c in
                                              ("SHELL", "LIKELY_SHELL", "UNCERTAIN", "LIKELY_NOT_SHELL", "NOT_SHELL")
                                              if counts.get(c, 0)]
        ax.text(0.01, 0.99, "\n".join(summary), transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#9aa0a6"))
        fig.colorbar(sc, ax=ax, shrink=0.6, label="shell probability")
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Parcel {parcel_pid}", fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path

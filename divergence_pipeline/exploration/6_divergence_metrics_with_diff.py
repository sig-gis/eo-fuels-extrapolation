import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
from sklearn.metrics import f1_score, jaccard_score, root_mean_squared_error


TEMP_ROOT = Path.cwd() / "temp"
PYROME_ROOT = Path.cwd() / "temp"
OUTPUT_ROOT = Path.cwd() / "outputs"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def normalize_tile_id(tile_value):
    return str(tile_value).zfill(5)


def fuel_label_suffix(label_name: str) -> str:
    if label_name == "FBFM40":
        return "fm40"
    return "fm40parent"


def list_output_tiles(tilenums=None):
    tiles = {}
    requested = {normalize_tile_id(t) for t in tilenums} if tilenums else None
    for path in sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir()):
        if not path.name.isdigit():
            continue
        tile = normalize_tile_id(path.name)
        if requested and tile not in requested:
            continue
        tiles[tile] = path
    return tiles


def list_pyrome_tiles(pyrome_id, tilenums=None):
    pyrome_dir = PYROME_ROOT / f"pyrome_{pyrome_id}"
    if not pyrome_dir.exists():
        raise FileNotFoundError(f"Pyrome temp folder not found: {pyrome_dir}")

    tiles = {}
    requested = {normalize_tile_id(t) for t in tilenums} if tilenums else None
    for path in sorted(p for p in pyrome_dir.iterdir() if p.is_dir()):
        folder = path.name
        if not folder.startswith("tile_"):
            continue
        tilenum = normalize_tile_id(folder.replace("tile_", "", 1))
        if requested and tilenum not in requested:
            continue
        tiles[tilenum] = path
    return tiles


def map_tiles_to_temp_dirs(pyrome_ids, tilenums=None):
    all_tiles = {}
    for pyrome_id in pyrome_ids:
        try:
            tiles = list_pyrome_tiles(pyrome_id, tilenums=tilenums)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue
        for tile, tile_dir in tiles.items():
            all_tiles[tile] = {
                "temp_dir": tile_dir,
                "pyrome_id": pyrome_id,
            }
    return all_tiles


def load_accuracy_rows(pyrome_ids, accuracy_col="accuracy"):
    csv_paths = sorted(OUTPUT_ROOT.glob("tile_fbfm40_accuracy_metrics_pyrome_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No step-4 tile accuracy CSV files found in {OUTPUT_ROOT}."
        )

    frames = []
    requested_ids = {int(p) for p in pyrome_ids} if pyrome_ids else None
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "tile" not in df.columns:
            continue
        if accuracy_col not in df.columns:
            continue

        if requested_ids and "pyrome_id" in df.columns:
            df = df[df["pyrome_id"].isin(requested_ids)]

        if df.empty:
            continue

        use_cols = ["tile", accuracy_col]
        if "pyrome_id" in df.columns:
            use_cols.append("pyrome_id")
        frames.append(df[use_cols].copy())

    if not frames:
        raise ValueError(
            "No usable accuracy rows found for requested pyromes. "
            "Check --pyromes and accuracy CSV contents."
        )

    merged = pd.concat(frames, ignore_index=True)
    merged["tile"] = merged["tile"].apply(normalize_tile_id)
    merged = merged.drop_duplicates(subset=["tile"], keep="last")
    return merged


def percent_diff(label_path, pred_path):
    with rasterio.open(pred_path) as src:
        pred = src.read(1).flatten()
    with rasterio.open(label_path) as src:
        label = src.read(1).flatten()

    diff = pred - label
    denom = label.max() - label.min()
    if denom == 0:
        pct = np.zeros_like(diff, dtype=float)
    else:
        pct = diff / denom * 100

    return (
        float(np.mean(diff)),
        float(np.mean(pct)),
        float(np.mean(np.abs(pct))),
    )


def accuracy_metric(y_true, y_pred, metric, **kwargs):
    metrics = {
        "jaccard": jaccard_score,
        "f1": f1_score,
        "rmse": lambda a, b: root_mean_squared_error(a, b),
    }
    fn = metrics[metric]

    with rasterio.open(y_true) as src:
        a = src.read(1).flatten()
    with rasterio.open(y_pred) as src:
        b = src.read(1).flatten()

    return float(fn(a, b, **kwargs))


def metric_filename(metric_name: str) -> str:
    cleaned = metric_name.lower().replace(" ", "_")
    cleaned = cleaned.replace("/", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def plot_metrics_vs_accuracy(df, label_name: str, accuracy_col="accuracy"):
    plot_df = df.dropna(subset=[accuracy_col]).copy()
    if plot_df.empty:
        print("[PLOT] No rows with accuracy values available; skipping plots")
        return

    if "pyrome_id" in plot_df.columns:
        plot_df["pyrome_cat"] = "Pyrome " + plot_df["pyrome_id"].astype(str)

    non_metric = {"tile", "pyrome_id", accuracy_col}
    metric_cols = [
        col
        for col in plot_df.columns
        if col not in non_metric and pd.api.types.is_numeric_dtype(plot_df[col])
    ]

    label_tag = fuel_label_suffix(label_name)
    for metric in metric_cols:
        color_col = "pyrome_cat" if "pyrome_cat" in plot_df.columns else None
        fig = px.scatter(
            plot_df,
            x=accuracy_col,
            y=metric,
            color=color_col,
            text="tile",
            title=f"{metric} vs Tile Accuracy ({label_name})",
            labels={
                accuracy_col: "Tile Accuracy",
                metric: metric,
                "pyrome_cat": "Pyrome",
            },
        )
        fig.update_traces(marker=dict(size=12), textposition="top center")

        fit_df = plot_df[[accuracy_col, metric]].dropna()
        if len(fit_df) >= 2:
            x = fit_df[accuracy_col].to_numpy(dtype=float)
            y = fit_df[metric].to_numpy(dtype=float)
            if np.unique(x).size >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                fig.add_scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="best fit",
                    line=dict(color="black", width=2),
                )
            else:
                print(f"[PLOT] Skipping best-fit for {metric}: x values are constant")
        else:
            print(f"[PLOT] Skipping best-fit for {metric}: fewer than 2 points")

        out_html = OUTPUT_ROOT / f"accuracy_vs_{metric_filename(metric)}_{label_tag}.html"
        fig.write_html(out_html)
        print(f"[PLOT] Saved {out_html}")


def run_metrics(tile_info_map, tile_to_outputdir, accuracy_map, year, label_name):
    categories = [
        "fire_type",
        "fireline_intensity",
        "flame_length",
        "spread_rate",
        "spread_direction",
    ]

    suffix = fuel_label_suffix(label_name)
    rows = []
    print("\n[METRICS] Computing divergence metrics")

    for tile in sorted(tile_info_map.keys()):
        tile_info = tile_info_map[tile]
        tile_dir = tile_info["temp_dir"]
        pyrome_id = tile_info["pyrome_id"]
        out_dir = tile_to_outputdir.get(tile)
        if out_dir is None:
            print(f"[WARN] Tile {tile} missing outputs folder; skipping")
            continue

        fm40_label = tile_dir / f"tilenum{tile}_{suffix}label_{year}.tif"
        fm40_pred = out_dir / f"tilenum{tile}_{suffix}_pred_{year}.tif"
        if not fm40_label.exists() or not fm40_pred.exists():
            print(f"[WARN] Missing FM40 label/pred rasters for tile {tile}; skipping")
            continue

        row = {
            "tile": tile,
            "pyrome_id": pyrome_id,
            "accuracy": accuracy_map.get(tile, np.nan),
        }

        diff, pct, pct_abs = percent_diff(fm40_label, fm40_pred)
        row["fm40_diff"] = diff
        row["fm40_pct_diff"] = pct
        row["fm40_pct_diff_abs"] = pct_abs
        row["fm40_f1_score"] = accuracy_metric(
            fm40_label,
            fm40_pred,
            "f1",
            average="weighted",
        )

        for category in categories:
            true_pt = out_dir / f"LF_FBFM40_{category}.tif"
            pred_pt = out_dir / f"predicted_FBFM40_{category}.tif"
            if not true_pt.exists() or not pred_pt.exists():
                print(f"[WARN] Missing pyretechnics outputs for tile {tile}, category {category}")
                continue

            diff, pct, pct_abs = percent_diff(true_pt, pred_pt)
            row[f"{category}_diff"] = diff
            row[f"{category}_pct_diff"] = pct
            row[f"{category}_pct_diff_abs"] = pct_abs

            metric = "jaccard" if category == "fire_type" else "rmse"
            kwargs = {"average": "weighted"} if category == "fire_type" else {}
            row[f"{category}_{metric}"] = accuracy_metric(
                true_pt,
                pred_pt,
                metric,
                **kwargs,
            )

        rows.append(row)
        print(f"[METRICS] Finished tile {tile}")

    if not rows:
        raise ValueError("No tile metrics were computed. Check inputs and output files.")

    df = pd.DataFrame(rows).sort_values("tile")
    label_tag = fuel_label_suffix(label_name)
    csv_path = OUTPUT_ROOT / f"fm40_divergence_metrics_{label_tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[METRICS] Saved CSV → {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Compute divergence metrics and plot each metric vs tile accuracy"
    )
    parser.add_argument(
        "--pyromes",
        nargs="+",
        type=int,
        required=True,
        help="Pyrome IDs to process",
    )
    parser.add_argument(
        "--tilenums",
        nargs="+",
        help="Optional tile IDs to process",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Year used in filenames (default: 2022)",
    )
    parser.add_argument(
        "--label",
        choices=["FBFM40", "FBFM40Parent"],
        default="FBFM40",
        help="Fuel label mode to evaluate.",
    )
    parser.add_argument(
        "--accuracy-col",
        default="accuracy",
        help="Column name in step-4 CSV to use on x-axis (default: accuracy).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable HTML plot export (metrics CSV still exported).",
    )

    args = parser.parse_args()

    try:
        accuracy_df = load_accuracy_rows(args.pyromes, accuracy_col=args.accuracy_col)
        accuracy_map = {
            normalize_tile_id(row["tile"]): float(row[args.accuracy_col])
            for _, row in accuracy_df.iterrows()
        }
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARN] {exc}")
        print("[WARN] Continuing without accuracy values; plots may be skipped.")
        accuracy_map = {}

    requested_tiles = (
        {normalize_tile_id(t) for t in args.tilenums}
        if args.tilenums
        else None
    )

    output_tiles = list_output_tiles(tilenums=requested_tiles)
    if not output_tiles:
        raise ValueError("No tile folders found in outputs/ for the requested filters.")

    temp_tiles = map_tiles_to_temp_dirs(args.pyromes, tilenums=set(output_tiles.keys()))

    shared_tiles = sorted(set(output_tiles.keys()) & set(temp_tiles.keys()))
    if not shared_tiles:
        raise ValueError(
            "No matching tiles found across accuracy CSVs, outputs/, and temp/pyrome_* folders."
        )

    tile_to_outputdir = {tile: output_tiles[tile] for tile in shared_tiles}
    tile_info_map = {tile: temp_tiles[tile] for tile in shared_tiles}
    accuracy_map = {tile: accuracy_map.get(tile, np.nan) for tile in shared_tiles}

    df = run_metrics(
        tile_info_map,
        tile_to_outputdir,
        accuracy_map,
        args.year,
        args.label,
    )

    if not args.no_plot:
        plot_metrics_vs_accuracy(df, args.label, accuracy_col=args.accuracy_col)


if __name__ == "__main__":
    main()
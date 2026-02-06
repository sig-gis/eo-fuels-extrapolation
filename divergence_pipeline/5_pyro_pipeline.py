import argparse
from pathlib import Path
import numpy as np
import joblib
import rasterio
import pandas as pd
import plotly.express as px

from pyretechnics.load_landfire import (
    load_and_convert_landfire_rasters,
    convert_rasters_to_space_time_cubes,
)
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.burn_cells as bc

from sklearn.metrics import (
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

TEMP_ROOT = Path.cwd() / "temp"
PYROME_ROOT = Path.cwd() / "temp"
OUTPUT_ROOT = Path.cwd() / "outputs"
MODEL_ROOT = Path.cwd() / "data"

TEMP_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MERGED_MODEL_NAME = "rf_pyromes_merged.joblib"


def seasonal_band_names():
    seasons = ["fall", "spring", "summer", "winter"]
    return [f"B{band}_{season}" for band in range(2, 8) for season in seasons]


def spectral_stat_names():
    stats = ["max", "median", "min"]
    indices = ["EVI", "MSAVI", "NBR", "NDMI", "NDVI", "SAVI", "TCB", "TCG", "TSW", "VARI"]
    return [f"{idx}_{stat}" for idx in indices for stat in stats]


BAND_NAME_OVERRIDES = {
    "aef": lambda count: [f"aef_b{idx + 1}" for idx in range(count)],
    "fm40label": lambda count: [f"fm40label_b{idx + 1}" for idx in range(count)],
    "fm40parentlabel": lambda count: [f"fm40parentlabel_b{idx + 1}" for idx in range(count)],
    "ch": lambda count: [f"ch_b{idx + 1}" for idx in range(count)],
    "cc": lambda count: [f"cc_b{idx + 1}" for idx in range(count)],
    "cbh": lambda count: [f"cbh_b{idx + 1}" for idx in range(count)],
    "cbd": lambda count: [f"cbd_b{idx + 1}" for idx in range(count)],
    "elevation": lambda count: [f"elevation_b{idx + 1}" for idx in range(count)],
    "slope": lambda count: [f"slope_b{idx + 1}" for idx in range(count)],
    "aspect": lambda count: [f"aspect_b{idx + 1}" for idx in range(count)],
    "mtpi": lambda count: [f"mtpi_b{idx + 1}" for idx in range(count)],
    "bps": lambda count: [f"bps_b{idx + 1}" for idx in range(count)],
    "evc": lambda count: [f"evc_b{idx + 1}" for idx in range(count)],
    "evt": lambda count: [f"evt_b{idx + 1}" for idx in range(count)],
    "evh": lambda count: [f"evh_b{idx + 1}" for idx in range(count)],
    "spectral_stats": lambda count: [f"spectral_stats_b{idx + 1}" for idx in range(count)],
    "hls_band_stats": lambda count: [f"hls_band_stats_b{idx + 1}" for idx in range(count)],
    "climatenormals": lambda count: [f"climatenormals_b{idx + 1}" for idx in range(count)],
    "prism": lambda count: [f"prism_b{idx + 1}" for idx in range(count)],
}


def guess_layer_prefix(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if parts and parts[0].startswith("tilenum"):
        parts = parts[1:]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    if not parts:
        return stem
    return "_".join(parts)


def get_band_names(dataset, prefix):
    override = BAND_NAME_OVERRIDES.get(prefix)
    descriptions = list(dataset.descriptions) if dataset.descriptions else []
    if override:
        return override(dataset.count)

    if dataset.count == 1:
        return [f"{prefix}_b1"]

    names = []
    for idx in range(dataset.count):
        desc = descriptions[idx] if idx < len(descriptions) else None
        names.append(desc if desc else f"{prefix}_b{idx + 1}")

    return names

def load_rf_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print("Loaded RF model:", type(model))
    scaler_path = model_path.with_name(f"{model_path.stem}_scaler.joblib")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("Loaded scaler:", type(scaler))
    return model, scaler


def resolve_model_path(pyrome_id: int, merge: bool) -> Path:
    if merge:
        return MODEL_ROOT / MERGED_MODEL_NAME
    return MODEL_ROOT / f"rf_pyrome_{pyrome_id}.joblib"

# def run_rf_inference(model, tilenum, year):
#     tile_dir = TEMP_ROOT / tilenum
#     aef_tile = next(tile_dir.glob("*aef*.tif"))

#     with rasterio.open(aef_tile) as src:
#         aef = src.read()
#         profile = src.profile

#     bands, h, w = aef.shape
#     data = np.transpose(aef, (1, 2, 0)).reshape(-1, bands)
#     preds = model.predict(data).reshape(h, w)

#     out_dir = OUTPUT_ROOT / tilenum
#     out_dir.mkdir(parents=True, exist_ok=True)

#     out_path = out_dir / f"tilenum{tilenum}_aef_{year}_pred.tif"

#     profile.update(count=1, dtype="uint8", compress="lzw")

#     with rasterio.open(out_path, "w", **profile) as dst:
#         dst.write(preds, 1)

#     print(f"[RF] Wrote {out_path}")

def run_rf_inference(model, scaler, tile_dir: Path, tilenum, year):
    tif_files = [f for f in tile_dir.glob("*.tif") if "label" not in f.name]

    print(f"Model expects {model.n_features_in_} features")
    print(f"Feature names: {model.feature_names_in_}")

    all_bands = []
    band_names = []
    profile = None
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            data = src.read()
            all_bands.append(data)
            # If band names are available
            layer_prefix = guess_layer_prefix(tif_file.name)
            band_names.extend(get_band_names(src, layer_prefix))
            if profile is None:
                profile = src.profile

    features = np.concatenate(all_bands, axis=0)
    bands, h, w = features.shape
    print(f"Input has {bands} features")
    print(f"Band names: {band_names}")

    expected_features = set(model.feature_names_in_)
    actual_features = set(band_names)

    missing = expected_features - actual_features
    extra = actual_features - expected_features

    print(f"Missing features ({len(missing)}): {sorted(missing)}")
    print(f"Extra features ({len(extra)}): {sorted(extra)}")

    # Filter to only common features in the model's expected order
    common = expected_features & actual_features
    ordered_features = [f for f in model.feature_names_in_ if f in common]
    indices = [band_names.index(f) for f in ordered_features]

    features_filtered = features[indices, :, :]
    bands = len(indices)
    print(f"Using {bands} common features: {ordered_features}")

    data = np.transpose(features_filtered, (1, 2, 0)).reshape(-1, bands)
    if scaler is not None:
        data = scaler.transform(data)
    preds = model.predict(data).reshape(h, w)

    out_dir = OUTPUT_ROOT / tilenum
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"tilenum{tilenum}_fm40parent_pred_{year}.tif"

    profile.update(count=1, dtype="uint8", compress="lzw")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(preds, 1)

    print(f"[RF] Wrote {out_path}")


def run_fire_behavior(tile_dir: Path, tilenum, year):
    out_dir = OUTPUT_ROOT / tilenum

    def f(k): return next(tile_dir.glob(f"*{k}*.tif"))

    lf_inputs = {
        "elevation": f("elevation"),
        "slope": f("slope"),
        "aspect": f("aspect"),
        "fuel_model": f("fm40parentlabel"),
        "canopy_cover": f("cc"),
        "canopy_height": f("ch"),
        "canopy_base_height": f("cbh"),
        "canopy_bulk_density": f("cbd"),
    }

    pred_inputs = lf_inputs | {
        "fuel_model": out_dir / f"tilenum{tilenum}_fm40parent_pred_{year}.tif"
    }

    with rasterio.open(lf_inputs["aspect"]) as src:
        rows, cols = src.shape
        transform = src.transform

    cube_shape = (1, rows, cols)

    lf_stc = convert_rasters_to_space_time_cubes(
        cube_shape, load_and_convert_landfire_rasters(lf_inputs)
    )
    pred_stc = convert_rasters_to_space_time_cubes(
        cube_shape, load_and_convert_landfire_rasters(pred_inputs)
    )

    wx = {k: SpaceTimeCube(cube_shape, v) for k, v in {
        "wind_speed_10m": 60.0,
        "upwind_direction": 0.0,
        "fuel_moisture_dead_1hr": 0.04,
        "fuel_moisture_dead_10hr": 0.05,
        "fuel_moisture_dead_100hr": 0.06,
        "fuel_moisture_live_herbaceous": 0.30,
        "fuel_moisture_live_woody": 0.60,
        "foliar_moisture": 0.80,
        "fuel_spread_adjustment": 1.0,
        "weather_spread_adjustment": 1.0,
    }.items()}

    lf_stc.update(wx)
    pred_stc.update(wx)

    lf_fire = bc.burn_all_cells_as_head_fire(lf_stc, 0, (0, rows), (0, cols))
    pred_fire = bc.burn_all_cells_as_head_fire(pred_stc, 0, (0, rows), (0, cols))

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:5070",
        "transform": transform,
        "nodata": -9999,
    }

    for k in lf_fire:
        for label, arr in [("LF", lf_fire[k]), ("predicted", pred_fire[k])]:
            with rasterio.open(out_dir / f"{label}_FBFM40_{k}.tif", "w", **profile) as dst:
                dst.write(arr, 1)

def scatter_fm40_vs_category(
    df,
    fm40_metric,
    category,
    category_metric_suffix,
    out_html,
    title_suffix="",
):
    """
    Scatter plot where:
      x = fm40_metric
      y = <category>_<category_metric_suffix>
      dot = tile
    """

    y_metric = f"{category}_{category_metric_suffix}"

    if fm40_metric not in df.index:
        raise ValueError(f"{fm40_metric} not found in dataframe")
    if y_metric not in df.index:
        raise ValueError(f"{y_metric} not found in dataframe")

    plot_df = pd.DataFrame({
        "tile": df.columns,
        fm40_metric: df.loc[fm40_metric].values,
        y_metric: df.loc[y_metric].values,
    })

    fig = px.scatter(
        plot_df,
        x=fm40_metric,
        y=y_metric,
        text="tile",
        title=(
            f"{fm40_metric.replace('_',' ').title()} vs "
            f"{category.replace('_',' ').title()} {title_suffix}"
        ),
        labels={
            fm40_metric: fm40_metric.replace("_", " ").title(),
            y_metric: f"{category.replace('_',' ').title()} {title_suffix}",
        },
    )

    fig.update_traces(
        marker=dict(size=12),
        textposition="top center",
    )

    fig.write_html(out_html)
    print(f"[PLOT] Saved {out_html}")

def percent_diff(label_path, pred_path):
    with rasterio.open(pred_path) as src:
        pred = src.read(1).flatten()
    with rasterio.open(label_path) as src:
        label = src.read(1).flatten()

    diff = pred - label
    pct = diff / (label.max() - label.min()) * 100

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

def run_metrics(pyrome_tiles, year, do_plot=False):
    categories = [
        "fire_type",
        "fireline_intensity",
        "flame_length",
        "spread_rate",
        "spread_direction",
    ]

    results = {}

    print("\n[METRICS] Computing divergence metrics")

    for tile, tile_dir in pyrome_tiles.items():
        results[tile] = {}

        fm40_label = tile_dir / f"tilenum{tile}_fm40parentlabel_{year}.tif"
        fm40_pred = (
            OUTPUT_ROOT / tile
            / f"tilenum{tile}_fm40parent_pred_{year}.tif"
        )

        diff, pct, pct_abs = percent_diff(fm40_label, fm40_pred)
        results[tile]["fm40_diff"] = diff
        results[tile]["fm40_pct_diff"] = pct
        results[tile]["fm40_pct_diff_abs"] = pct_abs
        results[tile]["fm40_f1_score"] = accuracy_metric(
            fm40_label,
            fm40_pred,
            "f1",
            average="weighted",
        )

        for category in categories:
            true_pt = OUTPUT_ROOT / tile / f"LF_FBFM40_{category}.tif"
            pred_pt = OUTPUT_ROOT / tile / f"predicted_FBFM40_{category}.tif"

            diff, pct, pct_abs = percent_diff(true_pt, pred_pt)
            results[tile][f"{category}_diff"] = diff
            results[tile][f"{category}_pct_diff"] = pct
            results[tile][f"{category}_pct_diff_abs"] = pct_abs

            metric = "jaccard" if category == "fire_type" else "rmse"
            kwargs = {"average": "weighted"} if category == "fire_type" else {}

            results[tile][f"{category}_{metric}"] = accuracy_metric(
                true_pt,
                pred_pt,
                metric,
                **kwargs,
            )

        print(f"[METRICS] Finished tile {tile}")

    df = pd.DataFrame(results)
    csv_path = OUTPUT_ROOT / "fm40_divergence_metrics.csv"
    df.to_csv(csv_path)

    print(f"\n[METRICS] Saved CSV → {csv_path}")

    if not do_plot:
        print("[METRICS] Plotting disabled (use --plot to enable)")
        return

    print("[METRICS] Generating Plotly HTML plots")

    plot_categories = [
        "fireline_intensity",
        "flame_length",
        "spread_rate",
        "spread_direction",
    ]

    for category in plot_categories:

        scatter_fm40_vs_category(
            df,
            fm40_metric="fm40_f1_score",
            category=category,
            category_metric_suffix="rmse",
            out_html=OUTPUT_ROOT / f"fm40_f1_vs_{category}_rmse.html",
            title_suffix="RMSE",
        )

        scatter_fm40_vs_category(
            df,
            fm40_metric="fm40_pct_diff",
            category=category,
            category_metric_suffix="pct_diff",
            out_html=OUTPUT_ROOT / f"fm40_pct_diff_vs_{category}_pct_diff.html",
            title_suffix="Percent Difference",
        )

        scatter_fm40_vs_category(
            df,
            fm40_metric="fm40_diff",
            category=category,
            category_metric_suffix="diff",
            out_html=OUTPUT_ROOT / f"fm40_diff_vs_{category}_diff.html",
            title_suffix="Difference",
        )

    print("[METRICS] Plotting complete")

def list_pyrome_tiles(pyrome_id, tilenums=None):
    pyrome_dir = PYROME_ROOT / f"pyrome_{pyrome_id}"
    if not pyrome_dir.exists():
        raise FileNotFoundError(f"Pyrome temp folder not found: {pyrome_dir}")
    tiles = {}
    for path in sorted(p for p in pyrome_dir.iterdir() if p.is_dir()):
        folder = path.name
        if not folder.startswith("tile_"):
            continue
        tilenum = folder.replace("tile_", "", 1)
        tiles[tilenum] = path
    if tilenums:
        tiles = {tile: path for tile, path in tiles.items() if tile in tilenums}
    return tiles, pyrome_dir

def main():
    parser = argparse.ArgumentParser(
        description="Run full FM40 RF → fire behavior → evaluation pipeline"
    )
    parser.add_argument(
        "--pyromes",
        nargs="+",
        type=int,
        help="Pyrome IDs to process (uses temp/pyrome_{id} tiles)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Use merged pyrome model (rf_pyromes_merged.joblib) if available.",
    )
    parser.add_argument(
        "--tilenums",
        nargs="+",
        help="Tile numbers to process (filters tiles within the listed pyromes).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Year used in filenames (default: 2022)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, export Plotly HTML plots in addition to CSV metrics",
    )

    args = parser.parse_args()

    if not args.pyromes:
        raise ValueError("Provide --pyromes (tiles are read from temp/pyrome_{id}/tile_*/)")

    all_tiles = {}
    for pyrome_id in args.pyromes:
        model_path = resolve_model_path(pyrome_id, args.merge)
        model, scaler = load_rf_model(model_path)
        tiles, pyrome_dir = list_pyrome_tiles(pyrome_id, tilenums=args.tilenums)
        if not tiles:
            print(f"No tiles found for pyrome {pyrome_id} in {pyrome_dir}")
            continue
        for tilenum, tile_dir in tiles.items():
            run_rf_inference(model, scaler, tile_dir, tilenum, args.year)
            run_fire_behavior(tile_dir, tilenum, args.year)
        all_tiles.update(tiles)

    run_metrics(all_tiles, args.year, do_plot=args.plot)

if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import numpy as np
import joblib
import rasterio
from google.cloud import storage
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

TEMP_ROOT = Path.cwd() / "temp" / "geoai-fuels-tiles"
OUTPUT_ROOT = Path.cwd() / "outputs"

TEMP_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BUCKET_NAME = "geoai-fuels-tiles"
MODEL_BLOB = "utils/rfmodel_conus_1000ptsperclass_150trees.joblib"

def load_rf_model():
    local_model = TEMP_ROOT / MODEL_BLOB
    local_model.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB)

    if not blob.exists():
        raise FileNotFoundError(f"Model not found: gs://{BUCKET_NAME}/{MODEL_BLOB}")

    if not local_model.exists():
        print(f"Downloading RF model: {MODEL_BLOB}")
        blob.download_to_filename(local_model)

    model = joblib.load(local_model)
    print("Loaded RF model:", type(model))
    return model

def run_rf_inference(model, tilenum, year):
    tile_dir = TEMP_ROOT / tilenum
    aef_tile = next(tile_dir.glob("*aef*.tif"))

    with rasterio.open(aef_tile) as src:
        aef = src.read()
        profile = src.profile

    bands, h, w = aef.shape
    data = np.transpose(aef, (1, 2, 0)).reshape(-1, bands)
    preds = model.predict(data).reshape(h, w)

    out_dir = OUTPUT_ROOT / tilenum
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"tilenum{tilenum}_aef_{year}_pred.tif"

    profile.update(count=1, dtype="uint8", compress="lzw")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(preds, 1)

    print(f"[RF] Wrote {out_path}")

def run_fire_behavior(tilenum, year):
    tile_dir = TEMP_ROOT / tilenum
    out_dir = OUTPUT_ROOT / tilenum

    def f(k): return next(tile_dir.glob(f"*{k}*.tif"))

    lf_inputs = {
        "elevation": f("elevation"),
        "slope": f("slope"),
        "aspect": f("aspect"),
        "fuel_model": f("fm40label"),
        "canopy_cover": f("cc"),
        "canopy_height": f("ch"),
        "canopy_base_height": f("cbh"),
        "canopy_bulk_density": f("cbd"),
    }

    pred_inputs = lf_inputs | {
        "fuel_model": out_dir / f"tilenum{tilenum}_aef_{year}_pred.tif"
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

def run_metrics(year, do_plot=False):
    categories = [
        "fire_type",
        "fireline_intensity",
        "flame_length",
        "spread_rate",
        "spread_direction",
    ]

    results = {}

    print("\n[METRICS] Computing divergence metrics")

    for tile_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not tile_dir.is_dir():
            continue

        tile = tile_dir.name
        results[tile] = {}

        fm40_label = (
            TEMP_ROOT
            / tile
            / f"tilenum{tile}_fm40label_{year}.tif"
        )
        fm40_pred = (
            tile_dir
            / f"tilenum{tile}_aef_{year}_pred.tif"
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
            true_pt = tile_dir / f"LF_FBFM40_{category}.tif"
            pred_pt = tile_dir / f"predicted_FBFM40_{category}.tif"

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

def main():
    parser = argparse.ArgumentParser(
        description="Run full FM40 RF → fire behavior → evaluation pipeline"
    )
    parser.add_argument(
        "--tilenums",
        nargs="+",
        required=True,
        help="Tile numbers to process (e.g. 01180 00371)",
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

    model = load_rf_model()

    for tilenum in args.tilenums:
        run_rf_inference(model, tilenum, args.year)
        run_fire_behavior(tilenum, args.year)

    run_metrics(args.year, do_plot=args.plot)

if __name__ == "__main__":
    main()
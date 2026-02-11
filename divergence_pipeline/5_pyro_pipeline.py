import argparse
from pathlib import Path

import joblib
import numpy as np
import rasterio

from pyretechnics.load_landfire import (
    convert_rasters_to_space_time_cubes,
    load_and_convert_landfire_rasters,
)
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.burn_cells as bc


TEMP_ROOT = Path.cwd() / "temp"
PYROME_ROOT = Path.cwd() / "temp"
OUTPUT_ROOT = Path.cwd() / "outputs"
MODEL_ROOT = Path.cwd() / "data"

TEMP_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MERGED_MODEL_NAME = "rf_pyromes_merged.joblib"


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


def normalize_tile_id(tile_value):
    return str(tile_value).zfill(5)


def fuel_label_suffix(label_name: str) -> str:
    if label_name == "FBFM40":
        return "fm40"
    return "fm40parent"


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
    return tiles, pyrome_dir


def run_rf_inference(model, scaler, tile_dir: Path, tilenum, year, label_name):
    tif_files = [f for f in tile_dir.glob("*.tif") if "label" not in f.name]

    all_bands = []
    band_names = []
    profile = None
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            data = src.read()
            all_bands.append(data)
            layer_prefix = guess_layer_prefix(tif_file.name)
            band_names.extend(get_band_names(src, layer_prefix))
            if profile is None:
                profile = src.profile

    features = np.concatenate(all_bands, axis=0)
    expected_features = set(model.feature_names_in_)
    actual_features = set(band_names)
    common = expected_features & actual_features

    ordered_features = [f for f in model.feature_names_in_ if f in common]
    indices = [band_names.index(f) for f in ordered_features]
    features_filtered = features[indices, :, :]

    bands, h, w = features_filtered.shape
    data = np.transpose(features_filtered, (1, 2, 0)).reshape(-1, bands)
    if scaler is not None:
        data = scaler.transform(data)
    preds = model.predict(data).reshape(h, w)

    out_dir = OUTPUT_ROOT / tilenum
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = fuel_label_suffix(label_name)
    out_path = out_dir / f"tilenum{tilenum}_{suffix}_pred_{year}.tif"

    profile.update(count=1, dtype="uint8", compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(preds, 1)

    print(f"[RF] Wrote {out_path}")


def run_fire_behavior(tile_dir: Path, tilenum, year, label_name):
    out_dir = OUTPUT_ROOT / tilenum
    suffix = fuel_label_suffix(label_name)

    def f(k):
        return next(tile_dir.glob(f"*{k}*.tif"))

    lf_inputs = {
        "elevation": f("elevation"),
        "slope": f("slope"),
        "aspect": f("aspect"),
        "fuel_model": tile_dir / f"tilenum{tilenum}_{suffix}label_{year}.tif",
        "canopy_cover": f("cc"),
        "canopy_height": f("ch"),
        "canopy_base_height": f("cbh"),
        "canopy_bulk_density": f("cbd"),
    }

    pred_inputs = lf_inputs | {
        "fuel_model": out_dir / f"tilenum{tilenum}_{suffix}_pred_{year}.tif"
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

    wx = {
        k: SpaceTimeCube(cube_shape, v)
        for k, v in {
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
        }.items()
    }

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
        for data_label, arr in [("LF", lf_fire[k]), ("predicted", pred_fire[k])]:
            out_path = out_dir / f"{data_label}_FBFM40_{k}.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Run RF inference + pyretechnics outputs per tile"
    )
    parser.add_argument(
        "--pyromes",
        nargs="+",
        type=int,
        required=True,
        help="Pyrome IDs to process (reads temp/pyrome_{id}/tile_*/)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Use merged pyrome model (rf_pyromes_merged.joblib).",
    )
    parser.add_argument(
        "--tilenums",
        nargs="+",
        help="Optional tile IDs to process.",
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
        help="Fuel label mode to run against.",
    )

    args = parser.parse_args()

    for pyrome_id in args.pyromes:
        model_path = resolve_model_path(pyrome_id, args.merge)
        model, scaler = load_rf_model(model_path)

        tiles, pyrome_dir = list_pyrome_tiles(pyrome_id, tilenums=args.tilenums)
        if not tiles:
            print(f"No tiles found for pyrome {pyrome_id} in {pyrome_dir}")
            continue

        for tilenum, tile_dir in tiles.items():
            run_rf_inference(model, scaler, tile_dir, tilenum, args.year, args.label)
            run_fire_behavior(tile_dir, tilenum, args.year, args.label)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import numpy as np
import joblib
import rasterio
from rasterio.features import rasterize
from google.cloud import storage
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

TEMP_ROOT = Path.cwd() / "temp"
OUTPUT_ROOT = Path.cwd() / "outputs"

TEMP_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BUCKET_NAME = "geoai-fuels-tiles"

# From train_models.py
from_vals = [
    91,92,93,98,99,
    101,102,103,104,105,106,107,108,109,
    121,122,123,124,
    141,142,143,144,145,146,147,148,149,
    161,162,163,164,165,
    181,182,183,184,185,186,187,188,189,
    201,202,203,204
]

to_vals = [
    1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,
    3,3,3,3,
    4,4,4,4,4,4,4,4,4,
    5,5,5,5,5,
    6,6,6,6,6,6,6,6,6,
    7,7,7,7
]

parent_classes = np.unique(np.array(to_vals))


def single_band_name(name):
    def _resolver(count):
        if count == 1:
            return [name]
        return [f"{name}_{idx + 1}" for idx in range(count)]

    return _resolver


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

def load_rf_model(model_path):
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


def resolve_model_path(model_dir: Path, pyrome_id: int, merge: bool) -> Path:
    if merge:
        return model_dir / "rf_pyromes_merged.joblib"
    return model_dir / f"rf_pyrome_{pyrome_id}.joblib"

def download_tile_from_gcs(tilenum, year):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    tile_prefix = f"{year}/{tilenum}/"
    tile_dir = TEMP_ROOT / str(year) / tilenum
    tile_dir.mkdir(parents=True, exist_ok=True)

    blobs = list(bucket.list_blobs(prefix=tile_prefix))

    if not blobs:
        raise FileNotFoundError(
            f"No blobs found for tile {tilenum} in gs://{BUCKET_NAME}/{tile_prefix}"
        )

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        local_path = tile_dir / Path(blob.name).name
        if not local_path.exists():
            print(f"Downloading {blob.name}")
            blob.download_to_filename(local_path)

    return tile_dir

def compute_tile_accuracy(
    model,
    scaler,
    tile_dir: Path,
    tilenum,
    year,
    label_name,
    label_classes,
):
    tif_files = [f for f in tile_dir.glob("*.tif") if "label" not in f.name and "parentlabel" not in f.name]

    if not tif_files:
        print(f"WARN: No predictor TIFFs found in {tile_dir}; skipping tile {tilenum}.")
        return None

    print(f"Model expects {model.n_features_in_} features")
    print(f"Feature names: {model.feature_names_in_}")

    all_bands = []
    band_names = []
    profile = None
    predictor_nodata_masks = []
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            data = src.read()
            all_bands.append(data)
            # If band names are available
            layer_prefix = guess_layer_prefix(tif_file.name)
            band_names.extend(get_band_names(src, layer_prefix))
            if profile is None:
                profile = src.profile
            nodata = src.nodata
            if nodata is not None:
                nodata_mask = np.any(data == nodata, axis=0)
                predictor_nodata_masks.append(nodata_mask)

    features = np.concatenate(all_bands, axis=0)
    bands, h, w = features.shape
    print(f"Input has {bands} features")
    print(f"Band names: {band_names}")

    expected_features = list(model.feature_names_in_)
    actual_features = set(band_names)

    missing = [f for f in expected_features if f not in actual_features]
    extra = [f for f in band_names if f not in expected_features]

    if missing:
        raise ValueError(
            "Tile predictor bands do not match model features. "
            f"Missing: {missing} Extra: {extra}"
        )

    feature_map = {name: idx for idx, name in enumerate(band_names)}
    bands = len(expected_features)
    data = np.transpose(features, (1, 2, 0)).reshape(-1, features.shape[0])
    data_df = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=expected_features)

    for name in expected_features:
        idx = feature_map.get(name)
        if idx is not None:
            data_df[name] = data[:, idx]

    data_np = data_df.to_numpy()
    if np.isnan(data_np).any():
        nan_ratio = np.isnan(data_np).mean()
        print(f"WARN: Predictor NaN ratio for tile {tilenum}: {nan_ratio:.6f}")
        data_np = np.nan_to_num(data_np, nan=0.0)
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        means = scaler.mean_
        scales = scaler.scale_
        feature_means = data_np.mean(axis=0)
        feature_stds = data_np.std(axis=0)
        zscores = np.where(scales != 0, (feature_means - means) / scales, 0)
        extreme_mask = np.abs(zscores) > 5
        if np.any(extreme_mask):
            extreme_features = [
                (expected_features[i], zscores[i], feature_means[i], feature_stds[i])
                for i in np.where(extreme_mask)[0]
            ]
            print(
                f"WARN: {len(extreme_features)} features with |z|>5 for tile {tilenum}."
            )
            for name, z, mean_val, std_val in extreme_features[:10]:
                print(
                    f"  {name}: z={z:.2f}, mean={mean_val:.4f}, std={std_val:.4f}"
                )
            focus_prefixes = ("elevation", "climatenormals", "prism", "bps")
            focus_rows = []
            for idx in np.where(extreme_mask)[0]:
                feat_name = expected_features[idx]
                if not feat_name.startswith(focus_prefixes):
                    continue
                focus_rows.append(
                    (
                        feat_name,
                        means[idx],
                        scales[idx],
                        feature_means[idx],
                        feature_stds[idx],
                        zscores[idx],
                    )
                )
            if focus_rows:
                print("Outlier diagnostics (train_mean, train_std, tile_mean, tile_std, z):")
                for feat_name, t_mean, t_std, tile_mean, tile_std, z in focus_rows:
                    print(
                        f"  {feat_name}: train_mean={t_mean:.4f}, train_std={t_std:.4f}, "
                        f"tile_mean={tile_mean:.4f}, tile_std={tile_std:.4f}, z={z:.2f}"
                    )
        low_var_mask = feature_stds < 1e-6
        if np.any(low_var_mask):
            low_var_features = [expected_features[i] for i in np.where(low_var_mask)[0]]
            print(
                f"WARN: {len(low_var_features)} near-constant features for tile {tilenum}."
            )
            print("  e.g.", ", ".join(low_var_features[:10]))
    scaled_features = scaler.transform(data_np)
    scaled_df = pd.DataFrame(scaled_features, columns=expected_features)
    preds_encoded = model.predict(scaled_df)
    model_classes = np.array(model.classes_)

    # Load true labels
    if label_name == "FBFM40Parent":
        label_file = tile_dir / f"tilenum{tilenum}_fm40parentlabel_{year}.tif"
    else:
        label_file = tile_dir / f"tilenum{tilenum}_fm40label_{year}.tif"
    with rasterio.open(label_file) as src:
        label_data = src.read(1)
        true_labels = label_data.flatten()
        label_nodata = src.nodata

    # Filter out nodata or invalid pixels (assuming 0 or negative are invalid)
    valid_mask = true_labels > 0
    if label_nodata is not None:
        valid_mask &= true_labels != label_nodata
    if predictor_nodata_masks:
        combined_nodata = np.logical_or.reduce(predictor_nodata_masks)
        valid_mask &= ~combined_nodata.flatten()
    true_labels = true_labels[valid_mask]
    preds_encoded = preds_encoded[valid_mask]

    if len(true_labels) == 0:
        print(f"WARN: No valid parent labels for tile {tilenum}; skipping.")
        return None

    valid_mask = np.isin(true_labels, label_classes)
    true_labels = true_labels[valid_mask]
    preds_encoded = preds_encoded[valid_mask]
    if len(true_labels) == 0:
        print(f"WARN: No valid parent labels for tile {tilenum}; skipping.")
        return None

    true_encoded = true_labels

    # Compute metrics
    acc = accuracy_score(true_encoded, preds_encoded)
    f1 = f1_score(true_encoded, preds_encoded, average='weighted', zero_division=0)
    precision = precision_score(true_encoded, preds_encoded, average='weighted', zero_division=0)
    recall = recall_score(true_encoded, preds_encoded, average='weighted', zero_division=0)

    # Per-class metrics
    report = classification_report(
        true_encoded,
        preds_encoded,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(true_encoded, preds_encoded, labels=label_classes)
    cm_flat = cm.flatten().tolist()
    cm_labels = [f"cm_{t}_{p}" for t in label_classes for p in label_classes]

    true_counts = np.bincount(
        true_encoded.astype(int),
        minlength=int(np.max(label_classes)) + 1,
    )
    pred_counts = np.bincount(
        preds_encoded.astype(int),
        minlength=int(np.max(label_classes)) + 1,
    )
    zero_ratio = (data_np == 0).mean()
    print(f"Predictor zero ratio for tile {tilenum}: {zero_ratio:.6f}")
    rng = np.random.default_rng(1917)
    values, counts = np.unique(true_encoded, return_counts=True)
    probabilities = counts / counts.sum()
    random_preds = rng.choice(values, size=len(true_encoded), replace=True, p=probabilities)
    baseline_acc = accuracy_score(true_encoded, random_preds)
    baseline_f1 = f1_score(true_encoded, random_preds, average='weighted', zero_division=0)
    baseline_precision = precision_score(
        true_encoded, random_preds, average='weighted', zero_division=0
    )
    baseline_recall = recall_score(
        true_encoded, random_preds, average='weighted', zero_division=0
    )
    return {
        'tile': tilenum,
        'accuracy': acc,
        'f1_weighted': f1,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'baseline_accuracy': baseline_acc,
        'baseline_f1_weighted': baseline_f1,
        'baseline_precision_weighted': baseline_precision,
        'baseline_recall_weighted': baseline_recall,
        'num_samples': len(true_encoded),
        'model_classes': ",".join(str(val) for val in model_classes),
        'label_path': str(label_file),
        'label_counts': ",".join(str(true_counts[c]) for c in label_classes),
        'pred_counts': ",".join(str(pred_counts[c]) for c in label_classes),
        **dict(zip(cm_labels, cm_flat)),
        **{
            f"class_{k}_{m}": v
            for k, metrics in report.items()
            if isinstance(metrics, dict)
            for m, v in metrics.items()
            if k != 'weighted avg' and k != 'macro avg'
        }
    }


def list_pyrome_tiles(pyrome_dir: Path) -> list[tuple[str, str]]:
    if not pyrome_dir.exists():
        raise FileNotFoundError(f"Pyrome temp folder not found: {pyrome_dir}")
    tiles = []
    for path in sorted(p for p in pyrome_dir.iterdir() if p.is_dir()):
        folder = path.name
        if not folder.startswith("tile_"):
            continue
        tilenum = folder.replace("tile_", "", 1) if folder.startswith("tile_") else folder
        tiles.append((folder, tilenum))
    return tiles

def main():
    parser = argparse.ArgumentParser(
        description="Compute FBFM40 accuracy metrics per tile using trained RF model"
    )
    parser.add_argument(
        "--pyromes",
        nargs="+",
        type=int,
        required=True,
        help="Pyrome IDs to process",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Use merged pyrome model (rf_pyromes_merged.joblib).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing trained model joblib files",
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("temp"),
        help="Root temp directory containing pyrome folders",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Year used in filenames (default: 2022)",
    )
    parser.add_argument(
        "--label",
        default="FBFM40Parent",
        help="Label column name to evaluate against (FBFM40Parent or FBFM40).",
    )

    args = parser.parse_args()

    label_name = args.label
    if label_name == "FBFM40Parent":
        label_classes = parent_classes
    else:
        label_classes = np.array(from_vals)

    results = []
    for pyrome_id in args.pyromes:
        model_path = resolve_model_path(args.model_dir, pyrome_id, args.merge)
        model, scaler = load_rf_model(model_path)
        pyrome_dir = args.temp_root / f"pyrome_{pyrome_id}"
        tiles = list_pyrome_tiles(pyrome_dir)
        if not tiles:
            print(f"No tiles found for pyrome {pyrome_id} in {pyrome_dir}")
            continue
        for folder_name, tilenum in tiles:
            print(f"\n=== Processing Pyrome {pyrome_id} Tile {tilenum} ===")
            tile_dir = pyrome_dir / folder_name
            metrics = compute_tile_accuracy(
                model,
                scaler,
                tile_dir,
                tilenum,
                args.year,
                label_name,
                label_classes,
            )
            if metrics is None:
                continue
            metrics["pyrome_id"] = pyrome_id
            results.append(metrics)
            print(
                f"Metrics for tile {tilenum}: Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1_weighted']:.4f}"
            )

    df = pd.DataFrame(results)
    if args.merge:
        pyrome_tag = "merged"
    else:
        pyrome_tag = "_".join(str(pid) for pid in args.pyromes)
    csv_path = OUTPUT_ROOT / f"tile_fbfm40_accuracy_metrics_pyrome_{pyrome_tag}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved metrics to {csv_path}")
    print(df)

if __name__ == "__main__":
    main()

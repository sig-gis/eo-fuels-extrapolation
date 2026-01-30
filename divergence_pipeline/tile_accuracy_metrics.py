import argparse
from pathlib import Path
import numpy as np
import joblib
import rasterio
from google.cloud import storage
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

TEMP_ROOT = Path.cwd() / "temp" / "geoai-fuels-tiles"
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

parent_classes = np.unique(np.array(from_vals))
encoder = LabelEncoder().fit(parent_classes)

def load_rf_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    print("Loaded RF model:", type(model))
    return model

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

def compute_tile_accuracy(model, tilenum, year):
    tile_dir = TEMP_ROOT / str(year) / tilenum
    tif_files = [f for f in tile_dir.glob("*.tif") if "label" not in f.name and "parentlabel" not in f.name]

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
            if src.descriptions:
                band_names.extend(src.descriptions)
            else:
                band_names.extend([f"{tif_file.stem}_band_{i}" for i in range(data.shape[0])])
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
    preds_encoded = model.predict(data)

    # Load true labels
    label_file = tile_dir / f"tilenum{tilenum}_fm40label_{year}.tif"
    with rasterio.open(label_file) as src:
        true_labels = src.read(1).flatten()

    # Filter out nodata or invalid pixels (assuming 0 or negative are invalid)
    valid_mask = true_labels > 0
    true_labels = true_labels[valid_mask]
    preds_encoded = preds_encoded[valid_mask]

    # Encode true labels
    try:
        true_encoded = encoder.transform(true_labels)
    except ValueError as e:
        print(f"Error encoding true labels: {e}")
        # Perhaps some labels are not in from_vals
        # Filter to only known classes
        known_mask = np.isin(true_labels, parent_classes)
        true_labels = true_labels[known_mask]
        preds_encoded = preds_encoded[known_mask]
        true_encoded = encoder.transform(true_labels)

    # Compute metrics
    acc = accuracy_score(true_encoded, preds_encoded)
    f1 = f1_score(true_encoded, preds_encoded, average='weighted')
    precision = precision_score(true_encoded, preds_encoded, average='weighted')
    recall = recall_score(true_encoded, preds_encoded, average='weighted')

    # Per-class metrics
    report = classification_report(true_encoded, preds_encoded, output_dict=True)

    return {
        'tile': tilenum,
        'accuracy': acc,
        'f1_weighted': f1,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'num_samples': len(true_encoded),
        **{f"class_{k}_{m}": v for k, metrics in report.items() if isinstance(metrics, dict) for m, v in metrics.items() if k != 'weighted avg' and k != 'macro avg'}
    }

def main():
    parser = argparse.ArgumentParser(
        description="Compute FBFM40 accuracy metrics per tile using trained RF model"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("temp/geoai-fuels-tiles/utils/rf_zone_27.joblib"),
        help="Local path to trained model joblib file",
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

    args = parser.parse_args()

    model = load_rf_model(args.model_path)

    results = []
    for tilenum in args.tilenums:
        print(f"\n=== Processing Tile {tilenum} ===")
        download_tile_from_gcs(tilenum, args.year)
        metrics = compute_tile_accuracy(model, tilenum, args.year)
        results.append(metrics)
        print(f"Metrics for tile {tilenum}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_weighted']:.4f}")

    df = pd.DataFrame(results)
    csv_path = OUTPUT_ROOT / "tile_fbfm40_accuracy_metrics.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved metrics to {csv_path}")
    print(df)

if __name__ == "__main__":
    main()

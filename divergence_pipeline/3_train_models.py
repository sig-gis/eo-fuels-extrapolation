import argparse
import joblib
from pathlib import Path

import json
from yaml import load,dump

import numpy as np
import pandas as pd
import geopandas as gpd

from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import rasterio as rio

FROM_VALS = [
    91, 92, 93, 98, 99,
    101, 102, 103, 104, 105, 106, 107, 108, 109,
    121, 122, 123, 124,
    141, 142, 143, 144, 145, 146, 147, 148, 149,
    161, 162, 163, 164, 165,
    181, 182, 183, 184, 185, 186, 187, 188, 189,
    201, 202, 203, 204,
]

TO_VALS = [
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7,
]

PARENT_CLASSES = np.unique(np.array(TO_VALS))
SEED = 1917


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF model from a single CSV or merged pyrome samples."
    )
    parser.add_argument(
        "--csv",
        help="Path to a single CSV to train from.",
    )
    parser.add_argument(
        "--pyromes",
        nargs="+",
        type=int,
        help="Pyrome IDs to merge from samples directory.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Sample year for pyrome CSVs.",
    )
    parser.add_argument(
        "--mode",
        default="equal",
        help="Sample mode used in filename (e.g., equal or dist).",
    )
    parser.add_argument(
        "--samples-dir",
        default="data/samples",
        help="Directory containing pyrome sample CSVs.",
    )
    parser.add_argument(
        "--bucket",
        default="geoai-fuels-tiles",
        help="GCS bucket containing pyrome sample CSVs.",
    )
    parser.add_argument(
        "--gcs-prefix",
        default="samples",
        help="GCS prefix for pyrome CSVs (default: samples).",
    )
    parser.add_argument(
        "--combined-out",
        help="Optional output path to write merged CSV.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge pyrome CSVs and train a single model instead of per-pyrome.",
    )
    parser.add_argument(
        "--train-pyromes",
        nargs="+",
        type=int,
        help="Subset of pyrome IDs to train models for (defaults to provided pyromes).",
    )
    parser.add_argument(
        "--label",
        choices=["FBFM40", "FBFM40Parent"],
        default="FBFM40",
        help="Label column to train on.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write trained model files.",
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/lr.yml',
        help='yml file cointaining model params & configurations'
    )

    return parser.parse_args()

def load_config(yml_path):
    with open(yml_path) as f:
        config = load(f)
    return config

def build_model(config):
    model_params = config['params']
    if config['name'] == 'RandomForest':
        model = RandomForestClassifier(**model_params)
    elif config['name'] == 'LogisticRegression':
        model = LogisticRegression(**model_params)
    elif config['name'] == 'KNeighborsClassifier':
        model = KNeighborsClassifier(**model_params)

    return model 

def load_support_rasters(data_config):
    evt_data_desc = pd.read_csv(data_config['evt']['data_dict'])
    evc_data_desc = pd.read_csv(data_config['evc']['data_dict']).rename(columns={'CLASSNAMES':'EVC_CLASS'})
    evh_data_desc = pd.read_csv(data_config['evh']['data_dict']).rename(columns={'CLASSNAMES':'EVH_CLASS'})
    bps_data_desc = pd.read_csv(data_config['bps']['data_dict']).rename(columns={'GROUPVEG':'BPS_CLASS'})

    evt_raster = rio.open(data_config['evt']['raster'])
    evc_raster = rio.open(data_config['evc']['raster'])
    evh_raster = rio.open(data_config['evh']['raster'])
    bps_raster = rio.open(data_config['bps']['raster'])
    nlcd_raster = rio.open(data_config['nlcd']['raster'])

    support_rasters = {
        'evc':{'data_dict':evc_data_desc,'raster':evc_raster},
        'evt':{'data_dict':evt_data_desc,'raster':evt_raster},
        'evh':{'data_dict':evh_data_desc,'raster':evh_raster},
        'bps':{'data_dict':bps_data_desc,'raster':bps_raster},
        'nlcd':{'raster':nlcd_raster}
    }

    return support_rasters

def append_locations(df):
    fuels_sample_loc = df.copy()

    geo = df['.geo']
    fuels_sample_loc['longitude'], fuels_sample_loc['latitude'] = geo.map(lambda x: list(json.loads(x)['coordinates'])[0]), geo.map(lambda x: list(json.loads(x)['coordinates'])[1])

    fuels_sample_loc[['latitude','longitude']]
    fuels_sample_loc.drop(columns=['system:index','.geo'],inplace=True)

    fuels_sample_gdf = gpd.GeoDataFrame(fuels_sample_loc,geometry=gpd.points_from_xy(fuels_sample_loc.longitude,fuels_sample_loc.latitude),crs='EPSG:4326').to_crs('EPSG:5070')

    return fuels_sample_gdf

def append_veg_features(gdf,support_rasters):
    def assign_tree_cover(record):
        if (record['evc'] > 100) and (record['evc'] < 200):
            tree_cover = (record['evc'] % 100) / 100.0
        else:
            tree_cover = 0
        return tree_cover
    def assign_shrub_cover(record):
        if (record['evc'] > 200) and (record['evc'] < 300):
                shrub_cover = (record['evc'] % 100) / 100.0
        else:
            shrub_cover = 0
        return shrub_cover
    def assign_herb_cover(record):
        if (record['evc'] > 300) and (record['evc'] < 400):
                herb_cover = (record['evc'] % 100) / 100.0
        else:
            herb_cover = 0
        return herb_cover
    def assign_tree_height(record):
        if (record['evh'] > 100) and (record['evh'] < 200):
            tree_height = (record['evh'] % 100)
        else:
            tree_height = 0
        return tree_height
    def assign_shrub_height(record):
        if (record['evh'] > 200) and (record['evh'] < 300):
                shrub_height = (record['evh'] % 100) / 10.0
        else:
            shrub_height = 0
        return shrub_height
    def assign_herb_height(record):
        if (record['evh'] > 300) and (record['evh'] < 400):
                herb_height = (record['evh'] % 100) / 10.0
        else:
            herb_height = 0
        return herb_height
    
    coord_list = [(x,y) for x,y in zip(gdf['geometry'].x,gdf['geometry'].y)]
    for key in support_rasters.keys():
        gdf[key] = [x[0] for x in support_rasters[key]['raster'].sample(coord_list)]

    gdf = gdf.merge(support_rasters['data_dict'][['LFRDB','EVT_NAME','EVT_ORDER','EVT_CLASS','EVT_SBCLS']],left_on='evt',right_on='LFRDB')
    gdf = gdf.merge(support_rasters['data_dict'][['VALUE','EVC_CLASS']],left_on='evc',right_on='VALUE')
    gdf = gdf.merge(support_rasters['data_dict'][['VALUE','EVH_CLASS']],left_on='evh',right_on='VALUE')
    gdf = gdf.merge(support_rasters['data_dict'][['VALUE','BPS_NAME','BPS_CLASS']],left_on='bps',right_on='VALUE')

    gdf['tree_height'] = gdf.apply(assign_tree_height,axis=1)
    gdf['shrub_height'] = gdf.apply(assign_shrub_height,axis=1)
    gdf['herb_height'] = gdf.apply(assign_herb_height,axis=1)

    gdf['tree_cover'] = gdf.apply(assign_tree_cover,axis=1)
    gdf['shrub_cover'] = gdf.apply(assign_shrub_cover,axis=1)
    gdf['herb_cover'] = gdf.apply(assign_herb_cover,axis=1)


    return gdf

def pyrome_csv_filename(pyrome_id: int, year: int, mode: str) -> str:
    return f"pyrome_{pyrome_id}_{year}_{mode}_local_samples.csv"


def pyrome_csv_path(samples_dir: Path, pyrome_id: int, year: int, mode: str) -> Path:
    filename = f"samples_pyrome_{pyrome_id}_{pyrome_csv_filename(pyrome_id, year, mode)}"
    return samples_dir / filename


def download_pyrome_csv(
    samples_dir: Path,
    pyrome_id: int,
    year: int,
    mode: str,
    bucket_name: str,
    gcs_prefix: str,
) -> Path:
    local_path = pyrome_csv_path(samples_dir, pyrome_id, year, mode)
    if local_path.exists():
        return local_path

    samples_dir.mkdir(parents=True, exist_ok=True)

    gcs_filename = pyrome_csv_filename(pyrome_id, year, mode)
    blob_name = f"{gcs_prefix}/pyrome_{pyrome_id}/{gcs_filename}".lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(
            f"CSV not found at gs://{bucket_name}/{blob_name}"
        )

    print(f"Downloading gs://{bucket_name}/{blob_name}")
    blob.download_to_filename(local_path)
    return local_path


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def merge_pyrome_csvs(
    pyromes,
    year,
    mode,
    samples_dir: Path,
    bucket_name: str,
    gcs_prefix: str,
) -> pd.DataFrame:
    frames = []
    missing = []
    for pyrome_id in pyromes:
        try:
            csv_path = download_pyrome_csv(
                samples_dir,
                pyrome_id,
                year,
                mode,
                bucket_name,
                gcs_prefix,
            )
        except FileNotFoundError:
            csv_path = pyrome_csv_path(samples_dir, pyrome_id, year, mode)
            missing.append(str(csv_path))
            continue
        frame = pd.read_csv(csv_path)
        if "pyrome_id" not in frame.columns:
            frame["pyrome_id"] = pyrome_id
        frames.append(frame)

    if missing:
        missing_str = "\n".join(missing)
        raise FileNotFoundError(
            f"Missing pyrome CSVs:\n{missing_str}"
        )

    if not frames:
        raise RuntimeError("No pyrome CSVs loaded.")
    return pd.concat(frames, ignore_index=True)


def get_feature_list(df: pd.DataFrame):
    feature_list = df.columns.to_list()
    for col in ["system:index", ".geo", "x", "y", "pyrome_id", "cc", "ch", "cbh", "cbd"]:
        if col in feature_list:
            feature_list.remove(col)

    fm40_like = [
        col for col in feature_list
        if "fm40" in col.lower() and col != "FBFM40Parent"
    ]
    for col in fm40_like:
        feature_list.remove(col)

    alphaearth_features = [f"A{str(i).zfill(2)}" for i in range(64)]
    if not all(feature in feature_list for feature in alphaearth_features):
        aef_features = [f"aef_b{idx + 1}" for idx in range(64)]
        if all(feature in feature_list for feature in aef_features):
            alphaearth_features = aef_features
    label_list = ["FBFM40", "FBFM40Parent"]
    feature_list_wo_alphaearth = [
        feature for feature in feature_list
        if feature not in (alphaearth_features + label_list)
    ]
    train_features = alphaearth_features + feature_list_wo_alphaearth
    train_features = [f for f in train_features if f not in ["ESP"]]
    return train_features




def remap_parent_labels(series: pd.Series) -> pd.Series:
    normalized = pd.to_numeric(series, errors="coerce").round().astype("Int64")
    mapping = dict(zip(FROM_VALS, TO_VALS))
    remapped = normalized.map(mapping)
    remapped = remapped.fillna(normalized)
    return remapped.astype("Int64")


def resolve_label_column(df: pd.DataFrame, label_name: str) -> str:
    """
    Ensure the requested label exists in df, creating it from compatible sources if needed.

    - FBFM40: prefer FBFM40, fallback to fm40label_b1.
    - FBFM40Parent: prefer FBFM40Parent, fallback to remap(FBFM40 or fm40label_b1).
    """
    available = ", ".join(df.columns)

    if label_name == "FBFM40":
        if "FBFM40" in df.columns:
            source = "FBFM40"
        elif "fm40label_b1" in df.columns:
            source = "fm40label_b1"
            df["FBFM40"] = df[source]
        else:
            raise ValueError(
                "Requested label 'FBFM40' not found. Expected 'FBFM40' or fallback "
                f"'fm40label_b1'. Available columns: {available}"
            )

        df["FBFM40"] = pd.to_numeric(df[source], errors="coerce").round().astype("Int64")
        if df["FBFM40"].isna().any():
            raise ValueError("FBFM40 contains non-numeric or missing values after normalization.")
        return "FBFM40"

    if "FBFM40Parent" in df.columns:
        df["FBFM40Parent"] = remap_parent_labels(df["FBFM40Parent"])
        if df["FBFM40Parent"].isna().any():
            raise ValueError(
                "Some FBFM40Parent labels could not be normalized/remapped. "
                "Check input classes."
            )
        return "FBFM40Parent"

    if "FBFM40" in df.columns:
        parent_source = "FBFM40"
    elif "fm40label_b1" in df.columns:
        parent_source = "fm40label_b1"
    else:
        raise ValueError(
            "Requested label 'FBFM40Parent' not found and no source to derive it. "
            "Expected one of: FBFM40Parent, FBFM40, fm40label_b1. "
            f"Available columns: {available}"
        )

    df["FBFM40Parent"] = remap_parent_labels(df[parent_source])
    if df["FBFM40Parent"].isna().any():
        raise ValueError(
            "Some parent labels could not be derived from original FBFM40 values. "
            "Check input classes."
        )
    return "FBFM40Parent"


def evaluate_metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def sample_random_labels(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values, counts = np.unique(y_true, return_counts=True)
    probabilities = counts / counts.sum()
    return rng.choice(values, size=len(y_true), replace=True, p=probabilities)


def train_model(df, train_features, label, test_size, model_config, output_path):
    X = np.nan_to_num(df[train_features].to_numpy(), 0)
    y = df[label].to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_encode = y_train

    clf = build_model(model_config)

    clf.fit(X_train_scaled, y_train_encode)
    clf.feature_names_in_ = np.array(train_features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    scaler_path = output_path.with_name(f"{output_path.stem}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved {output_path}")
    print(f"Saved {scaler_path}")

    X_test_scaled = scaler.transform(X_test)
    X_test_df = pd.DataFrame(X_test_scaled, columns=train_features)
    y_pred = clf.predict(X_test_df)
    model_metrics = evaluate_metrics(y_test, y_pred, "model")

    rng = np.random.default_rng(SEED)
    random_preds = sample_random_labels(y_test, rng)
    baseline_metrics = evaluate_metrics(y_test, random_preds, "baseline")

    metrics = {**model_metrics, **baseline_metrics}
    metrics["num_samples"] = len(y_test)
    return metrics


def main():
    args = parse_args()

    if args.config:
        config = load_config(args.config)

    if not args.csv and not args.pyromes:
        raise ValueError("Provide --csv or --pyromes")
    if args.csv and args.pyromes:
        raise ValueError("Use either --csv or --pyromes, not both")

    if args.csv:
        df = load_csv(Path(args.csv))
    else:
        samples_dir = Path(args.samples_dir)
        df = merge_pyrome_csvs(
            args.pyromes,
            args.year,
            args.mode,
            samples_dir,
            args.bucket,
            args.gcs_prefix,
        )

        if args.combined_out:
            combined_path = Path(args.combined_out)
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(combined_path, index=False)
            print(f"Wrote combined CSV to {combined_path}")

    label_name = resolve_label_column(df, args.label)

    train_features = get_feature_list(df)

    output_dir = Path(args.output_dir)

    metrics_rows = []

    if args.pyromes and not args.csv:
        if args.merge:
            output_path = output_dir / f"{config['exp_name']}_pyromes_merged.joblib"
            metrics = train_model(
                df,
                train_features,
                label=label_name,
                test_size=args.test_size,
                config = config['model'],
                output_path=output_path,
            )
            metrics_rows.append({"pyrome_id": "merged", **metrics})
        else:
            zones_to_train = args.train_pyromes or args.pyromes
            for zone in zones_to_train:
                print(f"Training model for pyrome {zone}")
                zone_df = df if len(zones_to_train) == 1 else df[df["pyrome_id"] == zone]
                if zone_df.empty:
                    raise RuntimeError(f"No data for pyrome {zone}")
                output_path = output_dir / f"rf_pyrome_{zone}.joblib"
                metrics = train_model(
                    zone_df,
                    train_features,
                    label=label_name,
                    test_size=args.test_size,
                    n_estimators=args.n_estimators,
                    output_path=output_path,
                )
                metrics_rows.append({"pyrome_id": zone, **metrics})
    else:
        output_path = output_dir / "rf_model.joblib"
        metrics = train_model(
            df,
            train_features,
            label=label_name,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            output_path=output_path,
        )
        metrics_rows.append({"pyrome_id": "all", **metrics})

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_out = output_dir / "rf_training_accuracy_baseline.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Saved training metrics to {metrics_out}")


if __name__ == "__main__":
    main()
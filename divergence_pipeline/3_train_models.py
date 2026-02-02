import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib


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
        "--combined-out",
        help="Optional output path to write merged CSV.",
    )
    parser.add_argument(
        "--train-pyromes",
        nargs="+",
        type=int,
        help="Subset of pyrome IDs to train models for (defaults to provided pyromes).",
    )
    parser.add_argument(
        "--label",
        default="FBFM40Parent",
        help="Label column to train on.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="RandomForest number of estimators.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write trained model files.",
    )
    return parser.parse_args()


def pyrome_csv_path(samples_dir: Path, pyrome_id: int, year: int, mode: str) -> Path:
    filename = f"pyrome_{pyrome_id}_{year}_{mode}_local_samples.csv"
    return samples_dir / filename


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def merge_pyrome_csvs(pyromes, year, mode, samples_dir: Path) -> pd.DataFrame:
    frames = []
    missing = []
    for pyrome_id in pyromes:
        csv_path = pyrome_csv_path(samples_dir, pyrome_id, year, mode)
        if not csv_path.exists():
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
    for col in ["system:index", ".geo"]:
        if col in feature_list:
            feature_list.remove(col)

    alphaearth_features = [f"A{str(i).zfill(2)}" for i in range(64)]
    label_list = ["FBFM40", "FBFM40Parent"]
    feature_list_wo_alphaearth = [
        feature for feature in feature_list
        if feature not in (alphaearth_features + label_list)
    ]
    train_features = alphaearth_features + feature_list_wo_alphaearth
    train_features = [f for f in train_features if f not in ["ESP"]]
    return train_features


def remap_parent_labels(series: pd.Series) -> pd.Series:
    mapping = dict(zip(FROM_VALS, TO_VALS))
    remapped = series.map(mapping)
    remapped = remapped.fillna(series)
    return remapped.astype("Int64")


def train_model(df, train_features, label, test_size, n_estimators, output_path):
    X = np.nan_to_num(df[train_features].to_numpy(), 0)
    y = df[label].to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )

    scaler = StandardScaler()
    encoder = LabelEncoder().fit(PARENT_CLASSES)

    X_train_scaled = scaler.fit_transform(X_train)
    y_train_encode = encoder.transform(y_train)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        max_features="sqrt",
        n_jobs=-1,
    )

    rf.fit(X_train_scaled, y_train_encode)
    rf.feature_names_in_ = np.array(train_features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, output_path)
    print(f"Saved {output_path}")


def main():
    args = parse_args()

    if not args.csv and not args.pyromes:
        raise ValueError("Provide --csv or --pyromes")
    if args.csv and args.pyromes:
        raise ValueError("Use either --csv or --pyromes, not both")

    if args.csv:
        df = load_csv(Path(args.csv))
    else:
        samples_dir = Path(args.samples_dir)
        df = merge_pyrome_csvs(args.pyromes, args.year, args.mode, samples_dir)

        if args.combined_out:
            combined_path = Path(args.combined_out)
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(combined_path, index=False)
            print(f"Wrote combined CSV to {combined_path}")

    if args.label != "FBFM40Parent":
        raise ValueError("This script is configured to train on FBFM40Parent")

    df["FBFM40Parent"] = remap_parent_labels(df["FBFM40Parent"])
    if df["FBFM40Parent"].isna().any():
        raise ValueError(
            "Some labels could not be remapped. Check input classes."
        )

    train_features = get_feature_list(df)

    output_dir = Path(args.output_dir)

    if args.pyromes and not args.csv:
        zones_to_train = args.train_pyromes or args.pyromes
        for zone in zones_to_train:
            print(f"Training model for pyrome {zone}")
            zone_df = df if len(zones_to_train) == 1 else df[df["pyrome_id"] == zone]
            if zone_df.empty:
                raise RuntimeError(f"No data for pyrome {zone}")
            output_path = output_dir / f"rf_pyrome_{zone}.joblib"
            train_model(
                zone_df,
                train_features,
                label="FBFM40Parent",
                test_size=args.test_size,
                n_estimators=args.n_estimators,
                output_path=output_path,
            )
    else:
        output_path = output_dir / "rf_model.joblib"
        train_model(
            df,
            train_features,
            label="FBFM40Parent",
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from google.cloud import storage
from rasterio.features import rasterize

GPKG_BUCKET = "geoai-fuels-tiles"
GPKG_BLOB = r"utils/tiles48km_polygons_firefactor.gpkg"

LAYER_SPECS = [
    ("aef", "tilenum{tilenum}_aef_{year}", True, False),
    ("fm40label", "tilenum{tilenum}_fm40label_{year}", True, False),
    ("fm40parentlabel", "tilenum{tilenum}_fm40parentlabel_{year}", False, True),
    ("ch", "tilenum{tilenum}_ch_{year}", True, False),
    ("cc", "tilenum{tilenum}_cc_{year}", True, False),
    ("cbh", "tilenum{tilenum}_cbh_{year}", True, False),
    ("cbd", "tilenum{tilenum}_cbd_{year}", True, False),
    ("elevation", "tilenum{tilenum}_elevation", True, False),
    ("slope", "tilenum{tilenum}_slope", True, False),
    ("aspect", "tilenum{tilenum}_aspect", True, False),
    ("mtpi", "tilenum{tilenum}_mtpi", True, False),
    ("spectral_stats", "tilenum{tilenum}_spectral_stats_{year}", True, False),
    ("hls_band_stats", "tilenum{tilenum}_hls_band_stats_{year}", True, False),
    ("climatenormals", "tilenum{tilenum}_climatenormals_{year}", True, False),
    ("prism", "tilenum{tilenum}_prism_{year}", True, False),
    ("evc", "tilenum{tilenum}_evc_{year}", True, False),
    ("evt", "tilenum{tilenum}_evt_{year}", True, False),
    ("evh", "tilenum{tilenum}_evh_{year}", True, False),
    ("bps", "tilenum{tilenum}_bps_{year}", True, False),
]


def download_gpkg_if_needed(client, local_path):
    if os.path.exists(local_path):
        return

    bucket = client.bucket(GPKG_BUCKET)
    blob = bucket.blob(GPKG_BLOB)

    if not blob.exists():
        raise FileNotFoundError(
            f"GPKG not found at gs://{GPKG_BUCKET}/{GPKG_BLOB}"
        )

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading gs://{GPKG_BUCKET}/{GPKG_BLOB}")
    blob.download_to_filename(local_path)


def get_pyrome_gdf(pyrome_ids, crs):
    ee.Initialize()
    pyromes = ee.FeatureCollection(
        "projects/pyregence-ee/assets/Pyromes_CONUS_20200206"
    ).filter(ee.Filter.inList("PYROME", pyrome_ids))
    data = pyromes.getInfo()
    gdf = gpd.GeoDataFrame.from_features(data["features"])
    gdf.set_crs("EPSG:4326", inplace=True)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return gdf


def select_tiles_for_pyromes(tiles_gdf, pyrome_gdf):
    tile_map = {}
    for _, row in pyrome_gdf.iterrows():
        pyrome_id = int(row["PYROME"])
        geom = row.geometry
        tile_map[pyrome_id] = tiles_gdf[tiles_gdf.intersects(geom)]["tilenum"].tolist()
    return tile_map


def download_tile_layers(bucket, year, tilenum, pyrome_id, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_paths = {}
    for layer_name, pattern, _, _ in LAYER_SPECS:
        blob_name = (
            f"pyrome_{pyrome_id}/{tilenum}/"
            f"{pattern.format(tilenum=tilenum, year=year)}.tif"
        )
        blob = bucket.blob(blob_name)
        if not blob.exists():
            print(f"WARN: Missing {blob_name}")
            continue
        local_path = output_dir / f"{pattern.format(tilenum=tilenum, year=year)}.tif"
        if not local_path.exists():
            print(f"Downloading gs://{bucket.name}/{blob_name}")
            blob.download_to_filename(local_path)
        layer_paths.setdefault(layer_name, []).append(local_path)
    return layer_paths


def build_vrt(layer_name, tif_paths, output_dir):
    if not tif_paths:
        raise RuntimeError(f"No inputs for layer {layer_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    list_path = output_dir / f"{layer_name}_inputs.txt"
    with list_path.open("w") as handle:
        for tif in tif_paths:
            handle.write(f"{tif}\n")
    vrt_path = output_dir / f"{layer_name}.vrt"
    subprocess.run(
        ["gdalbuildvrt", "-input_file_list", str(list_path), str(vrt_path)],
        check=True,
    )
    return vrt_path


def get_band_names(dataset, prefix):
    descriptions = list(dataset.descriptions) if dataset.descriptions else []
    names = []
    for idx in range(dataset.count):
        desc = descriptions[idx] if idx < len(descriptions) else None
        names.append(desc if desc else f"{prefix}_b{idx + 1}")
    return names


def compute_class_points(counts, points_per_class, mode):
    class_values = sorted(counts.keys())
    if mode == "dist":
        total_points = points_per_class * 7
        total_count = sum(counts.values())
        class_points = {
            cls: max(1, int(round((counts[cls] / total_count) * total_points)))
            for cls in class_values
        }
    else:
        class_points = {cls: points_per_class for cls in class_values}
    return class_points


def window_mask(geom, dataset, window):
    transform = dataset.window_transform(window)
    mask = rasterize(
        [geom],
        out_shape=(window.height, window.width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    )
    return mask


def stratified_sample(label_dataset, geom, mode, points_per_class, seed):
    rng = np.random.default_rng(seed)
    counts = {}
    nodata = label_dataset.nodata

    for _, window in label_dataset.block_windows(1):
        labels = label_dataset.read(1, window=window)
        mask = window_mask(geom, label_dataset, window)
        valid = mask == 1
        if nodata is not None:
            valid &= labels != nodata
        if not np.any(valid):
            continue
        vals = labels[valid].astype(int)
        unique, cts = np.unique(vals, return_counts=True)
        for val, count in zip(unique, cts):
            counts[int(val)] = counts.get(int(val), 0) + int(count)

    if not counts:
        return [], {}

    class_points = compute_class_points(counts, points_per_class, mode)
    remaining = class_points.copy()
    samples = []

    for _, window in label_dataset.block_windows(1):
        if all(value <= 0 for value in remaining.values()):
            break
        labels = label_dataset.read(1, window=window)
        mask = window_mask(geom, label_dataset, window)
        valid = mask == 1
        if nodata is not None:
            valid &= labels != nodata
        if not np.any(valid):
            continue

        for cls, need in list(remaining.items()):
            if need <= 0:
                continue
            cls_mask = valid & (labels == cls)
            if not np.any(cls_mask):
                continue
            rows, cols = np.where(cls_mask)
            take = min(need, len(rows))
            pick_idx = rng.choice(len(rows), size=take, replace=False)
            rows = rows[pick_idx]
            cols = cols[pick_idx]
            for r, c in zip(rows, cols):
                x, y = rasterio.transform.xy(
                    label_dataset.window_transform(window), r, c, offset="center"
                )
                samples.append((x, y, cls))
            remaining[cls] -= take

    return samples, class_points


def sample_predictors(predictor_datasets, points):
    coords = [(x, y) for x, y, _ in points]
    data = {}
    for prefix, dataset in predictor_datasets:
        band_names = get_band_names(dataset, prefix)
        values = list(dataset.sample(coords))
        values = np.array(values)
        for idx, name in enumerate(band_names):
            data[name] = values[:, idx]
    return data


def upload_csv(bucket, local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded gs://{bucket.name}/{gcs_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download tile GeoTIFFs, mosaic per pyrome, and sample locally."
    )
    parser.add_argument("--pyromes", nargs="+", required=True, help="Pyrome IDs")
    parser.add_argument("--year", type=int, default=2022, help="Data year")
    parser.add_argument("--bucket", default="geoai-fuels-tiles", help="GCS bucket")
    parser.add_argument("--mode", choices=["dist", "equal"], default="equal")
    parser.add_argument("--points-per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument(
        "--temp-dir",
        default=r"C:\Users\edalt\RD_Fuels\eo-fuels-extrapolation\temp",
        help="Temporary working directory (deleted after completion)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete temp folder after completion",
    )
    parser.add_argument(
        "--gpkg-path",
        default=r"C:\Users\edalt\RD_Fuels\eo-fuels-extrapolation\temp\tiles48km_polygons_firefactor.gpkg",
        help="Local path to tiles GPKG",
    )

    args = parser.parse_args()

    gcs_client = storage.Client()
    bucket = gcs_client.bucket(args.bucket)

    download_gpkg_if_needed(gcs_client, args.gpkg_path)
    tiles = gpd.read_file(args.gpkg_path)

    pyrome_ids = [int(pid) for pid in args.pyromes]
    pyrome_gdf = get_pyrome_gdf(pyrome_ids, tiles.crs)
    tile_map = select_tiles_for_pyromes(tiles, pyrome_gdf)

    base_temp = Path(args.temp_dir)
    base_temp.mkdir(parents=True, exist_ok=True)

    for _, row in pyrome_gdf.iterrows():
        pyrome_id = int(row["PYROME"])
        geom = row.geometry
        tilenums = tile_map.get(pyrome_id, [])
        if not tilenums:
            print(f"No tiles intersect pyrome {pyrome_id}")
            continue

        print(f"Pyrome {pyrome_id}: {len(tilenums)} tiles")
        pyrome_temp = base_temp / f"pyrome_{pyrome_id}"
        if pyrome_temp.exists():
            shutil.rmtree(pyrome_temp)
        pyrome_temp.mkdir(parents=True, exist_ok=True)

        all_layer_paths = {layer: [] for layer, _, _, _ in LAYER_SPECS}

        for tilenum in tilenums:
            tile_dir = pyrome_temp / f"tile_{tilenum}"
            tile_layers = download_tile_layers(
                bucket,
                args.year,
                tilenum,
                pyrome_id,
                tile_dir,
            )
            for layer, paths in tile_layers.items():
                all_layer_paths[layer].extend(paths)

        vrt_dir = pyrome_temp / "vrt"
        label_vrt = None
        predictor_vrts = []

        for layer_name, _, is_predictor, is_label in LAYER_SPECS:
            tif_paths = all_layer_paths.get(layer_name, [])
            if not tif_paths:
                print(f"WARN: No data for layer {layer_name} in pyrome {pyrome_id}")
                continue
            vrt_path = build_vrt(layer_name, tif_paths, vrt_dir)
            if is_label:
                label_vrt = vrt_path
            if is_predictor:
                predictor_vrts.append((layer_name, vrt_path))

        if label_vrt is None:
            print(f"Skipping pyrome {pyrome_id}: no label VRT")
            continue

        with rasterio.open(label_vrt) as label_ds:
            samples, class_points = stratified_sample(
                label_ds,
                geom,
                mode=args.mode,
                points_per_class=args.points_per_class,
                seed=args.seed,
            )

        if not samples:
            print(f"No samples generated for pyrome {pyrome_id}")
            continue

        predictor_datasets = []
        for prefix, vrt_path in predictor_vrts:
            predictor_datasets.append((prefix, rasterio.open(vrt_path)))

        try:
            predictor_data = sample_predictors(predictor_datasets, samples)
        finally:
            for _, ds in predictor_datasets:
                ds.close()

        df = pd.DataFrame(
            {
                "x": [p[0] for p in samples],
                "y": [p[1] for p in samples],
                "FBFM40Parent": [p[2] for p in samples],
            }
        )
        for name, values in predictor_data.items():
            df[name] = values

        output_name = f"pyrome_{pyrome_id}_{args.year}_{args.mode}_local_samples.csv"
        output_path = pyrome_temp / output_name
        df.to_csv(output_path, index=False)

        gcs_path = f"samples/pyrome_{pyrome_id}/{output_name}"
        upload_csv(bucket, output_path, gcs_path)

        if args.cleanup:
            shutil.rmtree(pyrome_temp)


if __name__ == "__main__":
    main()
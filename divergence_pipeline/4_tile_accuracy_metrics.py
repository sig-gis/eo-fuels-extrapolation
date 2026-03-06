import argparse
from pathlib import Path
import joblib
from yaml import safe_load, safe_dump

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio as rio
from rasterio.features import rasterize

from google.cloud import storage

from sklearn.preprocessing import LabelEncoder
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

PROJECT_NAME = 'pyregence-ee'
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


def load_config(yml_path):
    with open(yml_path) as f:
        config = safe_load(f)
    return config

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

    gdf = gdf.merge(support_rasters['evt']['data_dict'][['VALUE','LFRDB','EVT_NAME','EVT_ORDER','EVT_CLASS','EVT_SBCLS']],left_on='evt',right_on='VALUE').drop(['VALUE'],axis=1)
    gdf = gdf.merge(support_rasters['evc']['data_dict'][['VALUE','EVC_CLASS']],left_on='evc',right_on='VALUE').drop(['VALUE'],axis=1)
    gdf = gdf.merge(support_rasters['evh']['data_dict'][['VALUE','EVH_CLASS']],left_on='evh',right_on='VALUE').drop(['VALUE'],axis=1)
    gdf = gdf.merge(support_rasters['bps']['data_dict'][['VALUE','BPS_NAME','BPS_CLASS']],left_on='bps',right_on='VALUE').drop(['VALUE'],axis=1)

    gdf['tree_height'] = gdf.apply(assign_tree_height,axis=1)
    gdf['shrub_height'] = gdf.apply(assign_shrub_height,axis=1)
    gdf['herb_height'] = gdf.apply(assign_herb_height,axis=1)

    gdf['tree_cover'] = gdf.apply(assign_tree_cover,axis=1)
    gdf['shrub_cover'] = gdf.apply(assign_shrub_cover,axis=1)
    gdf['herb_cover'] = gdf.apply(assign_herb_cover,axis=1)

    gdf = gdf.drop(columns=['evt','evc','evh','bps','LFRDB','EVT_NAME','EVT_ORDER','EVT_SBCLS','EVC_CLASS','EVH_CLASS','BPS_NAME'])

    out_df = pd.DataFrame(gdf.drop(columns=['geometry','lat','lon']))

    return out_df

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

LF_VARS = ['EVT_CLASS','BPS_CLASS','nlcd','tree_height','shrub_height','herb_height','tree_cover','shrub_cover','herb_cover']
CATEGORICAL_BANDS = ['EVT_CLASS','BPS_CLASS','nlcd']

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
    # "bps": lambda count: [f"bps_b{idx + 1}" for idx in range(count)],
    # "evc": lambda count: [f"evc_b{idx + 1}" for idx in range(count)],
    # "evt": lambda count: [f"evt_b{idx + 1}" for idx in range(count)],
    # "evh": lambda count: [f"evh_b{idx + 1}" for idx in range(count)],
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

# def load_aef_data(aef_config):
#     if aef_config['bands'] == 'all':
#         num_bands = 64
    
#     band_names = BAND_NAME_OVERRIDES['aef']

# def load_topography_data(topo_config):
#     pass

# def load_spectral_stats_data(spx_config):
#     pass

# def load_seasonal_band_data(bnd_config):
#     pass

# def load_climate_data(climate_config):
#     pass

# def load_vegetation_data(veg_config):
#     pass

# BAND_GROUP_LOADING_FUNCS = {
#     'aef': load_aef_data,
#     'topography':load_topography_data,
#     'spectral_stats':load_spectral_stats_data,
#     'seasonal_bands':load_seasonal_band_data,
#     'climate':load_climate_data,
#     'vegetation':load_vegetation_data
# }

# def build_dataset(band_config):
#     df = pd.DataFrame()

#     return df

def load_rf_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    print("Loaded RF model:", type(model))

    cont_scaler_path = model_path.with_name(f"{model_path.stem}_cont_scaler.joblib")
    cat_scaler_path = model_path.with_name(f"{model_path.stem}_cat_scaler.joblib")
    label_encoder_path = model_path.with_name(f"{model_path.stem}_label_encoder.joblib")
    
    if not cont_scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {cont_scaler_path}")
    
    if not cat_scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {cont_scaler_path}")
    
    cat_scaler = joblib.load(cat_scaler_path)
    print('Loaded categorical scaler:', type(cat_scaler))

    cont_scaler = joblib.load(cont_scaler_path)
    print("Loaded continuous scaler:", type(cont_scaler))

    label_encoder = joblib.load(label_encoder_path)
    print("Loaded label encoder:", type(label_encoder))

    return model, cont_scaler, cat_scaler, label_encoder


def resolve_model_path(model_dir: Path, exp_name: str, pyrome_id: int, merge: bool) -> Path:
    if merge:
        return model_dir / f"{exp_name}_pyromes_merged.joblib"
    return model_dir / f"{exp_name}_pyrome_{pyrome_id}.joblib"

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
    cat_scaler,
    cont_scaler,
    label_encoder,
    tile_dir: Path,
    tilenum,
    year,
    label_name,
    label_classes,
    support_rasters,
    drop_labels
):
    tif_files = [f for f in tile_dir.glob("*.tif") if "label" not in f.name and "parentlabel" not in f.name and not any([tag in f.name for tag in ['evt','evc','bps','evh']])]

    if not tif_files:
        print(f"WARN: No predictor TIFFs found in {tile_dir}; skipping tile {tilenum}.")
        return None

    # print(f"Model expects {model.n_features_in_} features")
    # print(f"Feature names: {model.feature_names_in_}")

    #get coordinate grid for sampling additional datasets
    with rio.open(tif_files[0]) as src:
        band = src.read(1)
        height, width = band.shape[0],band.shape[1]
        cols, rows = np.meshgrid(np.arange(width),np.arange(height))
        xs, ys = rio.transform.xy(src.transform,rows,cols)
        lons = np.array(xs)
        lats = np.array(ys)
        source_crs = src.crs

    loc_gdf = pd.DataFrame.from_dict({'lat':lats,'lon':lons})
    loc_gdf = gpd.GeoDataFrame(
        loc_gdf,
        geometry = gpd.points_from_xy(loc_gdf.lon,loc_gdf.lat),
        crs=source_crs
    ).to_crs('EPSG:5070')

    veg_gdf = append_veg_features(loc_gdf,support_rasters)

    #scaler input features
    continuous_input_features = cont_scaler.feature_names_in_
    categorical_input_features = cat_scaler.feature_names_in_
    full_input_feature_list = np.concatenate([continuous_input_features,categorical_input_features])

    full_model_input_feature_list = model.feature_names_in_

    # print(f'Cont features from scaler: {continuous_input_features}')
    # print(f'Cat features from scaler: {categorical_input_features}')


    all_bands = []
    band_names = []
    predictor_nodata_masks = []
    for tif_file in tif_files:
        with rio.open(tif_file) as src:
            data = src.read()
            all_bands.append(data)
            # If band names are available
            layer_prefix = guess_layer_prefix(tif_file.name)
            band_names.extend(get_band_names(src, layer_prefix))
            nodata = src.nodata
            if nodata is not None:
                nodata_mask = np.any(data == nodata, axis=0)
                predictor_nodata_masks.append(nodata_mask)

    features = np.concatenate(all_bands, axis=0)
    bands, h, w = features.shape

    veg_np = np.transpose(np.reshape(veg_gdf.to_numpy(), shape=(h, w, veg_gdf.shape[-1])),axes=(2,0,1))
    features = np.concatenate([features,veg_np],axis=0)

    bands, h, w = features.shape
    band_names.extend(veg_gdf.columns)

    continuous_band_names = [band for band in band_names if band not in CATEGORICAL_BANDS]
    categorical_bands_names = CATEGORICAL_BANDS



    print(f"Input has {bands} features")
    # print(f"Band names: {band_names}")

    expected_features = full_input_feature_list
    actual_features = set(band_names)

    missing = [f for f in expected_features if f not in actual_features]
    extra = [f for f in band_names if f not in expected_features]

    if missing:
        raise ValueError(
            "Tile predictor bands do not match model features. "
            f"Missing: {missing} Extra: {extra}"
        )

    feature_map = {name: idx for idx, name in enumerate(band_names)}
    # bands = len(expected_features)
    data = np.transpose(features, (1, 2, 0)).reshape(-1, features.shape[0])
    data_df = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=expected_features)



    for name in expected_features:
        idx = feature_map.get(name)
        if idx is not None:
            data_df[name] = data[:, idx]

    cont_data_np = data_df[continuous_input_features].to_numpy().astype(np.float64)
    cat_data_np = data_df[categorical_bands_names].to_numpy()

    if np.isnan(cont_data_np).any():
        nan_ratio = np.isnan(cont_data_np).mean()
        print(f"WARN: Predictor NaN ratio for tile {tilenum}: {nan_ratio:.6f}")
        cont_data_np = np.nan_to_num(cont_data_np, nan=0.0)
    if hasattr(cont_scaler, "mean_") and hasattr(cont_scaler, "scale_"):
        means = cont_scaler.mean_
        scales = cont_scaler.scale_
        feature_means = cont_data_np.mean(axis=0)
        feature_stds = cont_data_np.std(axis=0)
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

    cont_scaled_features = cont_scaler.transform(data_df[continuous_input_features].astype(np.float32).fillna(0))
    cat_scaled_features = cat_scaler.transform(data_df[categorical_input_features])

    scaled_features = pd.DataFrame(np.concatenate([cont_scaled_features,cat_scaled_features],axis=1),columns=full_model_input_feature_list)

    print(f'scaled_features columns: {scaled_features.columns}')
    # scaled_df = pd.DataFrame(scaled_features, columns=expected_features)
    preds_encoded = model.predict(scaled_features)
    print(f'Input Pixels: {scaled_features.shape[0]}')
    model_classes = np.array(model.classes_)

    # Load true labels
    if label_name == "FBFM40Parent":
        label_file = tile_dir / f"tilenum00{tilenum}_fm40parentlabel_{year}.tif"
    else:
        label_file = tile_dir / f"tilenum00{tilenum}_fm40label_{year}.tif"
    with rio.open(label_file) as src:
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
    tile_label_vals, tile_label_counts = np.unique(true_labels,return_counts=True)

    print(f'Labels seen in tile: {dict(zip(tile_label_vals,tile_label_counts))}')

    valid_mask = np.isin(true_labels, label_classes)
    print(f'Valid Eval Pixels: {np.sum(valid_mask)}')
    true_labels = true_labels[valid_mask]
    preds_encoded = preds_encoded[valid_mask]

    

    if len(true_labels) == 0:
        print(f"WARN: No valid parent labels for tile {tilenum}; skipping.")
        return None

    true_encoded = label_encoder.transform(true_labels)

    print(f'true_labels: {true_encoded}')
    print(f'preds_encoded: {preds_encoded}')

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

    out_report = classification_report(label_encoder.inverse_transform(true_encoded),label_encoder.inverse_transform(preds_encoded),zero_division=0)
    print(out_report)

    label_vals, label_counts = np.unique(label_encoder.inverse_transform(true_encoded),return_counts=True)
    print('True Label Distribution:')
    for k,v in zip(list(label_vals),list(label_counts)):
        print(f'Class: {int(k)} | Pixel Count: {int(v)}')

    pred_vals, pred_counts = np.unique(label_encoder.inverse_transform(preds_encoded),return_counts=True)
    print('Predicted Label Distribution: ')
    for k,v in zip(list(pred_vals),list(pred_counts)):
        print(f'Class: {int(k)} | Pixel Count: {int(v)}')
    

    cm = confusion_matrix(true_encoded, preds_encoded, labels=model_classes)
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
    zero_ratio = (cont_data_np == 0).mean()
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
        
        tiles.append((folder, int(folder)))
    return tiles

def main():
    parser = argparse.ArgumentParser(
        description="Compute FBFM40 accuracy metrics per tile using trained RF model"
    )
    parser.add_argument(
        '--tiles',
        nargs='+',
        type=int,
        help='List of tilenums to compute metrics'
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
        '--out-dir',
        type=Path,
        default=Path('./output/'),
        help='Directory for output metrics'
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
    parser.add_argument(
        '--training_config',
        help='Config used for training model'
    )

    args = parser.parse_args()

    if args.training_config:
        config = load_config(args.training_config)

    exp_name = config['exp_name']

    support_rasters = load_support_rasters(config['data'])

    drop_labels = config['target']['drop_labels']

    label_name = args.label
    if label_name == "FBFM40Parent":
        label_classes = parent_classes
    else:
        label_classes = np.array(from_vals)
        label_classes = np.array([label for label in label_classes if not label in drop_labels])

    print(f'Evaluating on classes: {label_classes}')

    results = []
    for pyrome_id in args.pyromes:
        model_path = resolve_model_path(args.model_dir, exp_name, pyrome_id, args.merge)
        model, cont_scaler, cat_scaler, label_encoder = load_rf_model(model_path)
        pyrome_dir = args.temp_root / f"pyrome_{pyrome_id}"


        tiles = list_pyrome_tiles(pyrome_dir)

        if args.tiles:
            tiles = [(folder_name,tilenum) for (folder_name,tilenum) in tiles if tilenum in args.tiles]
        
        if not tiles:
            print(f"No tiles found for pyrome {pyrome_id} in {pyrome_dir}")
            continue
        for folder_name, tilenum in tiles:
            print(f"\n=== Processing Pyrome {pyrome_id} Tile {tilenum} ===")
            tile_dir = pyrome_dir / folder_name
            metrics = compute_tile_accuracy(
                model,
                cat_scaler,
                cont_scaler,
                label_encoder,
                tile_dir,
                tilenum,
                args.year,
                label_name,
                label_classes,
                support_rasters,
                drop_labels
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
    csv_path = args.out_dir / f"tile_fbfm40_accuracy_metrics_pyrome_{pyrome_tag}_{exp_name}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved metrics to {csv_path}")
    print(df)

if __name__ == "__main__":
    main()

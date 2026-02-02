# eo-fuels-extrapolation
Repository gathering code and documentation for work on extrapolating fuels to unmapped regions using EO data

## Data

Most recent  tabluar data version can be downloaded from at the following link: [FuelsData3](https://drive.google.com/drive/folders/1N17aDPLlQQGUKxlJlQuSQNarZ4KrnkMq?usp=sharing)

Data points are separated per pyrome per year (2021,2022,2023) in California with ~2000 points sampled for each, 30m resolution

Current Data Attributes are as follows:

- A00-A63 AlphaEarth
- Spectral Indices: Min, Max, and Median of Select Spectral & Vegetation index values as well as HLS Bands
    - NDVI
    - NBR
    - SAVI
    - mSAVI
    - NDMI
    - TCB
    - TCG
    - TSW
    - VARI
- Climate Norms and Yearly Divergence
    - ppt
    - solclear
    - soltotal
    - tdmean
    - tmax
    - tmean
    - tmin
    - vpdmax
    - vpdmin
- Topography
    - elevation
    - slope
    - aspect
    - mtpi
- Vegetation
    - Existing Vegetation Type (EVT)
    - Existing Vegetation Height (EVH)
    - Existing Vegetation Cover (EVC)
    - Bio-physical Setting (BPS)

Target
    - FBFM40: LANDFIRE Fuel Classification 45 Classes 

## Analysis Notebooks

- fuels_characteristics_prediction.ipynb

## Pyretechnics Analysis
This workflow is an end-to-end pipeline for evaluating how machine-learned fuel models affect wildfire behavior modeling. It exports per-tile geospatial inputs from Google Earth Engine (satellite embeddings, LANDFIRE fuels, canopy structure, and terrain) to cloud storage, runs a trained random-forest model to predict fuel model classes from satellite data, and then uses both the original and predicted fuels to simulate fire behavior with `pyretechnics`. The results are compared to quantify how fuel prediction errors propagate into fire behavior metrics, producing a CSV of tile-level divergence statistics and, optionally, interactive plots for visual analysis.

## Environment Setup
### Initialize and authenticate to Google Cloud Storage
1. Download and run the Google Cloud CLI installer for your Operating System from [here](https://docs.cloud.google.com/sdk/docs/install-sdk), following all installation recommendations.
2. Open your terminal, run `gcloud version` to ensure gcloud is installed properly.
3. After you install the gcloud CLI, perform initial setup tasks by running `gcloud init` in your terminal. 
- You can also run `gcloud init` to change your settings or create a new configuration.
- You can also run `gcloud auth login` as an alternative to `gcloud init` to authorize with a user account without setting up a configuration. 

### Setup using Conda and `requirements.txt`
Follow these steps to set up your environment and install the required Python packages using conda and the `requirements.txt` file.

1. Download and run the Anaconda installer for your Operating System from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), following all installation recommendations.
2. Open your terminal, run `conda --version` to ensure Anaconda is installed properly. 
3. Create a fresh conda environment with python installed: `conda create -n your-env-name python=3.10.19`.
4. Activate your new environment: `conda activate your-env-name`.
5. Make sure the `requirements.txt` file is in the current directory, then run: 
```
pip install -r requirements.txt
```
6. The `requirements.txt` file installs `numpy==1.24.4`. We actually need `numpy==1.26.4`, however, `pyretechnics==2025.5.15` can't be installed with `numpy==1.26.4` so we'll upgrade `numpy==1.24.4` to `numpy==1.26.4` after installing the packages with the `requirements.txt` file.
```
conda install numpy==1.26.4 
```

# Main Pyretechnics Workflow
The Pyretechnics analysis is now organized as a five-step pipeline in `divergence_pipeline/`.
Run the scripts in order to export EO inputs, sample training data, train models, evaluate
tile-level accuracy, and run the full fire behavior + divergence analysis.

### 1) Export per-tile EO inputs from Earth Engine
Exports satellite embeddings, LANDFIRE fuels, canopy structure, terrain, HLS stats, climate
normals, and vegetation layers to GCS, skipping any outputs that already exist.

```
python divergence_pipeline/1_export_embeddings_tiles.py \
  --year 2022 \
  --tilenums 01180 00371
```

Optional: limit by pyrome IDs instead of explicit tiles.
```
python divergence_pipeline/1_export_embeddings_tiles.py \
  --year 2022 \
  --pyromes 12 27
```

### 2) Mosaic tiles locally and sample training points
Downloads the GCS GeoTIFFs for a pyrome, mosaics by layer, and creates stratified samples.
Outputs CSVs are uploaded back to GCS.

```
python divergence_pipeline/2_local_mosaic_sample.py \
  --pyromes 27 \
  --year 2022 \
  --mode equal \
  --points-per-class 3000
```

### 3) Train random-forest models from sample CSVs
Trains a RF classifier on the sampled data. Use `--pyromes` to merge per-pyrome CSVs or
`--csv` to train from a single local file.

```
python divergence_pipeline/3_train_models.py \
  --pyromes 27 \
  --year 2022 \
  --mode equal \
  --output-dir data
```

### 4) Compute tile-level accuracy metrics
Downloads tile GeoTIFFs, runs inference using a trained model, and writes a CSV of metrics.

```
python divergence_pipeline/4_tile_accuracy_metrics.py \
  --model-path temp/geoai-fuels-tiles/utils/rf_zone_27.joblib \
  --tilenums 01180 00371 \
  --year 2022
```

### 5) Run full Pyretechnics fire behavior + divergence analysis
Runs inference, simulates fire behavior for LANDFIRE vs predicted fuels, and outputs
divergence statistics (plus optional Plotly HTML plots).

```
python divergence_pipeline/5_pyro_pipeline.py \
  --tilenums 01180 00371 \
  --year 2022
```

Optional: generate Plotly plots of divergence metrics.
```
python divergence_pipeline/5_pyro_pipeline.py \
  --tilenums 01180 00371 \
  --year 2022 \
  --plot
```
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

Before completing the workflow, install the necessary packages by using the `requirements.txt` file.
```
pip install -r requirements.txt
```

This workflow can be completed in two steps. First step, run the `export_embeddings_tiles.py` file, with a list of tiles, to export to the necessary files from Earth Engine.
Example: 
```
python export_embeddings_tiles.py --tilenums 01180 00371
```
Second step, run the `pyro_pipeline.py` file, with a list of tiles, to complete the analysis and get the final behavior metrics and divergence stats. 
```
python pyro_pipeline.py --tilenums 01180 00371
```
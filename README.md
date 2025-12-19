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
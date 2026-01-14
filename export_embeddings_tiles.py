import argparse
import ee
import geemap
import geopandas as gpd
from google.cloud import storage
import os
import numpy as np

GPKG_BUCKET = "geoai-fuels-tiles"
GPKG_BLOB = r"utils/tiles48km_polygons_firefactor.gpkg"
LOCAL_GPKG = r"C:\Users\edalt\RD_Fuels\eo-fuels-extrapolation\temp\tiles48km_polygons_firefactor.gpkg"

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

def gcs_prefix_exists(bucket, prefix):
    """Return True if any object exists with this prefix."""
    blobs = bucket.list_blobs(prefix=prefix, max_results=1)
    return any(True for _ in blobs)

def download_gpkg_if_needed(client, bucket_name, blob_name, local_path):
    if os.path.exists(local_path):
        return

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(
            f"GPKG not found at gs://{bucket_name}/{blob_name}"
        )

    print(f"Downloading gs://{bucket_name}/{blob_name}")
    blob.download_to_filename(local_path)

def export_if_missing(
    *,
    image,
    desc,
    bucket,
    region,
    scale,
    crs,
    prefix=None,
    max_pixels=1e13,
):
    gcs_path = f"{prefix}/{desc}" if prefix else desc

    if gcs_prefix_exists(bucket, gcs_path):
        print(f"SKIP (exists in GCS): {gcs_path}")
        return

    print(f"EXPORT: {gcs_path}")
    ee.batch.Export.image.toCloudStorage(
        image=image,
        description=gcs_path.replace("/", "_"),  # EE requires unique task names
        bucket=bucket.name,
        fileNamePrefix=gcs_path,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=max_pixels,
    ).start()

def scale_hls(img):
    return img.multiply(0.0001).copyProperties(img, img.propertyNames())

def add_indices(img):
    red   = img.select("B4")
    nir   = img.select("B5")
    swir1 = img.select("B6")
    swir2 = img.select("B7")
    blue  = img.select("B2")
    green = img.select("B3")

    ndvi  = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    nbr   = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")
    ndmi  = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")

    savi  = nir.subtract(red).multiply(1.5).divide(
        nir.add(red).add(0.5)
    ).rename("SAVI")

    msavi = (
        nir.multiply(2).add(1)
        .subtract(
            nir.multiply(2).add(1).pow(2)
            .subtract(nir.subtract(red).multiply(8))
            .sqrt()
        )
        .divide(2)
        .rename("MSAVI")
    )

    vari = green.subtract(red).divide(
        green.add(red).subtract(blue)
    ).rename("VARI")

    # Tasselled Cap (Landsat-style coefficients; commonly used for HLS)
    tcb = (
        blue.multiply(0.2043)
        .add(green.multiply(0.4158))
        .add(red.multiply(0.5524))
        .add(nir.multiply(0.5741))
        .add(swir1.multiply(0.3124))
        .add(swir2.multiply(0.2303))
        .rename("TCB")
    )

    tcg = (
        blue.multiply(-0.1603)
        .add(green.multiply(-0.2819))
        .add(red.multiply(-0.4934))
        .add(nir.multiply(0.7940))
        .add(swir1.multiply(-0.0002))
        .add(swir2.multiply(-0.1446))
        .rename("TCG")
    )

    tsw = (
        blue.multiply(0.0315)
        .add(green.multiply(0.2021))
        .add(red.multiply(0.3102))
        .add(nir.multiply(0.1594))
        .add(swir1.multiply(-0.6806))
        .add(swir2.multiply(-0.6109))
        .rename("TSW")
    )

    evi = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).rename("EVI")

    return img.addBands([
        ndvi, nbr, savi, msavi, ndmi, vari, tcb, tcg, tsw, evi
    ])

def seasonal_hls_band_medians(year):
    seasons = [
        ("winter", ee.Date(f"{year}-01-01"), ee.Date(f"{year}-04-01")),
        ("spring", ee.Date(f"{year}-04-01"), ee.Date(f"{year}-07-01")),
        ("summer", ee.Date(f"{year}-07-01"), ee.Date(f"{year}-10-01")),
        ("fall",   ee.Date(f"{year}-10-01"), ee.Date(f"{year+1}-01-01")),
    ]

    def per_season(season):
        name, start, end = season

        ic = (
            ee.ImageCollection("NASA/HLS/HLSL30/v002")
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVERAGE", 20))
            .map(scale_hls)
            .select(["B2", "B3", "B4", "B5", "B6", "B7"])
        )

        img = ic.median()

        return img.rename([
            f"B2_{name}", f"B3_{name}", f"B4_{name}",
            f"B5_{name}", f"B6_{name}", f"B7_{name}",
        ])

    seasonal_images = [per_season(s) for s in seasons]
    return ee.Image.cat(seasonal_images)

def monthly_index_stats(year):
    months = ee.List.sequence(1, 12)

    index_bands = [
        "NDVI", "NBR", "NDMI", "SAVI", "MSAVI", "VARI", "EVI",
        "TCB", "TCG", "TSW"
    ]

    reducers = (
        ee.Reducer.min()
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
    )

    def per_month(m):
        m = ee.Number(m)
        start = ee.Date.fromYMD(year, m, 1)
        end = start.advance(1, "month")

        ic = (
            ee.ImageCollection("NASA/HLS/HLSL30/v002")
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVERAGE", 20))
            .map(scale_hls)
            .map(add_indices)
            .select(index_bands)
        )

        stats = ic.reduce(reducers)

        band_names = stats.bandNames()
        renamed = band_names.map(
            lambda b: ee.String(b)
            .cat("_m")
            .cat(m.format("%02d"))
        )

        return stats.rename(renamed)

    monthly_images = months.map(per_month)
    return ee.ImageCollection.fromImages(monthly_images).toBands()

def annual_index_stats(year):
    index_bands = [
        "NDVI", "NBR", "NDMI", "SAVI", "MSAVI", "VARI", "EVI",
        "TCB", "TCG", "TSW"
    ]

    reducers = (
        ee.Reducer.min()
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
    )

    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")

    ic = (
        ee.ImageCollection("NASA/HLS/HLSL30/v002")
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVERAGE", 20))
        .map(scale_hls)
        .map(add_indices)
        .select(index_bands)
    )

    stats = ic.reduce(reducers)

    band_names = stats.bandNames()
    # renamed = band_names.map(
    #     lambda b: ee.String(b)
    #     .cat("_y")
    #     .cat(ee.Number(year).format())
    # )

    return stats.rename(band_names)

def main(args):
    ee.Initialize(project=args.ee_project)

    gcs_client = storage.Client()
    gcs_bucket = gcs_client.bucket(args.bucket)

    download_gpkg_if_needed(
        gcs_client,
        GPKG_BUCKET,
        GPKG_BLOB,
        LOCAL_GPKG,
    )

    tiles = gpd.read_file(LOCAL_GPKG)

    if args.tilenums:
        tiles = tiles[tiles["tilenum"].isin(args.tilenums)]

    if tiles.empty:
        raise RuntimeError("No tiles selected after filtering")

    print(f"Processing {len(tiles)} tiles")

    to_fc = geemap.gdf_to_ee(tiles)

    for tile in to_fc.toList(to_fc.size()).getInfo():
        tilenum = tile["properties"]["tilenum"]
        region = tile["geometry"]["coordinates"]

        print(f"\n=== Tile {tilenum} ===")

        embeddings = (
            ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
            .mosaic()
        )

        fbfm40 = (
            ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fbfm40")
            .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
            .first()
            .select("FBFM40")
        )

        parent = (
            ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fbfm40")
            .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
            .first()
            .select("FBFM40")
            .remap(from_vals, to_vals)
            .rename("FBFM40Parent")
        )

        ch = (
            ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/ch")
            .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
            .first()
            .select("CH")
        )

        cc = (
            ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cc")
            .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
            .first()
            .select("CC")
        )

        cbh = (
            ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cbh")
            .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
            .first()
            .select("CBH")
        )

        cbd = (
            ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cbd")
            .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
            .first()
            .select("CBD")
        )

        elevation = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        slope = ee.Terrain.slope(elevation)
        aspect = ee.Terrain.aspect(elevation)

        mtpi = ee.Image('CSP/ERGo/1_0/Global/SRTM_mTPI').select('elevation').rename('mtpi')

        # UPDATE/ADD TO EE ASSETS
        # RYA
        bps = ee.ImageCollection("LANDFIRE/Vegetation/BPS/v1_4_0").select('BPS').mosaic()
        evc = ee.ImageCollection('LANDFIRE/Vegetation/EVC/v1_4_0').select('EVC').mosaic()
        evt = (
            ee.ImageCollection('LANDFIRE/Vegetation/EVT/v1_4_0')
            .select('EVT')
            .map(lambda img: img.toShort())
            .mosaic()
        )
        evh = ee.ImageCollection('LANDFIRE/Vegetation/EVH/v1_4_0').select('EVH').mosaic()

        # evc = (
        #     ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/evc")
        #     .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
        #     .first()
        #     .select("EVC")
        # )

        # evh = (
        #     ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/evh")
        #     .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
        #     .first()
        #     .select("EVH")
        # )

        # evt = (
        #     ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/evt")
        #     .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
        #     .first()
        #     .select("EVT")
        # )

        # bps = (
        #     ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/bps")
        #     .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
        #     .first()
        #     .select("BPS")
        # )
        ###############

        annual_hls_bands = annual_index_stats(args.year)
        seasonal_hls_bands = seasonal_hls_band_medians(args.year)
        # monthly_hls_indices = monthly_index_stats(args.year)

        climate_norm_variables = ['ppt','tmean','tmin','tmax','tdmean','vpdmin','vpdmax','solclear','solslope','soltotal']
        climatenormals = (ee.ImageCollection('OREGONSTATE/PRISM/Norm91m')
            .first()
            .select(climate_norm_variables,[climvar + '_norm' for climvar in climate_norm_variables])
        )

        prism = (ee.ImageCollection('OREGONSTATE/PRISM/ANm')
        .filter(ee.Filter.calendarRange(args.year, args.year, 'year'))
        .select(['ppt','tmean','tmin','tmax','tdmean','vpdmin','vpdmax'])
        .mean()
        )

        tile_prefix = f"{tilenum}"

        # print(f"Band names for tilenum{tilenum}_aef_{args.year}: {embeddings.bandNames().getInfo()}")
        # export_if_missing(
        #     image=embeddings,
        #     desc=f"tilenum{tilenum}_aef_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_fm40label_{args.year}: {fbfm40.bandNames().getInfo()}")
        # export_if_missing(
        #     image=fbfm40,
        #     desc=f"tilenum{tilenum}_fm40label_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_fm40parentlabel_{args.year}: {parent.bandNames().getInfo()}")
        # export_if_missing(
        #     image=parent,
        #     desc=f"tilenum{tilenum}_fm40parentlabel_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_ch_{args.year}: {ch.bandNames().getInfo()}")
        # export_if_missing(
        #     image=ch,
        #     desc=f"tilenum{tilenum}_ch_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_cc_{args.year}: {cc.bandNames().getInfo()}")
        # export_if_missing(
        #     image=cc,
        #     desc=f"tilenum{tilenum}_cc_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_cbh_{args.year}: {cbh.bandNames().getInfo()}")
        # export_if_missing(
        #     image=cbh,
        #     desc=f"tilenum{tilenum}_cbh_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_cbd_{args.year}: {cbd.bandNames().getInfo()}")
        # export_if_missing(
        #     image=cbd,
        #     desc=f"tilenum{tilenum}_cbd_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_elevation: {elevation.bandNames().getInfo()}")
        # export_if_missing(
        #     image=elevation,
        #     desc=f"tilenum{tilenum}_elevation",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_slope: {slope.bandNames().getInfo()}")
        # export_if_missing(
        #     image=slope,
        #     desc=f"tilenum{tilenum}_slope",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_aspect: {aspect.bandNames().getInfo()}")
        # export_if_missing(
        #     image=aspect,
        #     desc=f"tilenum{tilenum}_aspect",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_mtpi: {mtpi.bandNames().getInfo()}")
        # export_if_missing(
        #     image=mtpi,
        #     desc=f"tilenum{tilenum}_mtpi",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        print(f"Band names for tilenum{tilenum}_annual_hls_bands: {annual_hls_bands.bandNames().getInfo()}")
        export_if_missing(
            image=annual_hls_bands,
            desc=f"tilenum{tilenum}_spectral_stats_{args.year}",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        # print(f"Band names for tilenum{tilenum}_monthly_hls_indices: {monthly_hls_indices.bandNames().getInfo()}")
        # export_if_missing(
        #     image=monthly_hls_indices,
        #     desc=f"tilenum{tilenum}_spectral_stats_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        print(f"Band names for tilenum{tilenum}_seasonal_hls_bands: {seasonal_hls_bands.bandNames().getInfo()}")
        export_if_missing(
            image=seasonal_hls_bands,
            desc=f"tilenum{tilenum}_hls_band_stats_{args.year}",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        # print(f"Band names for tilenum{tilenum}_climatenormals: {climatenormals.bandNames().getInfo()}")
        # export_if_missing(
        #     image=climatenormals,
        #     desc=f"tilenum{tilenum}_climatenormals_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_prism: {prism.bandNames().getInfo()}")
        # export_if_missing(
        #     image=prism,
        #     desc=f"tilenum{tilenum}_prism_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_evc: {evc.bandNames().getInfo()}")
        # export_if_missing(
        #     image=evc,
        #     desc=f"tilenum{tilenum}_evc_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_evt: {evt.bandNames().getInfo()}")
        # export_if_missing(
        #     image=evt,
        #     desc=f"tilenum{tilenum}_evt_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_evh: {evh.bandNames().getInfo()}")
        # export_if_missing(
        #     image=evh,
        #     desc=f"tilenum{tilenum}_evh_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

        # print(f"Band names for tilenum{tilenum}_bps: {bps.bandNames().getInfo()}")
        # export_if_missing(
        #     image=bps,
        #     desc=f"tilenum{tilenum}_bps_{args.year}",
        #     bucket=gcs_bucket,
        #     region=region,
        #     scale=args.scale,
        #     crs=args.crs,
        #     prefix=tile_prefix,
        # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export EO fuels tiles to GCS with EE, skipping existing outputs"
    )

    parser.add_argument(
        "--bucket",
        default="geoai-fuels-tiles",
        help="GCS bucket name",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Embedding / LANDFIRE year",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=30,
        help="Export pixel scale (meters)",
    )

    parser.add_argument(
        "--crs",
        default="EPSG:5070",
        help="Export CRS",
    )

    parser.add_argument(
        "--ee-project",
        default="pyregence-ee",
        help="Earth Engine project",
    )

    parser.add_argument(
        "--tilenums",
        nargs="+",
        help="Optional list of tilenums to process (e.g. 01180 00371)",
    )

    args = parser.parse_args()
    main(args)

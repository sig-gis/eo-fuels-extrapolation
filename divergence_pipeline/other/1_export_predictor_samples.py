import argparse

import ee


def scale_hls(img):
    return img.multiply(0.0001).copyProperties(img, img.propertyNames())


def add_indices(img):
    red = img.select("B4")
    nir = img.select("B5")
    swir1 = img.select("B6")
    swir2 = img.select("B7")
    blue = img.select("B2")
    green = img.select("B3")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")

    savi = nir.subtract(red).multiply(1.5).divide(
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


def seasonal_hls_band_medians(year, region, cloud_cover_max):
    seasons = [
        ("winter", ee.Date(f"{year}-01-01"), ee.Date(f"{year}-04-01")),
        ("spring", ee.Date(f"{year}-04-01"), ee.Date(f"{year}-07-01")),
        ("summer", ee.Date(f"{year}-07-01"), ee.Date(f"{year}-10-01")),
        ("fall", ee.Date(f"{year}-10-01"), ee.Date(f"{year + 1}-01-01")),
    ]

    def per_season(season):
        name, start, end = season

        ic = (
            ee.ImageCollection("NASA/HLS/HLSL30/v002")
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVERAGE", cloud_cover_max))
            .filterBounds(region)
            .map(scale_hls)
            .select(["B2", "B3", "B4", "B5", "B6", "B7"])
            .map(lambda img: img.clip(region))
        )

        img = ic.median()

        return img.rename([
            f"B2_{name}", f"B3_{name}", f"B4_{name}",
            f"B5_{name}", f"B6_{name}", f"B7_{name}",
        ])

    seasonal_images = [per_season(s) for s in seasons]
    return ee.Image.cat(seasonal_images)


def annual_index_stats(year, region, cloud_cover_max, parallel_scale):
    index_bands = [
        "NDVI", "NBR", "NDMI", "SAVI", "MSAVI", "VARI", "EVI",
        "TCB", "TCG", "TSW",
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
        .filter(ee.Filter.lt("CLOUD_COVERAGE", cloud_cover_max))
        .filterBounds(region)
        .map(scale_hls)
        .map(add_indices)
        .select(index_bands)
        .map(lambda img: img.clip(region))
    )

    stats = ic.reduce(reducers, parallelScale=parallel_scale)
    band_names = stats.bandNames()
    return stats.rename(band_names)


def build_predictor_stack(year, region, cloud_cover_max, parallel_scale):
    embeddings = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mosaic()
    )

    annual_hls_bands = annual_index_stats(
        year,
        region,
        cloud_cover_max=cloud_cover_max,
        parallel_scale=parallel_scale,
    )
    seasonal_hls_bands = seasonal_hls_band_medians(
        year,
        region,
        cloud_cover_max=cloud_cover_max,
    )

    elevation = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    mtpi = (
        ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI")
        .select("elevation")
        .rename("mtpi")
    )

    bps = ee.ImageCollection("LANDFIRE/Vegetation/BPS/v1_4_0").select("BPS").mosaic()
    evc = ee.ImageCollection("LANDFIRE/Vegetation/EVC/v1_4_0").select("EVC").mosaic()
    evt = (
        ee.ImageCollection("LANDFIRE/Vegetation/EVT/v1_4_0")
        .select("EVT")
        .map(lambda img: img.toShort())
        .mosaic()
    )
    evh = ee.ImageCollection("LANDFIRE/Vegetation/EVH/v1_4_0").select("EVH").mosaic()
    esp = ee.Image("LANDFIRE/Vegetation/ESP/v1_2_0/CONUS").clip(region).rename("ESP")

    climate_norm_variables = [
        "ppt",
        "tmean",
        "tmin",
        "tmax",
        "tdmean",
        "vpdmin",
        "vpdmax",
        "solclear",
        "solslope",
        "soltotal",
    ]

    climatenormals = (
        ee.ImageCollection("OREGONSTATE/PRISM/Norm91m")
        .first()
        .select(
            climate_norm_variables,
            [f"{var}_norm" for var in climate_norm_variables],
        )
    )

    prism = (
        ee.ImageCollection("OREGONSTATE/PRISM/ANm")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .select([
            "ppt",
            "tmean",
            "tmin",
            "tmax",
            "tdmean",
            "vpdmin",
            "vpdmax",
        ])
        .mean()
    )

    return embeddings.addBands([
        annual_hls_bands,
        seasonal_hls_bands,
        elevation,
        slope,
        aspect,
        mtpi,
        bps,
        evc,
        evt,
        evh,
        esp,
        climatenormals,
        prism,
    ])


def load_sample_asset(asset_root, pyrome_id, year, mode):
    asset_id = f"{asset_root}/pyrome_{pyrome_id}_{year}_{mode}"
    return ee.FeatureCollection(asset_id)


def export_pyrome_samples(args, pyrome_id):
    samples = load_sample_asset(args.asset_root, pyrome_id, args.year, args.mode)
    region = samples.geometry()
    predictors = build_predictor_stack(
        args.year,
        region,
        cloud_cover_max=args.cloud_cover_max,
        parallel_scale=args.parallel_scale,
    )

    sampled = predictors.sampleRegions(
        collection=samples,
        properties=["pyrome_id", "lat", "lon", "FBFM40Parent"],
        scale=args.scale,
        geometries=True,
        tileScale=args.tile_scale,
    )

    description = f"pyrome_{pyrome_id}_{args.year}_{args.mode}_samples"
    prefix = f"samples/pyrome_{pyrome_id}/{description}"

    task = ee.batch.Export.table.toCloudStorage(
        collection=sampled,
        description=description,
        bucket=args.bucket,
        fileNamePrefix=prefix,
        fileFormat="CSV",
    )
    task.start()
    print(f"Started export: gs://{args.bucket}/{prefix}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Sample predictor stacks using pyrome assets and export CSVs to GCS."
    )
    parser.add_argument(
        "--pyromes",
        type=int,
        nargs="+",
        required=True,
        help="Pyrome IDs for sampling assets",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="FBFM40/embedding year",
    )
    parser.add_argument(
        "--mode",
        choices=["dist", "equal"],
        default="equal",
        help="Sampling mode used for the asset naming",
    )
    parser.add_argument(
        "--asset-root",
        default="projects/pyregence-ee/assets/fuels-ai",
        help="Root folder for sample assets",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=30,
        help="Sampling scale (meters)",
    )
    parser.add_argument(
        "--bucket",
        default="geoai-fuels-tiles",
        help="GCS bucket name",
    )
    parser.add_argument(
        "--ee-project",
        default="pyregence-ee",
        help="Earth Engine project",
    )
    parser.add_argument(
        "--cloud-cover-max",
        type=int,
        default=10,
        help="Max HLS CLOUD_COVERAGE to include (lower reduces memory)",
    )
    parser.add_argument(
        "--parallel-scale",
        type=int,
        default=4,
        help="Parallel scale for annual HLS reduce to reduce memory usage",
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=4,
        help="Tile scale for sampling to reduce memory usage",
    )

    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)

    for pyrome_id in args.pyromes:
        export_pyrome_samples(args, pyrome_id)


if __name__ == "__main__":
    main()
import argparse

import ee

from_vals = [
    91, 92, 93, 98, 99,
    101, 102, 103, 104, 105, 106, 107, 108, 109,
    121, 122, 123, 124,
    141, 142, 143, 144, 145, 146, 147, 148, 149,
    161, 162, 163, 164, 165,
    181, 182, 183, 184, 185, 186, 187, 188, 189,
    201, 202, 203, 204,
]

to_vals = [
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7,
]


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


def get_pyrome_geometry(pyrome_id):
    pyromes = ee.FeatureCollection(
        "projects/pyregence-ee/assets/Pyromes_CONUS_20200206"
    )
    pyrome = pyromes.filter(ee.Filter.eq("PYROME", pyrome_id))
    return pyrome.geometry()


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

    # return embeddings.addBands([
    #     # annual_hls_bands,
    #     seasonal_hls_bands,
    #     elevation,
    #     slope,
    #     aspect,
    #     mtpi,
    #     bps,
    #     evc,
    #     evt,
    #     evh,
    #     esp,
    #     climatenormals,
    #     prism,
    # ])

    return annual_hls_bands


def build_parent_labels(year):
    fbfm40 = (
        ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fbfm40")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .first()
        .select("FBFM40")
    )

    parent = fbfm40.remap(from_vals, to_vals).rename("FBFM40Parent")
    return parent


def compute_class_points(hist_dict, points_per_class, mode):
    class_values = hist_dict.keys().map(lambda k: ee.Number.parse(k))
    counts = class_values.map(lambda k: ee.Number(hist_dict.get(k)))
    total_count = ee.Number(counts.reduce(ee.Reducer.sum()))

    if mode == "dist":
        total_points = ee.Number(points_per_class).multiply(7)
        class_points = counts.map(
            lambda c: ee.Number(c)
            .divide(total_count)
            .multiply(total_points)
            .round()
            .max(1)
        )
    else:
        class_points = class_values.map(lambda _: ee.Number(points_per_class))

    return class_values, class_points


def export_samples(args):
    ee.Initialize(project=args.ee_project)

    region = get_pyrome_geometry(args.pyrome)
    parent = build_parent_labels(args.year)
    predictors = build_predictor_stack(
        args.year,
        region,
        cloud_cover_max=args.cloud_cover_max,
        parallel_scale=args.parallel_scale,
    )

    image_to_sample = predictors.addBands(parent)

    histogram = parent.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=region,
        scale=args.scale,
        maxPixels=1e13,
        tileScale=args.tile_scale,
    ).get("FBFM40Parent")

    hist_dict = ee.Dictionary(histogram)
    class_values, class_points = compute_class_points(
        hist_dict,
        args.points_per_class,
        args.mode,
    )

    samples = image_to_sample.stratifiedSample(
        numPoints=args.points_per_class,
        classBand="FBFM40Parent",
        region=region,
        scale=args.scale,
        classValues=class_values,
        classPoints=class_points,
        geometries=True,
        seed=args.seed,
        dropNulls=False,
        tileScale=args.tile_scale,
    )

    prefix = f"samples/pyrome_{args.pyrome}"
    description = f"pyrome_{args.pyrome}_{args.year}_{args.mode}"
    file_prefix = f"{prefix}/{description}"

    task = ee.batch.Export.table.toCloudStorage(
        collection=samples,
        description=description,
        bucket=args.bucket,
        fileNamePrefix=file_prefix,
        fileFormat="CSV",
    )
    task.start()
    print(
        f"Started export: gs://{args.bucket}/{file_prefix}.csv"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export stratified pyrome samples with predictor bands and FBFM40 parent labels"
    )
    parser.add_argument(
        "--pyrome",
        type=int,
        required=True,
        help="Pyrome ID (PYROME field) to sample",
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
        help="Sampling mode: dist (match distribution) or equal (fixed points/class)",
    )
    parser.add_argument(
        "--points-per-class",
        type=int,
        default=3000,
        help="Points per class (and basis for total points in dist mode)",
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
        "--prefix",
        default=None,
        help="Deprecated. Samples now always export to samples/pyrome_<id>/",
    )
    parser.add_argument(
        "--ee-project",
        default="pyregence-ee",
        help="Earth Engine project",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=333,
        help="Random seed for sampling",
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
        help="Tile scale for stratifiedSample to reduce memory usage",
    )

    args = parser.parse_args()
    export_samples(args)


if __name__ == "__main__":
    main()
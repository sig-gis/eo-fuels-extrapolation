import argparse

import ee

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


def get_pyrome_geometry(pyrome_id):
    pyromes = ee.FeatureCollection(
        "projects/pyregence-ee/assets/Pyromes_CONUS_20200206"
    )
    pyrome = pyromes.filter(ee.Filter.eq("PYROME", pyrome_id))
    return pyrome.geometry()


def build_parent_labels(year):
    fbfm40 = (
        ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fbfm40")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .first()
        .select("FBFM40")
    )

    parent = fbfm40.remap(FROM_VALS, TO_VALS).rename("FBFM40Parent")
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


def add_sample_fields(pyrome_id):
    def _add_fields(feature):
        coords = feature.geometry().coordinates()
        return feature.set({
            "pyrome_id": pyrome_id,
            "lon": coords.get(0),
            "lat": coords.get(1),
        })

    return _add_fields


def export_pyrome_samples(args, pyrome_id):
    region = get_pyrome_geometry(pyrome_id)
    parent = build_parent_labels(args.year)

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

    samples = parent.stratifiedSample(
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

    samples = (
        samples.map(add_sample_fields(pyrome_id))
        .select(["pyrome_id", "lat", "lon", "FBFM40Parent"])
    )

    description = f"pyrome_{pyrome_id}_{args.year}_{args.mode}"
    asset_id = f"{args.asset_root}/{description}"

    task = ee.batch.Export.table.toAsset(
        collection=samples,
        description=description,
        assetId=asset_id,
    )
    task.start()
    print(f"Started export: {asset_id}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export stratified pyrome samples with FBFM40 parent labels "
            "to Earth Engine assets"
        )
    )
    parser.add_argument(
        "--pyromes",
        type=int,
        nargs="+",
        required=True,
        help="Pyrome IDs (PYROME field) to sample",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="FBFM40 year",
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
        "--asset-root",
        default="projects/pyregence-ee/assets/fuels-ai",
        # default="projects/pc568-usfs-liberia/assets",
        help="Earth Engine asset folder for outputs",
    )
    parser.add_argument(
        "--ee-project",
        default="pyregence-ee",
        # default="pc568-usfs-liberia",
        help="Earth Engine project",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=333,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=4,
        help="Tile scale for stratifiedSample to reduce memory usage",
    )

    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)

    for pyrome_id in args.pyromes:
        export_pyrome_samples(args, pyrome_id)


if __name__ == "__main__":
    main()
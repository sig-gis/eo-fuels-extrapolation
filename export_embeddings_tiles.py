import argparse
import ee
import geemap
import geopandas as gpd
from google.cloud import storage
import os

GPKG_BUCKET = "geoai-fuels-tiles"
GPKG_BLOB = r"utils/tiles48km_polygons_firefactor.gpkg"
LOCAL_GPKG = r"C:\Users\edalt\RD_Fuels\eo-fuels-extrapolation\temp\tiles48km_polygons_firefactor.gpkg"

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

        tile_prefix = f"{tilenum}"

        export_if_missing(
            image=embeddings,
            desc=f"tilenum{tilenum}_aef_{args.year}",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        export_if_missing(
            image=fbfm40,
            desc=f"tilenum{tilenum}_fm40label_{args.year}",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        export_if_missing(
            image=ch,
            desc=f"tilenum{tilenum}_ch_{args.year}",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        export_if_missing(
            image=cc,
            desc=f"tilenum{tilenum}_cc_{args.year}",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        export_if_missing(
            image=cbh,
            desc=f"tilenum{tilenum}_cbh_{args.year}",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        export_if_missing(
            image=cbd,
            desc=f"tilenum{tilenum}_cbd_{args.year}",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        export_if_missing(
            image=elevation,
            desc=f"tilenum{tilenum}_elevation",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        export_if_missing(
            image=slope,
            desc=f"tilenum{tilenum}_slope",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

        export_if_missing(
            image=aspect,
            desc=f"tilenum{tilenum}_aspect",
            bucket=gcs_bucket,
            region=region,
            scale=args.scale,
            crs=args.crs,
            prefix=tile_prefix,
        )

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
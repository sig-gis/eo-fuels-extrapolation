import argparse
import os

import ee
import folium
import geemap
import geopandas as gpd
from google.cloud import storage

GPKG_BUCKET = "geoai-fuels-tiles"
GPKG_BLOB = r"utils/tiles48km_polygons_firefactor.gpkg"
LOCAL_GPKG = r"C:\Users\edalt\RD_Fuels\eo-fuels-extrapolation\temp\tiles48km_polygons_firefactor.gpkg"


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


def get_pyrome_geometry(zone_value):
    pyromes = ee.FeatureCollection(
        "projects/pyregence-ee/assets/Pyromes_CONUS_20200206"
    )
    pyrome = pyromes.filter(ee.Filter.eq("PYROME", zone_value))

    size = pyrome.size().getInfo()
    if size == 0:
        raise ValueError(
            f"No pyrome found for PYROME='{zone_value}'. Check the value."
        )

    pyrome_gdf = geemap.ee_to_gdf(pyrome)
    if pyrome_gdf.empty:
        raise ValueError(
            f"Pyrome selection for PYROME='{zone_value}' returned no features."
        )

    return pyrome_gdf


def select_tiles_intersecting_pyrome(tiles_gdf, pyrome_gdf):
    if tiles_gdf.crs is None:
        raise ValueError("Tiles GeoDataFrame has no CRS defined.")

    if pyrome_gdf.crs is None:
        raise ValueError("Pyrome GeoDataFrame has no CRS defined.")

    if pyrome_gdf.crs != tiles_gdf.crs:
        pyrome_gdf = pyrome_gdf.to_crs(tiles_gdf.crs)

    pyrome_geom = pyrome_gdf.geometry.unary_union
    intersect_mask = tiles_gdf.geometry.intersects(pyrome_geom)
    selected_tiles = tiles_gdf[intersect_mask]
    return selected_tiles


def save_overlay_map(tiles_gdf, pyrome_gdf, output_path, zone_value):
    if pyrome_gdf.crs != "EPSG:4326":
        pyrome_gdf = pyrome_gdf.to_crs("EPSG:4326")
    if tiles_gdf.crs != "EPSG:4326":
        tiles_gdf = tiles_gdf.to_crs("EPSG:4326")

    center = pyrome_gdf.geometry.unary_union.centroid
    folium_map = folium.Map(location=[center.y, center.x], zoom_start=6)

    folium.GeoJson(
        pyrome_gdf.__geo_interface__,
        name=f"Pyrome {zone_value}",
        style_function=lambda _: {
            "fillColor": "black",
            "color": "black",
            "weight": 2.5,
            "fillOpacity": 0.2,
        },
    ).add_to(folium_map)

    folium.GeoJson(
        tiles_gdf.__geo_interface__,
        name="Intersecting tiles",
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 1.0,
            "fillOpacity": 0.0,
        },
    ).add_to(folium_map)

    folium.LayerControl().add_to(folium_map)
    folium_map.save(output_path)


def main(args):
    ee.Initialize(project=args.ee_project)

    gcs_client = storage.Client()
    download_gpkg_if_needed(
        gcs_client,
        GPKG_BUCKET,
        GPKG_BLOB,
        args.local_gpkg,
    )

    tiles_gdf = gpd.read_file(args.local_gpkg)

    pyrome_gdf = get_pyrome_geometry(args.zone)
    selected_tiles = select_tiles_intersecting_pyrome(tiles_gdf, pyrome_gdf)

    tilenums = selected_tiles["tilenum"].tolist() if "tilenum" in selected_tiles.columns else []

    print(f"Pyrome '{args.zone}': {len(selected_tiles)} tiles intersecting")
    if tilenums:
        print("Tile numbers:", ", ".join(map(str, tilenums)))

    if args.map_path:
        save_overlay_map(selected_tiles, pyrome_gdf, args.map_path, args.zone)
        print(f"Saved overlay map to {args.map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count tiles fully contained within a selected pyrome"
    )

    parser.add_argument(
        "--zone",
        required=True,
        type=int,
        help="Pyrome value (PYROME field) to select",
    )

    parser.add_argument(
        "--local-gpkg",
        default=LOCAL_GPKG,
        help="Local path to tiles GPKG",
    )

    parser.add_argument(
        "--map-path",
        default=None,
        help="Optional path to save an HTML map of tiles and pyrome",
    )

    parser.add_argument(
        "--ee-project",
        default="pyregence-ee",
        help="Earth Engine project",
    )

    args = parser.parse_args()
    main(args)
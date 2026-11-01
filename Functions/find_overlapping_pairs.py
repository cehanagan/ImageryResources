#!/usr/bin/env python3
"""Find spatially overlapping mapprojected rasters using GDAL valid-data masks.

Input TSV rows: image_index<TAB>raster_path
Output TSV rows: left_index<TAB>right_index<TAB>overlap_area_m2<TAB>fraction_of_smaller_footprint
"""
import argparse
import itertools
import os
import sys

from osgeo import gdal, ogr, osr

gdal.UseExceptions()


def valid_footprint(path, target_srs, mask_resolution):
    """Polygonize a coarsened valid-data mask to avoid using the shared crop bounds."""
    src = gdal.Open(path, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    if not src.GetProjection():
        raise RuntimeError(f"Raster has no CRS: {path}")

    # Mapprojected rasters often share the same output extent. The valid-data
    # mask, rather than the full raster rectangle, identifies actual coverage.
    opts = gdal.WarpOptions(
        format="MEM",
        dstSRS=target_srs.ExportToWkt(),
        xRes=mask_resolution,
        yRes=mask_resolution,
        resampleAlg="near",
        multithread=True,
    )
    coarse = gdal.Warp("", src, options=opts)
    src = None
    if coarse is None:
        raise RuntimeError(f"Could not resample mask for: {path}")

    mask = coarse.GetRasterBand(1).GetMaskBand()
    mem_driver = ogr.GetDriverByName("Memory")
    vector_ds = mem_driver.CreateDataSource("")
    layer_srs = target_srs.Clone()
    layer = vector_ds.CreateLayer("valid", srs=layer_srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("valid", ogr.OFTInteger))

    # Mask-band values are 0 for invalid and nonzero for valid pixels.
    gdal.Polygonize(mask, mask, layer, 0, [], callback=None)

    union = None
    layer.ResetReading()
    for feature in layer:
        if feature.GetField("valid") == 0:
            continue
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom_copy = geom.Clone()
        union = geom_copy if union is None else union.Union(geom_copy)

    coarse = None
    vector_ds = None
    if union is None or union.IsEmpty():
        raise RuntimeError(f"No valid pixels found in: {path}")
    return union


def read_indexed_list(path):
    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            index, raster = line.split("\t", 1)
            result.append((int(index), raster))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-list", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--crs", default="EPSG:32617")
    p.add_argument("--mask-resolution", type=float, default=10.0)
    p.add_argument("--min-overlap-m2", type=float, default=1000.0)
    p.add_argument("--min-overlap-fraction", type=float, default=0.01)
    args = p.parse_args()

    target_srs = osr.SpatialReference()
    target_srs.SetFromUserInput(args.crs)
    if hasattr(target_srs, "SetAxisMappingStrategy"):
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    images = read_indexed_list(args.input_list)
    footprints = {}
    for index, path in images:
        try:
            footprints[index] = (path, valid_footprint(path, target_srs, args.mask_resolution))
        except Exception as exc:
            print(f"WARNING: skipping footprint for {path}: {exc}", file=sys.stderr)

    pairs = []
    for (i, (path_a, a)), (j, (path_b, b)) in itertools.combinations(footprints.items(), 2):
        if not a.Intersects(b):
            continue
        overlap = a.Intersection(b)
        if overlap is None or overlap.IsEmpty():
            continue
        area = overlap.GetArea()
        smaller = min(a.GetArea(), b.GetArea())
        fraction = area / smaller if smaller > 0 else 0.0
        if area >= args.min_overlap_m2 and fraction >= args.min_overlap_fraction:
            pairs.append((i, j, area, fraction, path_a, path_b))

    pairs.sort(key=lambda row: row[2], reverse=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("left_index\tright_index\toverlap_area_m2\tfraction_of_smaller_footprint\tleft_path\tright_path\n")
        for i, j, area, fraction, path_a, path_b in pairs:
            f.write(f"{i}\t{j}\t{area:.2f}\t{fraction:.6f}\t{path_a}\t{path_b}\n")

    print(f"Found {len(pairs)} overlapping pairs from {len(footprints)} usable images.")
    print(f"Pair list: {args.output}")


if __name__ == "__main__":
    main()

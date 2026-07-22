import numpy as np
import pandas as pd
import re
from pyproj import Transformer


def read_usgs_fsp_to_tdcalc(fsp_file, crs_out="EPSG:4326"):
    """
    Read USGS finite fault .fsp file and convert subfaults to TDCALC triangles.

    Parameters
    ----------
    fsp_file : str
        Path to USGS .fsp file

    crs_out : str
        Output projected CRS for triangle coordinates

    Returns
    -------
    top_tris : list
        List of triangles, each shape (3,3)
        Coordinates: [x(m), y(m), z(m)]

    slips : list
        List of slip vectors:
        [strike-slip, dip-slip, tensile-slip]
    """

    # --------------------------------------------------
    # Read header + subfaults
    # --------------------------------------------------

    segments = []
    rows = []

    current = None
    seg_id = -1

    with open(fsp_file) as f:

        for line in f:
            print(line)
            print(current)

            if "SEGMENT #" in line and "STRIKE" in line:
            
                seg_id += 1

                current = {
                    "strike": float(
                        re.search(r"STRIKE\s*=\s*([0-9.]+)", line).group(1)
                    ),
                    "dip": float(
                        re.search(r"DIP\s*=\s*([0-9.]+)", line).group(1)
                    )
                }

                segments.append(current)

                continue


            elif "LEN =" in line and current is not None:
            
                current["length"] = float(
                    re.search(r"LEN\s*=\s*([0-9.]+)", line).group(1)
                )

                current["width"] = float(
                    re.search(r"WID\s*=\s*([0-9.]+)", line).group(1)
                )

                continue


            elif "Dx =" in line and current is not None:
            
                current["dx"] = float(
                    re.search(r"Dx\s*=\s*([0-9.]+)", line).group(1)
                ) * 1000

                current["dz"] = float(
                    re.search(r"Dz\s*=\s*([0-9.]+)", line).group(1)
                ) * 1000

                continue


            elif re.match(r"\s*\d+\.\d+\s+-?\d+", line):

                vals = line.split()

                rows.append({
                    "lat": float(vals[0]),
                    "lon": float(vals[1]),
                    "depth": float(vals[4]),  # km
                    "slip": float(vals[5]),
                    "rake": float(vals[6]),
                    "segment": int(seg_id)
                })


    df = pd.DataFrame(rows)
    df["segment"] = df["segment"].astype(int)


    # --------------------------------------------------
    # Project lon/lat
    # --------------------------------------------------

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs_out,
        always_xy=True
    )

    df["x"], df["y"] = transformer.transform(
        df["lon"].values,
        df["lat"].values
    )

    # TDCALC convention: depth positive upward
    df["z"] = -df["depth"].values * 1000.0


    # --------------------------------------------------
    # Build triangles
    # --------------------------------------------------

    top_tris = []
    slips = []


    for _, row in df.iterrows():

        seg = segments[int(row.segment)]

        strike = np.deg2rad(seg["strike"])
        dip = np.deg2rad(seg["dip"])

        dx = seg["dx"]
        dz = seg["dz"]


        # Along strike unit vector
        strike_vec = np.array([
            np.sin(strike),
            np.cos(strike),
            0
        ])


        # Down dip direction
        dip_az = strike + np.pi/2

        dip_vec = np.array([
            np.sin(dip_az)*np.cos(dip),
            np.cos(dip_az)*np.cos(dip),
            -np.sin(dip)
        ])


        center = np.array([
            row.x,
            row.y,
            row.z
        ])


        # rectangle corners
        p1 = center - strike_vec*dx/2 - dip_vec*dz/2
        p2 = center + strike_vec*dx/2 - dip_vec*dz/2
        p3 = center + strike_vec*dx/2 + dip_vec*dz/2
        p4 = center - strike_vec*dx/2 + dip_vec*dz/2


        # split rectangle into two triangles
        tris = [
            np.vstack([p1, p2, p3]),
            np.vstack([p1, p3, p4])
        ]


        # slip decomposition
        rake = np.deg2rad(row.rake)

        ss = row.slip * np.cos(rake)
        ds = row.slip * np.sin(rake)

        slip_vec = np.array([
            ss,
            ds,
            0.0
        ])


        for tri in tris:
            top_tris.append(tri)
            slips.append(slip_vec)


    return top_tris, slips
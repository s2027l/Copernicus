#!/usr/bin/env python3
"""
Daily Copernicus Marine pull for the Flamingo Fish Finder.
Subsets sea-surface height (eddy edges) and subsurface temperature
(depth of the 20 degC isotherm = thermocline proxy) for the box around
Marina Flamingo, and writes a compact data/latest.json the map app reads.

Requires env vars COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD
(set as GitHub Actions secrets).
"""
import json, datetime, os, sys
import numpy as np
import copernicusmarine as cm
import xarray as xr

BOX = dict(minimum_longitude=-87.3, maximum_longitude=-84.9,
           minimum_latitude=9.1,   maximum_latitude=11.8)

def grid_payload(lats, lons, arr):
    a = np.where(np.isfinite(arr), np.round(arr.astype(float), 4), None)
    return {"lats": [round(float(x), 4) for x in lats],
            "lons": [round(float(x), 4) for x in lons],
            # ERDDAP-table-like row list the app already knows how to parse
            "grid": [[None if v is None else v for v in row] for row in a.tolist()]}

def main():
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=2)).isoformat()
    end = today.isoformat()

    # 1) SSH (zos) + surface currents from the global physics analysis/forecast
    phys = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["zos"],
        start_datetime=start, end_datetime=end, **BOX)
    zos = phys["zos"].isel(time=-1)

    # 2) Subsurface temperature -> depth of the 20C isotherm (thermocline proxy)
    theta = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        start_datetime=start, end_datetime=end,
        minimum_depth=0, maximum_depth=250, **BOX)
    t = theta["thetao"].isel(time=-1)          # (depth, lat, lon)
    depths = t["depth"].values
    tv = t.values
    # first depth where temp drops below 20C, linearly interpolated
    t20 = np.full(tv.shape[1:], np.nan)
    for j in range(tv.shape[1]):
        for i in range(tv.shape[2]):
            prof = tv[:, j, i]
            if not np.isfinite(prof[0]) or prof[0] < 20:
                continue
            below = np.where(prof < 20)[0]
            if len(below) == 0:
                t20[j, i] = depths[-1]; continue
            k = below[0]
            t1, t2, d1, d2 = prof[k-1], prof[k], depths[k-1], depths[k]
            t20[j, i] = d1 + (t1 - 20.0) / max(1e-6, (t1 - t2)) * (d2 - d1)

    out = {
        "updated": datetime.datetime.utcnow().isoformat() + "Z",
        "source": "Copernicus Marine (GLOBAL_ANALYSISFORECAST_PHY_001_024)",
        "ssh":  grid_payload(zos["latitude"].values, zos["longitude"].values, zos.values),
        "t20d": grid_payload(t["latitude"].values, t["longitude"].values, t20),
    }
    os.makedirs("data", exist_ok=True)
    with open("data/latest.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote data/latest.json", os.path.getsize("data/latest.json"), "bytes")

if __name__ == "__main__":
    sys.exit(main())

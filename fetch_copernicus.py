#!/usr/bin/env python3
"""Daily Copernicus Marine pull for The Magic 8-Ball — v2.1 (fault-tolerant)"""
import json, datetime, os, sys, traceback
import numpy as np
import copernicusmarine as cm

BOX = dict(minimum_longitude=-87.3, maximum_longitude=-84.9,
           minimum_latitude=9.1,   maximum_latitude=11.8)
ERRORS = []

def grid_payload(lats, lons, arr, nd=4):
    a = np.where(np.isfinite(arr), np.round(arr.astype(float), nd), None)
    return {"lats": [round(float(x), 4) for x in lats],
            "lons": [round(float(x), 4) for x in lons],
            "grid": [[None if v is None else v for v in row] for row in a.tolist()]}

def grad_norm(field, lats, lons):
    la = np.asarray(lats); lo = np.asarray(lons)
    km_lat = 111.32 * float(np.mean(np.diff(la)))
    km_lon = 111.32 * float(np.mean(np.diff(lo))) * float(np.cos(np.deg2rad(np.mean(la))))
    gy, gx = np.gradient(field, edge_order=1)
    g = np.sqrt((gy / km_lat) ** 2 + (gx / km_lon) ** 2)
    return np.clip((g - 0.008) / (0.05 - 0.008), 0, 1)

def attempt(name, fn):
    try:
        return fn()
    except Exception as e:
        msg = f"{name}: {type(e).__name__}: {e}"
        ERRORS.append(msg)
        print("SKIPPED —", msg)
        traceback.print_exc()
        return None

def main():
    today = datetime.date.today()
    start2 = (today - datetime.timedelta(days=2)).isoformat()
    start5 = (today - datetime.timedelta(days=5)).isoformat()
    end = today.isoformat()
    out = {"updated": datetime.datetime.utcnow().isoformat() + "Z",
           "source": "Copernicus Marine (GLOBAL_ANALYSISFORECAST_PHY_001_024) v2.1"}

    def do_ssh():
        phys = cm.open_dataset(dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["zos"], start_datetime=start2, end_datetime=end, **BOX)
        z = phys["zos"].isel(time=-1)
        return grid_payload(z["latitude"].values, z["longitude"].values, z.values)
    r = attempt("ssh", do_ssh)
    if r: out["ssh"] = r

    theta = attempt("thetao-open", lambda: cm.open_dataset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"], start_datetime=start5, end_datetime=end,
        minimum_depth=0, maximum_depth=250, **BOX))

    if theta is not None:
        def do_t20():
            tlast = theta["thetao"].isel(time=-1)
            depths = tlast["depth"].values; tv = tlast.values
            t20 = np.full(tv.shape[1:], np.nan)
            for j in range(tv.shape[1]):
                for i in range(tv.shape[2]):
                    prof = tv[:, j, i]
                    if not np.isfinite(prof[0]) or prof[0] < 20: continue
                    below = np.where(prof < 20)[0]
                    if len(below) == 0: t20[j, i] = depths[-1]; continue
                    k = below[0]
                    t1, t2, d1, d2 = prof[k-1], prof[k], depths[k-1], depths[k]
                    t20[j, i] = d1 + (t1 - 20.0) / max(1e-6, (t1 - t2)) * (d2 - d1)
            return grid_payload(tlast["latitude"].values, tlast["longitude"].values, t20, nd=1)
        r = attempt("t20d", do_t20)
        if r: out["t20d"] = r

        def do_persist():
            surf = theta["thetao"].isel(depth=0)
            lats = surf["latitude"].values; lons = surf["longitude"].values
            ndays = min(4, surf.sizes["time"])
            norms = [grad_norm(surf.isel(time=-1-d).values, lats, lons) for d in range(ndays)]
            return grid_payload(lats, lons, np.mean(np.stack(norms), axis=0), nd=3)
        r = attempt("frontPersist", do_persist)
        if r: out["frontPersist"] = r

    def do_cur():
        last_err = None
        for ds in ["cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
                   "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"]:
            try:
                cur = cm.open_dataset(dataset_id=ds, variables=["uo", "vo"],
                    start_datetime=start2, end_datetime=end,
                    minimum_depth=0, maximum_depth=1, **BOX)
                uo = cur["uo"].isel(time=-1).squeeze(drop=True)
                vo = cur["vo"].isel(time=-1).squeeze(drop=True)
                if uo.ndim > 2: uo = uo.isel(depth=0)
                if vo.ndim > 2: vo = vo.isel(depth=0)
                return {"lats": [round(float(x),4) for x in uo["latitude"].values],
                        "lons": [round(float(x),4) for x in uo["longitude"].values],
                        "u": grid_payload(uo["latitude"].values, uo["longitude"].values, uo.values, nd=3)["grid"],
                        "v": grid_payload(vo["latitude"].values, vo["longitude"].values, vo.values, nd=3)["grid"],
                        "dataset": ds}
            except Exception as e:
                last_err = e
        raise last_err
    r = attempt("cur", do_cur)
    if r: out["cur"] = r

    if ERRORS: out["errors"] = ERRORS
    os.makedirs("data", exist_ok=True)
    with open("data/latest.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote data/latest.json", os.path.getsize("data/latest.json"), "bytes;",
          "sections:", [k for k in out if k not in ("updated","source","errors")],
          "errors:", len(ERRORS))

if __name__ == "__main__":
    sys.exit(main())

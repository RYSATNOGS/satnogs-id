"""Container-side: run the full wrap on each batch .h5 -- extract near-vertical track,
un-correct with the obs TLE, write rffit .dat + site, run `rffit -I` against that obs's
epoch-matched catalog, and tabulate whether the TRUE object (from the filename) is the
top match. This is the 4-pass batch validation of the Milestone-0 result."""

import glob
import os

from satnogs_id.id.dat import build_dat, site_line
from satnogs_id.id.identify import run_rffit_identify
from satnogs_id.shared.waterfall import TrackParams, load_waterfall


def make_dat(h5path, siteid, datpath):
    """Extract the track and write rffit's .dat + site line; return the point count. Same
    extraction as the shipped wrap, but min_points=1 keeps this diagnostic's gate-less behaviour."""
    wf = load_waterfall(h5path)
    n = build_dat(wf, siteid, datpath, TrackParams(min_points=1))
    with open("/opt/strf/data/sites.txt", "a", encoding="utf-8") as g:
        g.write(site_line(wf, siteid))
    return n


results = []
for k, h5file in enumerate(sorted(glob.glob("/data/batch/*.h5"))):
    base = os.path.basename(h5file)
    oid = base.split("_")[0][3:]
    truenorad = int(base.split("_n")[1].split("_")[0])
    site = 9001 + k
    datp = f"/data/strf/{oid}.dat"
    catp = f"/data/strf/soup_{oid}.tle"
    npts = make_dat(h5file, site, datp)
    res = run_rffit_identify(datp, catp, site)
    bconf = next((rms for rms, n in res.ranking if n != truenorad), None)
    results.append(
        (
            oid,
            truenorad,
            res.predicted,
            res.rank_of(truenorad),
            res.rms_of(truenorad),
            bconf,
            npts,
        )
    )

print(
    f"\n{'obs':>9} {'true':>6} {'predicted':>9} {'rank':>4} "
    f"{'true_rms':>8} {'conf_rms':>8} {'pts':>4}  result"
)
ncorrect = 0
for oid, tn, pr, rk, trms, brms, npts in results:
    ok = pr == tn
    ncorrect += ok
    print(
        f"{oid:>9} {tn:>6} {str(pr):>9} {str(rk):>4} "
        f"{(trms or 0):>8.3f} {(brms or 0):>8.3f} {npts:>4}  "
        f"{'CORRECT' if ok else 'WRONG'}"
    )
print(f"\n=== {ncorrect}/{len(results)} passes correctly identified ===")

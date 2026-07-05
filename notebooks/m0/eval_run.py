"""Container-side scaled eval: run the strf/rffit wrap over every pass in /data/eval/,
and report honest metrics -- top-1 identification accuracy with a Wilson 95% CI, the rank
distribution of the true object, the margin distribution, and a per-object breakdown."""

import glob
import math
import os

import numpy as np

from satnogs_id.data.build import CLUSTERS
from satnogs_id.id.dat import build_dat, site_line
from satnogs_id.id.identify import run_rffit_identify
from satnogs_id.shared.waterfall import load_waterfall

NAME = CLUSTERS["geoscan"]["truth"]


def make_dat(h5path, siteid, datpath):
    """Extract the track and write rffit's .dat + site line; return the point count. build_dat
    returns 0 when <10 usable points, in which case the caller skips the site line and the pass."""
    wf = load_waterfall(h5path)
    n = build_dat(wf, siteid, datpath)
    if n == 0:
        return 0
    with open("/opt/strf/data/sites.txt", "a", encoding="utf-8") as g:
        g.write(site_line(wf, siteid))
    return n


def wilson(k, n, z=1.96):
    """Wilson score confidence interval (lo, hi) for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


rows_out, by_obj, ranks, margins = [], {}, [], []
for idx, h5file in enumerate(sorted(glob.glob("/data/eval/*.h5"))):
    base = os.path.basename(h5file)
    oid = base.split("_")[0][3:]
    truenorad = int(base.split("_n")[1].split("_")[0])
    site = 9001 + (idx % 900)
    datp = f"/data/eval/{oid}.dat"
    catp = f"/data/eval/soup_{oid}.tle"
    if not os.path.exists(catp):
        continue
    npts = make_dat(h5file, site, datp)
    if npts < 10:
        rows_out.append((oid, truenorad, None, None, npts))
        continue
    res = run_rffit_identify(datp, catp, site)
    if not res.ranking:
        rows_out.append((oid, truenorad, None, None, npts))
        continue
    pred = res.predicted
    rank = res.rank_of(truenorad)
    trms = res.rms_of(truenorad)
    bconf = next((rms for rms, n in res.ranking if n != truenorad), None)
    correct = pred == truenorad
    rows_out.append((oid, truenorad, pred, rank, npts))
    by_obj.setdefault(truenorad, [0, 0])
    by_obj[truenorad][1] += 1
    by_obj[truenorad][0] += correct
    if rank:
        ranks.append(rank)
    if correct and trms is not None and bconf is not None:
        margins.append(bconf - trms)

scored = [row for row in rows_out if row[2] is not None]
ncorrect = sum(1 for row in scored if row[2] == row[1])
N = len(scored)
lo, hi = wilson(ncorrect, N)
print(
    f"=== Scaled eval: Geoscan cluster, {len(rows_out)} passes "
    f"({N} scored, {len(rows_out) - N} unusable) ==="
)
print(
    f"TOP-1 ACCURACY: {ncorrect}/{N} = {100 * ncorrect / max(N, 1):.1f}%  "
    f"(95% Wilson CI {100 * lo:.0f}-{100 * hi:.0f}%)"
)
print(
    "true-object rank distribution: "
    + ", ".join(f"rank{r}:{ranks.count(r)}" for r in sorted(set(ranks)))
)
if margins:
    a = np.array(margins)
    print(
        f"margin over best confuser (correct cases): median {np.median(a):.2f} kHz, "
        f"min {a.min():.2f}, max {a.max():.2f}"
    )
print("per-object top-1:")
for norad in sorted(by_obj):
    n_ok, n_tot = by_obj[norad]
    print(f"  {NAME.get(norad, norad)} ({norad}): {n_ok}/{n_tot}")

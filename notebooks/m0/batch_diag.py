"""Empirical correction test across MANY strong passes from DIFFERENT stations.
For each .h5: matched-filter profile vs carrier offset for two hypotheses --
  vertical line (constant offset)  -> Doppler-CORRECTED
  embedded-TLE Doppler curve       -> RAW/UNCORRECTED
Score per offset = mean of the top 25% along-line power (robust to intermittent
beacons). Prominence = (peak - median)/std of that profile. The hypothesis with the
sharper, more prominent peak reveals the signal's shape. No conclusion is asserted
here -- the per-obs table is what we read."""

import glob
from datetime import timedelta

import numpy as np

from satnogs_id.shared import geometry
from satnogs_id.shared.waterfall import axes, load_waterfall, observation_id, window_max


def topmean(x, frac=0.25):
    """Mean of the top ``frac`` fraction of ``x`` (robust to intermittent beacons)."""
    k = max(1, int(len(x) * frac))
    return float(np.mean(np.sort(x)[-k:]))


def _prominence(p):
    """Peak prominence of a profile: (max - median) / std."""
    return (p.max() - np.median(p)) / (np.std(p) + 1e-9)


def analyze(path):
    """Vertical-line vs Doppler-curve matched-filter prominences for one pass."""
    wf = load_waterfall(path)
    f0, freqax, relt, db, n_time, n_freq = axes(wf)
    times = [wf.start + timedelta(seconds=float(r)) for r in relt]
    pred = geometry.doppler_offset_hz(
        f0, geometry.range_rate_km_s(wf.tle[1], wf.tle[2], wf.station, times)
    )
    grid = np.arange(-12000, 12000, 50.0)

    def profile(curve):
        out = np.empty(len(grid))
        for i, df in enumerate(grid):
            idx = np.clip(np.searchsorted(freqax, curve + df), 0, n_freq - 1)
            out[i] = topmean(window_max(db, idx, 3))
        return out

    vp, dp = profile(np.zeros(n_time)), profile(pred)
    return (
        observation_id(path),
        _prominence(vp),
        _prominence(dp),
        grid[int(np.argmax(dp))],
        wf.tle[0].strip(),
    )


print(
    f"{'obs':>9} {'object':>11} {'vert_prom':>9} {'dopp_prom':>9} {'dopp_off':>8}  reading"
)
rows = []
for h5file in sorted(glob.glob("/data/batch/*.h5")):
    oid, vprom, dprom, doff, obj = analyze(h5file)
    reading = (
        "UNCORRECTED (curve)"
        if dprom > vprom * 1.3
        else ("CORRECTED (vertical)" if vprom > dprom * 1.3 else "ambiguous")
    )
    print(f"{oid:>9} {obj:>11} {vprom:>9.1f} {dprom:>9.1f} {doff:>+8.0f}  {reading}")
    rows.append(reading)
print("\nper-obs readings:", rows)

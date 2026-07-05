"""M0 geometry validation: extract the measured Doppler track from a SatNOGS .h5
waterfall and overlay skyfield's predicted Doppler from the embedded per-obs TLE.
If they agree, the TEME->topocentric->range-rate geometry is trustworthy (the §5
non-negotiable gate) before any scoring. Runs in the python:3.14-slim container."""

from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

from satnogs_id.shared import geometry
from satnogs_id.shared.waterfall import axes, load_waterfall, observation_id

plt.switch_backend("Agg")

PATH = "/data/sample.h5"
obs_id = observation_id(PATH)
wf = load_waterfall(PATH)
f0, freqax, relt, db, n_time, _ = axes(wf)
l1, l2 = wf.tle[1].strip(), wf.tle[2].strip()
start_dt = wf.start
print(
    f"waterfall {db.shape}  freqax[{freqax.min():.0f}..{freqax.max():.0f}] "
    f"step={freqax[1] - freqax[0]:.2f}"
)
print(f"start={start_dt.isoformat()}  reltime span={relt[-1] - relt[0]:.1f}s")

# --- extract measured track: peak frequency bin per time row, dB-normalised per bin ---
# FIX 1 (structured noise): background subtraction. A fixed-frequency interference
# line is constant in time -> subtract each bin's time-median to remove it; the
# Doppler-sweeping satellite is NOT constant in any bin, so it survives.
db_sub = db - np.median(db, axis=0, keepdims=True)
peak = np.argmax(db_sub, axis=1)
peakp = db_sub[np.arange(n_time), peak]
base = np.median(db_sub, axis=1)
snr = peakp - base
keep = snr > (np.median(snr) + 2.0 * np.std(snr))  # signal-present rows
mt = relt[keep]  # rel seconds
mf = freqax[peak[keep]]  # measured offset (Hz)
print(
    f"signal-present rows: {keep.sum()}/{n_time}; "
    f"measured offset range [{mf.min():.0f}..{mf.max():.0f}]"
)

# --- predicted Doppler from the embedded TLE (shared skyfield geometry) ---
dts = [start_dt + timedelta(seconds=float(rt)) for rt in relt]
rrate = geometry.range_rate_km_s(l1, l2, wf.station, dts)  # km/s radial (range rate)
pred_off_hz = geometry.doppler_offset_hz(f0, rrate)  # Doppler offset, Hz
rng = geometry.range_km(l1, l2, wf.station, dts)
print(
    f"predicted Doppler offset range "
    f"[{pred_off_hz.min():.0f}..{pred_off_hz.max():.0f}] Hz; min range {rng.min():.0f} km"
)

# --- overlay (test both Hz and kHz interpretations of the freq axis) ---
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
trel = relt
for a, (scl, lab) in zip(ax, [(1.0, "freq axis as Hz"), (1000.0, "freq axis as kHz")]):
    a.scatter(mt, mf * scl, s=6, c="tab:blue", label="measured peak (track)")
    a.plot(
        trel, pred_off_hz, c="tab:red", lw=2, label="predicted Doppler (embedded TLE)"
    )
    a.set_title(lab)
    a.set_xlabel("time since start (s)")
    a.set_ylabel(f"offset from {f0 / 1e6:.4f} MHz (Hz)")
    a.legend()
    a.grid(alpha=0.3)
fig.suptitle(f"Geoscan-1 / NORAD 64880 obs {obs_id} — measured vs predicted Doppler")
fig.tight_layout()
fig.savefig("/data/overlay.png", dpi=90)
print("saved /data/overlay.png")

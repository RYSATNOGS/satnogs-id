"""M0 matched-filter detector: instead of extracting a noisy per-row track, integrate
the (background-subtracted) waterfall power ALONG a candidate's predicted Doppler curve,
searching the unknown carrier offset. Demonstrates the two layered noise fixes:
  FIX 1 background subtraction  -> removes fixed-frequency interference lines
  FIX 2 matched filter          -> robust to broadband noise (integrate along the curve)
Compares the Doppler-curve score against a flat zero-Doppler line: if the signal really
follows the Doppler shape, the curve wins (and that gap is the basis of separability)."""

from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

from satnogs_id.shared import geometry
from satnogs_id.shared.waterfall import axes, load_waterfall, window_max

plt.switch_backend("Agg")

wf = load_waterfall("/data/sample.h5")
f0, freqax, relt, db, n_time, n_freq = axes(wf)
db = db - np.median(db, axis=0, keepdims=True)  # FIX 1: background subtraction
times = [wf.start + timedelta(seconds=float(r)) for r in relt]


def predicted_offset(l1, l2):
    """Predicted Doppler offset (Hz) over the pass for a candidate's TLE lines."""
    rr = geometry.range_rate_km_s(l1, l2, wf.station, times)
    return geometry.doppler_offset_hz(f0, rr)  # Hz


pred = predicted_offset(wf.tle[1], wf.tle[2])

# FIX 2: matched filter. For a carrier offset df, take the max power in a small window
# around the curve at each time, average over the pass.
df_grid = np.arange(-10000, 10000, 50.0)


def mf(curve, win=2):
    """Matched-filter score vs trial carrier offset: mean windowed power along ``curve``."""
    out = np.empty(len(df_grid))
    for i, df in enumerate(df_grid):
        idx = np.clip(np.searchsorted(freqax, curve + df), 0, n_freq - 1)
        out[i] = window_max(db, idx, win).mean()
    return out


s_dopp = mf(pred)  # Geoscan-1's real Doppler curve
s_flat = mf(np.zeros(n_time))  # a flat zero-Doppler line (null hypothesis)
best = df_grid[np.argmax(s_dopp)]
print(
    f"Doppler-curve matched-filter peak: {s_dopp.max():.3f} dB at carrier offset {best:+.0f} Hz"
)
print(f"Flat zero-Doppler line peak:       {s_flat.max():.3f} dB")
SHAPE = (
    "signal follows the Doppler curve (uncorrected)"
    if s_dopp.max() > s_flat.max() + 0.2
    else "signal ~flat -> waterfall is Doppler-CORRECTED"
)
print(f"--> Doppler shape advantage: {s_dopp.max() - s_flat.max():+.3f} dB  ({SHAPE})")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].plot(df_grid, s_dopp, label="Geoscan-1 Doppler curve")
ax[0].plot(df_grid, s_flat, "--", label="flat zero-Doppler line")
ax[0].axvline(best, color="k", lw=0.7)
ax[0].set_xlabel("carrier offset (Hz)")
ax[0].set_ylabel("mean power along curve (dB)")
ax[0].set_title("matched-filter score vs carrier offset")
ax[0].legend()
ax[0].grid(alpha=0.3)
ds = max(1, n_time // 1200)
img = db[::ds]
ax[1].imshow(
    img,
    aspect="auto",
    extent=[freqax[0], freqax[-1], relt[::ds][-1], relt[::ds][0]],
    cmap="viridis",
    vmin=np.percentile(img, 60),
    vmax=np.percentile(img, 99.7),
)
ax[1].plot(pred + best, relt, "r", lw=1.2, label="best-fit Geoscan-1 curve")
ax[1].set_xlim(-15000, 15000)
ax[1].set_xlabel(f"offset from {f0 / 1e6:.4f} MHz (Hz)")
ax[1].set_ylabel("time (s)")
ax[1].set_title("cleaned waterfall + matched curve")
ax[1].legend(loc="upper right")
fig.tight_layout()
fig.savefig("/data/matched.png", dpi=90)
print("saved /data/matched.png")

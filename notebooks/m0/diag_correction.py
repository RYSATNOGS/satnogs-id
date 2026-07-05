"""Decide quantitatively whether the waterfall is Doppler-corrected: compare a matched
filter for a VERTICAL line (constant offset = corrected) vs the embedded-TLE DOPPLER CURVE
(raw), both on the same raw dB. Whichever 'excess above a random line' is larger reveals the
signal's shape. Also emits a high-contrast image."""

from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

from satnogs_id.shared import geometry
from satnogs_id.shared.waterfall import axes, load_waterfall, observation_id, window_max

plt.switch_backend("Agg")

PATH = "/data/good.h5"
obs_id = observation_id(PATH)
wf = load_waterfall(PATH)
f0, freqax, relt, db, n_time, n_freq = axes(wf)
db_bg = db - np.median(db, axis=0, keepdims=True)

times = [wf.start + timedelta(seconds=float(r)) for r in relt]
pred = geometry.doppler_offset_hz(
    f0, geometry.range_rate_km_s(wf.tle[1], wf.tle[2], wf.station, times)
)

grid = np.arange(-12000, 12000, 40.0)


def score(curve, win=3):
    """Matched-filter mean power along ``curve`` swept over the trial-offset ``grid``."""
    out = np.empty(len(grid))
    for i, df in enumerate(grid):
        idx = np.clip(np.searchsorted(freqax, curve + df), 0, n_freq - 1)
        out[i] = window_max(db, idx, win).mean()
    return out


vline = score(np.zeros(n_time))  # vertical (corrected) hypothesis on raw
dcurv = score(pred)  # doppler-curve (raw) hypothesis on raw
v_exc = vline.max() - np.median(vline)
d_exc = dcurv.max() - np.median(dcurv)
print(
    f"VERTICAL line  : peak {vline.max():.3f} at {grid[np.argmax(vline)]:+5.0f} Hz "
    f"| excess above median {v_exc:.3f}"
)
print(
    f"DOPPLER curve  : peak {dcurv.max():.3f} at {grid[np.argmax(dcurv)]:+5.0f} Hz "
    f"| excess above median {d_exc:.3f}"
)
verdict = (
    "DOPPLER-CORRECTED (signal is a vertical line)"
    if v_exc > d_exc * 1.3
    else (
        "RAW/UNCORRECTED (signal follows the Doppler curve)"
        if d_exc > v_exc * 1.3
        else "AMBIGUOUS"
    )
)
print("VERDICT:", verdict)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ds = max(1, n_time // 1400)
for a, (panel, title, ov) in zip(
    ax,
    [
        (db[::ds], "RAW waterfall (high contrast)", None),
        (db_bg[::ds], "background-subtracted + predicted Doppler", pred),
    ],
):
    a.imshow(
        panel,
        aspect="auto",
        extent=[freqax[0], freqax[-1], relt[::ds][-1], relt[::ds][0]],
        cmap="inferno",
        vmin=np.percentile(panel, 90),
        vmax=np.percentile(panel, 99.9),
    )
    if ov is not None:
        a.plot(ov, relt, "c", lw=1.2, label="predicted Doppler")
        a.legend(loc="upper right")
    a.set_xlim(-12000, 12000)
    a.set_xlabel(f"offset from {f0 / 1e6:.4f} MHz (Hz)")
    a.set_ylabel("time (s)")
    a.set_title(title)
fig.suptitle(f"Geoscan-2 obs {obs_id} — correction diagnosis ({verdict})")
fig.tight_layout()
fig.savefig("/data/diag.png", dpi=100)
print("saved /data/diag.png")

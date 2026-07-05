"""Visual confirmation: is the SatNOGS waterfall Doppler-corrected? Show a strong,
vetted-good pass WITHOUT background subtraction and overlay the embedded-TLE predicted
Doppler curve. Vertical signal line + red curve sweeping away from it => corrected."""

from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

from satnogs_id.shared import geometry
from satnogs_id.shared.waterfall import axes, load_waterfall, observation_id

plt.switch_backend("Agg")

PATH = "/data/good.h5"
obs_id = observation_id(PATH)
wf = load_waterfall(PATH)
f0, freqax, relt, db, n_time, _ = axes(wf)  # per-bin normalized; NO time-median subtraction
print("obs", obs_id, "|", wf.tle[0], "| pass", round(relt[-1] - relt[0]), "s")

times = [wf.start + timedelta(seconds=float(rt)) for rt in relt]
pred = geometry.doppler_offset_hz(
    f0, geometry.range_rate_km_s(wf.tle[1], wf.tle[2], wf.station, times)
)
rng = geometry.range_km(wf.tle[1], wf.tle[2], wf.station, times)
print(
    f"predicted Doppler [{pred.min():.0f}..{pred.max():.0f}] Hz; min range {rng.min():.0f} km"
)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ds = max(1, n_time // 1400)
img = db[::ds]
ext = [freqax[0], freqax[-1], relt[::ds][-1], relt[::ds][0]]
for a, (title, overlay) in zip(
    ax,
    [
        ("raw waterfall (as stored)", False),
        ("raw waterfall + embedded-TLE predicted Doppler", True),
    ],
):
    a.imshow(
        img,
        aspect="auto",
        extent=ext,
        cmap="viridis",
        vmin=np.percentile(img, 70),
        vmax=np.percentile(img, 99.8),
    )
    if overlay:
        a.plot(pred, relt, "r", lw=1.2, label="predicted Doppler (embedded TLE)")
        a.legend(loc="upper right")
    a.set_xlim(-15000, 15000)
    a.set_xlabel(f"offset from {f0 / 1e6:.4f} MHz (Hz)")
    a.set_ylabel("time (s)")
    a.set_title(title)
fig.suptitle(f"Geoscan-2 obs {obs_id} — 85deg pass, vetted with-signal")
fig.tight_layout()
fig.savefig("/data/good_viz.png", dpi=95)
print("saved /data/good_viz.png")

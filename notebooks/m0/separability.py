"""M0 separability test (grounded in the verified fact that the waterfall is Doppler-
corrected with the embedded TLE T0). The stored signal sits at s(t)=carrier+(true-T0)
Doppler. For each candidate c, the expected track if c were the emitter is
track_c(t)=df+(c_doppler-T0_doppler). Matched-filter that track against the stored
waterfall. The TRUE object (c=T0=Geoscan-2) -> flat track -> best fit; near-identical
siblings -> a sloped residual -> worse fit. Margin(true - best sibling) = separability."""

import json
from datetime import timedelta

import numpy as np

from satnogs_id.shared import geometry
from satnogs_id.shared.waterfall import axes, load_waterfall

wf = load_waterfall("/data/good.h5")
f0, freqax, relt, db, n_time, n_freq = axes(wf)
times = [wf.start + timedelta(seconds=float(r)) for r in relt]


def dopp(l1, l2):
    """Predicted Doppler offset (Hz) over the pass for a candidate TLE."""
    rr = geometry.range_rate_km_s(l1, l2, wf.station, times)
    return geometry.doppler_offset_hz(f0, rr)


T0 = dopp(wf.tle[1], wf.tle[2])  # correction TLE actually applied (Geoscan-2)
with open("/data/soup_tles.json", encoding="utf-8") as _f:
    soup = json.load(_f)
dfbin = float(freqax[1] - freqax[0])


def score(diff_track):
    """Incoherent integration: shift each row by -(candidate differential) so that, IF the
    candidate is the emitter, its signal aligns vertically; sum over time. True object ->
    sharp peak; near-identical siblings (different phase) -> smeared -> no peak. Returns the
    aligned signal's peak prominence (SNR)."""
    sh = np.round(diff_track / dfbin).astype(int)
    integ = np.zeros(n_freq)
    cnt = np.zeros(n_freq)
    base = np.arange(n_freq)
    for t in range(n_time):
        src = base + sh[t]
        v = (src >= 0) & (src < n_freq)
        integ[v] += db[t, src[v]]
        cnt[v] += 1
    integ = integ / np.maximum(cnt, 1)
    integ -= np.median(integ)
    return float(integ.max() / (np.std(integ) + 1e-9))


GEOSCANS = {64879, 64880, 64890, 64891, 64892, 64893}
res = []
for n, d in soup.items():
    track = dopp(d["tle1"], d["tle2"]) - T0  # differential vs the applied correction
    res.append(
        (
            score(track),
            int(n),
            d.get("tle0", "").strip(),
            d.get("days_off"),
            float(np.ptp(track)),
        )
    )
res.sort(reverse=True)
print(
    f"{'rk':>3} {'norad':>6} {'object':>20} {'score':>7} {'diff_span_Hz':>12} {'epoch_d':>7}"
)
for i, (s, n, nm, doff, span) in enumerate(res):
    tag = (
        "  <-- TRUE (Geoscan-2)"
        if n == 64890
        else ("  [sibling]" if n in GEOSCANS else "")
    )
    print(f"{i + 1:>3} {n:>6} {nm:>20} {s:>7.3f} {span:>12.0f} {doff:>7}{tag}")
true = next(r for r in res if r[1] == 64890)
sibs = [r for r in res if r[1] in GEOSCANS and r[1] != 64890]
bestsib = max(sibs)
print(f"\nTRUE Geoscan-2: score {true[0]:.3f}, rank {res.index(true) + 1}/{len(res)}")
print(
    f"best sibling  : norad {bestsib[1]} score {bestsib[0]:.3f}, "
    f"differential span over pass {bestsib[4]:.0f} Hz"
)
print(f"SEPARABILITY MARGIN (true - best sibling): {true[0] - bestsib[0]:+.3f} dB")
print(f"sibling differential spans (Hz): {sorted(int(r[4]) for r in sibs)}")

"""Host-side: build a larger eval set for the Geoscan cluster -- for each of the 6
near-identical Geoscan objects, download up to K strong (with-signal, .h5) passes from
distinct stations, and write each pass's epoch-matched candidate catalog. -> /scratch/eval/."""

import glob
import json
import os
import time
import urllib.error
import urllib.request

from satnogs_id.data.build import CLUSTERS, write_soup_catalog

NET = "https://network.satnogs.org/api"
DB = "https://db.satnogs.org/api"
SC = (
    "/private/tmp/claude-501/-Users-ryan-GitHub-satnogs-id/"
    "e9fc3766-e6dd-4352-9d46-489818e4c3a6/scratchpad"
)
KEY = None
with open("/Users/ryan/GitHub/satnogs-id/.env", encoding="utf-8") as _env:
    for line in _env:
        if line.startswith("satnogs_db_api_key="):
            KEY = line.strip().split("=", 1)[1]


def getj(url, auth=False, tries=6):
    """GET a URL and parse JSON, retrying with backoff on HTTP 429 (rate limiting)."""
    h = {"Accept": "application/json"}
    if auth:
        h["Authorization"] = f"Token {KEY}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=h), timeout=30
            ) as resp:
                r = json.load(resp)
            time.sleep(0.4)
            return r
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1:
                time.sleep(8 * (k + 1))
                continue
            raise
    raise RuntimeError(f"gave up after {tries} tries (rate limited): {url}")


GEOSCANS = CLUSTERS["geoscan"]["truth"]
soup = CLUSTERS["geoscan"]["soup"]
K = 12
os.makedirs(SC + "/eval", exist_ok=True)
CACHE = SC + "/eval/cand_obs.json"
if os.path.exists(CACHE):
    with open(CACHE, encoding="utf-8") as _c:
        cand_obs = {int(k): v for k, v in json.load(_c).items()}
else:
    cand_obs = {
        n: getj(f"{NET}/observations/?norad_cat_id={n}&format=json") for n in soup
    }
    with open(CACHE, "w", encoding="utf-8") as _c:
        json.dump({str(k): v for k, v in cand_obs.items()}, _c)
downloaded = []
existing = {os.path.basename(p).split("_")[0][3:] for p in glob.glob(SC + "/eval/*.h5")}
for norad, name in GEOSCANS.items():
    cands = sorted(
        [
            o
            for o in cand_obs[norad]
            if o.get("waterfall_status") == "with-signal"
            and (o.get("max_altitude") or 0) >= 25
        ],
        key=lambda o: -(o.get("max_altitude") or 0),
    )
    got = 0
    for o in cands:
        if got >= K:
            break
        oid = o["id"]
        st = o.get("ground_station")
        fn = f"{SC}/eval/obs{oid}_n{norad}_st{st}.h5"
        if str(oid) in existing or os.path.exists(fn):
            downloaded.append((oid, norad))
            got += 1
            continue
        rows = getj(f"{DB}/artifacts/?network_obs_id={oid}&format=json", auth=True)
        rows = rows if isinstance(rows, list) else rows.get("results", [])
        h5url = next((a["artifact_file"] for a in rows if a.get("artifact_file")), None)
        if not h5url:
            continue
        try:
            urllib.request.urlretrieve(h5url, fn)
        except urllib.error.HTTPError:
            req = urllib.request.Request(h5url, headers={"Authorization": f"Token {KEY}"})
            with urllib.request.urlopen(req, timeout=90) as stream, open(fn, "wb") as out:
                out.write(stream.read())
        downloaded.append((oid, norad))
        got += 1
    print(f"{name} ({norad}): {got} passes")
# per-obs epoch-matched catalogs
for oid, norad in downloaded:
    tdate = getj(f"{NET}/observations/?id={oid}&format=json")[0]["start"][:10]
    write_soup_catalog(f"{SC}/eval/soup_{oid}.tle", soup, cand_obs, tdate)
print(
    f"\neval set: {len(downloaded)} passes across {len(GEOSCANS)} near-identical objects"
)

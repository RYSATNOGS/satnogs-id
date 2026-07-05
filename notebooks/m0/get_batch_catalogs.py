"""Host-side: for each batch .h5, build a candidate catalog (the 2025-155 soup,
NORAD 64876-64895) with each candidate's TLE epoch-matched to THAT observation's date,
pulled from SatNOGS per-obs TLEs. Writes /scratch/strf/soup_<obsid>.tle."""

import glob
import json
import os
import urllib.request

from satnogs_id.data.build import CLUSTERS, write_soup_catalog

NET = "https://network.satnogs.org/api"
SC = (
    "/private/tmp/claude-501/-Users-ryan-GitHub-satnogs-id/"
    "e9fc3766-e6dd-4352-9d46-489818e4c3a6/scratchpad"
)


def getj(url):
    """GET a URL and parse its JSON body."""
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"Accept": "application/json"}),
        timeout=30,
    ) as resp:
        return json.load(resp)


soup = CLUSTERS["geoscan"]["soup"]
cand_obs = {n: getj(f"{NET}/observations/?norad_cat_id={n}&format=json") for n in soup}
os.makedirs(SC + "/strf", exist_ok=True)
for p in sorted(glob.glob(SC + "/batch/*.h5")):
    oid = os.path.basename(p).split("_")[0][3:]
    tdate = getj(f"{NET}/observations/?id={oid}&format=json")[0]["start"][:10]
    n_match = write_soup_catalog(f"{SC}/strf/soup_{oid}.tle", soup, cand_obs, tdate)
    print(f"obs {oid} date {tdate}: soup_{oid}.tle ({n_match} candidates)")

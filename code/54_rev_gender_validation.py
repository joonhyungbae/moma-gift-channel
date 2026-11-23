# -*- coding: utf-8 -*-
"""
Revision analysis A4b (reviewer 2 comment 3): independent check of MoMA's
artist-gender field.

For every artist in the donor-identified gift sample that MoMA links to a
Wikidata item, the script reads Wikidata's 'sex or gender' statement (P21) and
compares it with MoMA's Gender field. Responses are cached in
data/wikidata_gender_cache.csv so the comparison can be re-run offline.

Outputs (output/): rev_gender_validation.csv, rev_gender_validation_disagreements.csv
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from coding import gender_token
from config import PROJECT_ROOT
from rev_utils import artist_frame, artist_names, constituents, load_gifts, out, pct

CACHE = os.path.join(PROJECT_ROOT, "data", "wikidata_gender_cache.csv")
API = "https://www.wikidata.org/w/api.php"
P21 = {"Q6581097": "M", "Q6581072": "F", "Q2449503": "M", "Q1052281": "F", "Q48270": "NB"}


def fetch(qids):
    params = {"action": "wbgetentities", "ids": "|".join(qids), "props": "claims", "format": "json"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "collection-gender-check/1.0 (research script)"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            wait = int(e.headers.get("Retry-After", 0) or 0) or 20 * (attempt + 1)
            time.sleep(min(wait, 120))
    rows = []
    for q, ent in data.get("entities", {}).items():
        vals = []
        for claim in ent.get("claims", {}).get("P21", []):
            try:
                vals.append(claim["mainsnak"]["datavalue"]["value"]["id"])
            except (KeyError, TypeError):
                pass
        rows.append({"qid": q, "p21": "|".join(vals)})
    return rows


def main():
    gifts = load_gifts()
    cons = constituents(gifts, artist_names())
    arts = artist_frame(cons)
    meta = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "Artists_full.csv"), dtype={"ConstituentID": str})
    meta = meta.set_index("ConstituentID")
    arts["qid"] = meta["Wiki QID"].reindex(arts.index)
    arts["moma_raw"] = meta["Gender"].reindex(arts.index)
    linked = arts[arts["qid"].notna()]

    cache = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=["qid", "p21"])
    todo = [q for q in linked["qid"].unique() if q not in set(cache["qid"])]
    if todo and "--offline" not in sys.argv:
        for i in range(0, len(todo), 50):
            rows = fetch(todo[i:i + 50])
            cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            cache.to_csv(CACHE, index=False)  # incremental, so an interrupted run resumes
            time.sleep(2.0)
    wd = cache.set_index("qid")["p21"].fillna("")
    def wd_code(v):
        codes = {P21.get(x) for x in str(v).split("|") if x}
        codes.discard(None)
        return next(iter(codes)) if len(codes) == 1 else ("" if not codes else "multiple")
    linked = linked.assign(wd=linked["qid"].map(wd).map(wd_code))

    both = linked[linked["gender"].isin(["F", "M"]) & linked["wd"].isin(["F", "M"])]
    agree = (both["gender"] == both["wd"])
    summary = [
        {"metric": "distinct artists in gift sample", "value": len(arts)},
        {"metric": "with MoMA binary gender code", "value": int(arts["gender"].isin(["F", "M"]).sum())},
        {"metric": "linked to a Wikidata item", "value": len(linked)},
        {"metric": "linked and Wikidata records P21", "value": int(linked["wd"].isin(["F", "M", "NB", "multiple"]).sum())},
        {"metric": "both sources give a binary code", "value": len(both)},
        {"metric": "agreement (n)", "value": int(agree.sum())},
        {"metric": "agreement (%)", "value": round(pct(agree.sum(), len(both)), 2)},
        {"metric": "MoMA F / Wikidata M", "value": int(((both["gender"] == "F") & (both["wd"] == "M")).sum())},
        {"metric": "MoMA M / Wikidata F", "value": int(((both["gender"] == "M") & (both["wd"] == "F")).sum())},
        {"metric": "MoMA uncoded, Wikidata F", "value": int((~linked["gender"].isin(["F", "M"]) & (linked["wd"] == "F")).sum())},
        {"metric": "MoMA uncoded, Wikidata M", "value": int((~linked["gender"].isin(["F", "M"]) & (linked["wd"] == "M")).sum())},
        {"metric": "female share, MoMA code (linked, both coded) %", "value": round(pct((both["gender"] == "F").sum(), len(both)), 2)},
        {"metric": "female share, Wikidata code (linked, both coded) %", "value": round(pct((both["wd"] == "F").sum(), len(both)), 2)},
        {"metric": "female share among artists not linked to Wikidata (MoMA code) %",
         "value": round(pct((arts.loc[arts["qid"].isna(), "gender"] == "F").sum(),
                            arts.loc[arts["qid"].isna(), "gender"].isin(["F", "M"]).sum()), 2)},
    ]
    pd.DataFrame(summary).to_csv(out("rev_gender_validation.csv"), index=False)
    both[~agree][["name", "moma_raw", "wd", "qid", "n_records"]].to_csv(out("rev_gender_validation_disagreements.csv"))
    print(pd.DataFrame(summary).to_string())
    print(both[~agree][["name", "moma_raw", "wd", "qid", "n_records"]].to_string())


if __name__ == "__main__":
    main()

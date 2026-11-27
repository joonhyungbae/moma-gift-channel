# -*- coding: utf-8 -*-
"""
Compare the parser with an independent reading of the 300 audit credit lines
(not part of run_all.py).

Inputs
  audit/audit_ai.csv    a blind reading, by a language model given only
                                   the published rules (supplement S1), of the 300
                                   credit lines script 58 drew on 2026-11-27, before
                                   the two parser fixes it led to. The ObjectIDs in
                                   this file are the fixed audit set.
  the current parser (rev_utils.load_gifts), applied to the same records
Output
  audit/parser_audit_compared.csv  one row per record, with an
      'agree' flag and the published rule that settles each disagreement
  printed summary: agreement on inclusion and on the donor name
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT  # noqa: E402
from coding import donor_key  # noqa: E402
from rev_utils import load_collection, load_gifts  # noqa: E402

AUDIT = os.path.join(PROJECT_ROOT, "audit")

# Remaining disagreements and the published rule (supplement S1) that settles each.
# The reader saw one credit line at a time, so rules that depend on the whole file
# (titles, parenthetical qualifiers) were out of its reach.
RESOLVED = {
    10993: "parser: a personal title is dropped when the untitled form also occurs",
    78793: "parser: a personal title is dropped when the untitled form also occurs",
    17365: "parser: the qualifier stays because 'Celeste and Armand Bartos' does not occur alone as a donor",
    63966: "parser: S.A. is a listed name suffix",
    114385: "parser: a credit line that mentions a purchase is classed as a purchase",
}


def norm(name):
    if not isinstance(name, str) or not name.strip():
        return ""
    return donor_key(re.sub(r"^the\s+", "", name.strip(), flags=re.I))


def main():
    a = pd.read_csv(os.path.join(AUDIT, "audit_ai.csv"))
    g = load_gifts().set_index("ObjectID")
    allrec = load_collection().drop_duplicates("ObjectID").set_index("ObjectID")
    p = pd.DataFrame({"ObjectID": a["ObjectID"]})
    p["CreditLine"] = p["ObjectID"].map(allrec["CreditLine"])
    p["in_sample"] = p["ObjectID"].isin(g.index)
    p["parsed_donor"] = p["ObjectID"].map(g["Donor"])
    a["ai_in_sample"] = a["ai_in_sample"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])
    t = p.merge(a[["ObjectID", "ai_in_sample", "ai_donor", "ai_note"]], on="ObjectID", how="left")
    t["agree_in_sample"] = t["in_sample"] == t["ai_in_sample"]
    both = t["in_sample"] & t["ai_in_sample"]
    t["agree_donor"] = ~both | (t["parsed_donor"].map(norm) == t["ai_donor"].map(norm))
    t["agree"] = t["agree_in_sample"] & t["agree_donor"]
    t["resolution"] = t["ObjectID"].map(RESOLVED).fillna("")
    cols = ["ObjectID", "CreditLine", "in_sample", "ai_in_sample", "parsed_donor", "ai_donor", "agree_in_sample",
            "agree_donor", "agree", "ai_note", "resolution"]
    t = t[cols].sort_values(["agree", "ObjectID"])
    t.to_csv(os.path.join(AUDIT, "parser_audit_compared.csv"), index=False)
    n = len(t)
    print(f"records: {n}")
    print(f"inclusion agrees: {int(t['agree_in_sample'].sum())} of {n}")
    print(f"donor agrees where both include: {int((t['agree_donor'] & both).sum())} of {int(both.sum())}")
    print(f"disagreements: {int((~t['agree']).sum())}, settled by a published rule: {int(((~t['agree']) & (t['resolution'] != '')).sum())}")


if __name__ == "__main__":
    main()

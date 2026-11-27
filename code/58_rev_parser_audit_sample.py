# -*- coding: utf-8 -*-
"""
Audit sample for the credit-line parser (not part of run_all.py).

Draws a fixed random sample of records from the five departments, 1929-2024:
  200 records inside the donor-identified gift sample, to check the donor name
      and the three flags the parser returns;
  100 records outside it whose credit line mentions a gift or bequest, to check
      that they were rightly left out.
The columns human_donor, human_in_sample and note are left blank for a person
to fill in. The parser's error rate is the share of rows where the human
reading differs.

Output (output/): parser_audit_sample.csv

The sample depends on the gift channel at the time it is drawn. The set that was
read (2026-11-27) is frozen in moma_paper/audit/audit_ai.csv; script 62 applies
the current parser to those records.
"""

import pandas as pd

from rev_utils import RNG_SEED, load_collection, load_gifts, out

N_IN, N_OUT = 200, 100


def main():
    g = load_gifts()
    allrec = load_collection()
    inside = g.sample(N_IN, random_state=RNG_SEED)[["ObjectID", "CreditLine", "Donor", "Unit", "co_donor", "fund_led", "by_exchange"]]
    inside.insert(2, "in_sample", True)
    rest = allrec[~allrec["ObjectID"].isin(g["ObjectID"])]
    rest = rest[rest["CreditLine"].fillna("").str.contains(r"gift|bequest|given|donat|collection|generosity|exchange", case=False, regex=True)]
    outside = rest.sample(N_OUT, random_state=RNG_SEED)[["ObjectID", "CreditLine"]]
    outside["in_sample"] = False
    t = pd.concat([inside, outside], ignore_index=True).rename(columns={"Donor": "parsed_donor", "Unit": "donor_unit"})
    for col in ("human_donor", "human_in_sample", "note"):
        t[col] = ""
    t.to_csv(out("parser_audit_sample.csv"), index=False)
    print(f"{len(t)} rows written ({N_IN} inside the sample, {N_OUT} outside); "
          f"{len(rest):,} outside records mention a gift or bequest")


if __name__ == "__main__":
    main()

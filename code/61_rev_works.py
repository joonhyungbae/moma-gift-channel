# -*- coding: utf-8 -*-
"""
Revision analysis (referees, round two): a third count, the work.

A work is the accession number before its part suffix ('401.2003.1-16' ->
'401.2003'), which groups the records of a portfolio, an illustrated book, a
set of cards, or a multi-part object. A work is coded female when every
gender-coded record of it is female and male when every one is male (mixed
works are outside the share, as mixed records are); the same for origin.
A work spanning several donor units is credited to the unit holding most of
its records.

For each unit with at least 100 records: number of works, work-level shares,
and the contribution to the gift-channel work-level share (share with minus
share without the unit's works).

Output: output/rev_works_units.csv, output/rev_works_channel.csv
"""

import pandas as pd

from rev_utils import load_gifts, out, pct


def work_table(g):
    w = g.groupby("Work").agg(f=("is_F", "sum"), gv=("g_valid", "sum"), nw=("is_NW", "sum"), ov=("o_valid", "sum"),
                              n=("Work", "size"))
    w["unit"] = g.groupby("Work")["Unit"].agg(lambda s: s.value_counts().index[0])
    w["F"] = ((w["gv"] > 0) & (w["f"] == w["gv"])).astype(int)
    w["GV"] = ((w["gv"] > 0) & ((w["f"] == w["gv"]) | (w["f"] == 0))).astype(int)
    w["NW"] = ((w["ov"] > 0) & (w["nw"] == w["ov"])).astype(int)
    w["OV"] = ((w["ov"] > 0) & ((w["nw"] == w["ov"]) | (w["nw"] == 0))).astype(int)
    return w


def main():
    g = load_gifts()
    w = work_table(g)
    tot = w[["F", "GV", "NW", "OV"]].sum()
    base_f, base_nw = pct(tot["F"], tot["GV"]), pct(tot["NW"], tot["OV"])
    pd.DataFrame([{"records": len(g), "works": len(w), "work_pct_f": base_f, "work_pct_nw": base_nw,
                   "works_one_record": int((w["n"] == 1).sum())}]).to_csv(out("rev_works_channel.csv"), index=False)
    sizes = g["Unit"].value_counts()
    su = w.groupby("unit")[["F", "GV", "NW", "OV"]].sum()
    nw_ = w.groupby("unit").size()
    rows = []
    for u in sizes[sizes >= 100].index:
        s = su.loc[u]
        rows.append({"unit": u, "n_records": int(sizes[u]), "n_works": int(nw_.get(u, 0)),
                     "work_pct_f": pct(s["F"], s["GV"]), "work_pct_nw": pct(s["NW"], s["OV"]),
                     "work_contrib_f": base_f - pct(tot["F"] - s["F"], tot["GV"] - s["GV"]),
                     "work_contrib_nw": base_nw - pct(tot["NW"] - s["NW"], tot["OV"] - s["OV"])})
    pd.DataFrame(rows).set_index("unit").to_csv(out("rev_works_units.csv"))
    print("records", len(g), "works", len(w), "work shares", round(base_f, 2), round(base_nw, 2))
    print(pd.DataFrame(rows).set_index("unit").head(12).round(2).to_string())


if __name__ == "__main__":
    main()

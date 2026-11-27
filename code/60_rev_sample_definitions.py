# -*- coding: utf-8 -*-
"""
Revision check (referees, round two): how the main results depend on the
definition of the gift channel.

Definitions
  main                    named-collection credit lines included, records acquired
                          by exchange of an earlier gift excluded (the article)
  with exchange           as main, plus the by-exchange records under the original donor
  without collections     as main, minus the named-collection credit lines
  neither change          without collections, with exchange
  without fund-led        as main, minus records whose credit line opens with a
                          museum fund or purchase formula
  MoMA nationality only   as main, with MoMA's nationality field and no Wikidata fill
  Rothschild units merged as main, with the Contemporary Drawings Collection in the
                          Judith Rothschild Foundation's unit

For each: records, donor units, record and artist shares, the six cases' joint
record-level contribution, and the ten largest record-level contributions
(absolute) against the same units' artist-level contributions, with artists
divided among donors by gift records.

Output: output/rev_sample_definitions.csv
"""

import pandas as pd

from rev_utils import (CASE_LABEL, artist_frame, artist_names, artist_shares, artist_table, constituents,
                       frac_artist_contribution, load_gifts, out, record_shares, removed_fraction)

DEFS = [("main", {}), ("with exchange", {"include_exchange": True}),
        ("without collections", {"include_collections": False}),
        ("neither change", {"include_collections": False, "include_exchange": True}),
        ("without fund-led", {"exclude_fund_led": True}),
        ("MoMA nationality only", {"moma_nationality": True}),
        ("Rothschild units merged", {"merge_rothschild": True})]
SIX = set(CASE_LABEL.values())


def summary(g, names):
    cons = constituents(g, names)
    cons["unit"] = g["Unit"].reindex(cons["rec"]).to_numpy()
    museum, art = record_shares(g), artist_shares(artist_frame(cons))
    AT = artist_table(cons)
    rest = record_shares(g[~g["Unit"].isin(SIX)])
    sizes = g["Unit"].value_counts()
    big = sizes[sizes >= 100].index
    rows = []
    for u in big:
        r = record_shares(g[g["Unit"] != u])
        af, an = frac_artist_contribution(AT, removed_fraction(cons, g.index[g["Unit"] == u], AT))
        rows.append((u, museum["pct_f"] - r["pct_f"], museum["pct_nw"] - r["pct_nw"], af, an))
    c = pd.DataFrame(rows, columns=["unit", "rf", "rn", "af", "an"]).set_index("unit")
    out_ = {"records": len(g), "units": g["Unit"].nunique(), "units_100": len(big),
            "largest_unit": sizes.index[0], "largest_unit_records": int(sizes.iloc[0]),
            "pct_f": museum["pct_f"], "pct_nw": museum["pct_nw"],
            "art_pct_f": art["art_pct_f"], "art_pct_nw": art["art_pct_nw"],
            "six_contrib_f": museum["pct_f"] - rest["pct_f"], "six_contrib_nw": museum["pct_nw"] - rest["pct_nw"]}
    # Units grouped by records per artist, against the channel's own ratio.
    n_art = cons[~cons["placeholder"]].groupby("unit")["cid"].nunique()
    rpa = (sizes / n_art.reindex(sizes.index)).reindex(c.index)
    ratio = len(g) / art["n_artists"]
    out_["channel_records_per_artist"] = ratio
    few = rpa >= ratio
    out_["units_few_artists"] = int(few.sum())
    for k, rc, ac in (("f", "rf", "af"), ("nw", "rn", "an")):
        out_[f"few_rec_sum_{k}"] = c.loc[few, rc].abs().sum(); out_[f"few_art_sum_{k}"] = c.loc[few, ac].abs().sum()
        out_[f"many_rec_sum_{k}"] = c.loc[~few, rc].abs().sum(); out_[f"many_art_sum_{k}"] = c.loc[~few, ac].abs().sum()
    gu = g[g["Unit"] == "Agnes Gund"]
    if len(gu):
        gs = record_shares(gu)
        gc = cons[(cons["unit"] == "Agnes Gund") & ~cons["placeholder"]].drop_duplicates("cid")
        gf, ggv = (gc["gender"] == "F").sum(), gc["gender"].isin(["F", "M"]).sum()
        out_.update({"gund_records": len(gu), "gund_pct_f": gs["pct_f"], "gund_art_gap_f": 100 * gf / ggv - art["art_pct_f"],
                     "gund_art_coded": int(ggv), "gund_top_dept_share": 100 * gu["Department"].value_counts().iloc[0] / len(gu)})
    for k, rc, ac in (("f", "rf", "af"), ("nw", "rn", "an")):
        top = c[rc].abs().sort_values(ascending=False).index[:10]
        out_[f"top10_rec_sum_{k}"] = c.loc[top, rc].abs().sum()
        out_[f"top10_art_sum_{k}"] = c.loc[top, ac].abs().sum()
        others = c.index.difference(top)
        out_[f"other_median_rec_{k}"] = c.loc[others, rc].abs().median()
        out_[f"other_median_art_{k}"] = c.loc[others, ac].abs().median()
    return out_


def main():
    names = artist_names()
    rows = []
    for label, kw in DEFS:
        rows.append({"definition": label, **summary(load_gifts(**kw), names)})
        print(label, "done", flush=True)
    df = pd.DataFrame(rows).set_index("definition")
    df.to_csv(out("rev_sample_definitions.csv"))
    pd.set_option("display.width", 250)
    print(df.T.to_string())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Revision analysis A1b (reviewer 1 comment 1): is diversification concentrated
in a few donors when artists are counted instead of records?

  1. First-source attribution: every artist is credited to the donor unit of
     the artist's first record in the gift channel.
  2. Concentration: number of units that supply half of the female /
     non-Western artists (and records), and the top-ten units' share, against
     a reference in which the female / non-Western label is shuffled across
     artists (for records: across artist clusters, so that all records of one
     artist move together).
  3. Volume versus artist-level diversity: what the largest-volume units and
     the smallest units each supply.
  4. Leave-one-unit-out artist-level contributions for every unit.
  5. Bound for the Judith Rothschild Foundation gifts, whose artists mostly
     lack a nationality in MoMA's metadata.

Outputs (output/): rev_artist_first_source.csv, rev_artist_concentration.csv,
rev_artist_size_classes.csv, rev_artist_unit_contributions.csv,
rev_rothschild_origin_bound.csv, rev_new_artists_by_decade.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rev_utils import RNG_SEED, artist_names, constituents, load_gifts, out, pct

N_SHUFFLE_ARTISTS = 2000
N_SHUFFLE_RECORDS = 500
ROTHSCHILD = ["Judith Rothschild Foundation", "Judith Rothschild Foundation (Boris Kerdimun Archive)"]


def half_cover(units, flag):
    c = pd.Series(units[flag]).value_counts()
    return int(((c.cumsum() / c.sum()) < 0.5).sum() + 1)


def top_share(units, flag, k=10):
    c = pd.Series(units[flag]).value_counts()
    return 100 * c.head(k).sum() / c.sum()


def main():
    rng = np.random.default_rng(RNG_SEED)
    g = load_gifts()
    cons = constituents(g, artist_names())
    cons = cons[~cons["placeholder"]].copy()
    cons["unit"] = g["Unit"].reindex(cons["rec"]).to_numpy()
    cons["year"] = g["YearAcquired"].reindex(cons["rec"]).to_numpy()
    sizes = g["Unit"].value_counts()

    # 1. First-source attribution.
    fs = cons.sort_values(["year", "rec"]).drop_duplicates("cid").set_index("cid")
    fs["F"] = fs["gender"] == "F"
    fs["GV"] = fs["gender"].isin(["F", "M"])
    fs["NW"] = fs["origin"] == "Non-Western"
    fs["OV"] = fs["origin"].notna()
    per_unit = fs.groupby("unit").agg(new_artists=("F", "size"), new_f=("F", "sum"), new_gv=("GV", "sum"),
                                      new_nw=("NW", "sum"), new_ov=("OV", "sum"))
    per_unit["n_records"] = sizes.reindex(per_unit.index)
    per_unit.sort_values("new_artists", ascending=False).to_csv(out("rev_artist_first_source.csv"))

    # 2. Concentration against a label-shuffle reference.
    rows = []
    for label, flag, valid in (("female artists", "F", "GV"), ("non-Western artists", "NW", "OV")):
        d = fs[fs[valid]]
        u, x = d["unit"].to_numpy(), d[flag].to_numpy()
        nh = [half_cover(u, rng.permutation(x)) for _ in range(N_SHUFFLE_ARTISTS)]
        nt = [top_share(u, rng.permutation(x)) for _ in range(N_SHUFFLE_ARTISTS)]
        rows.append({"measure": label, "n_flagged": int(x.sum()), "n_coded": len(d),
                     "units_supplying_any": int(pd.Series(u[x]).nunique()),
                     "units_for_half": half_cover(u, x), "shuffled_mean": np.mean(nh),
                     "shuffled_lo": np.percentile(nh, 2.5), "shuffled_hi": np.percentile(nh, 97.5),
                     "top10_share": top_share(u, x), "top10_shuffled_mean": np.mean(nt),
                     "top10_shuffled_lo": np.percentile(nt, 2.5), "top10_shuffled_hi": np.percentile(nt, 97.5)})
    for label, flag, valid in (("female records", "is_F", "g_valid"), ("non-Western records", "is_NW", "o_valid")):
        d = g[g[valid] == 1]
        cl = d.groupby("Cluster")[flag].first()
        codes = d["Cluster"].map({c: i for i, c in enumerate(cl.index)}).to_numpy()
        lab, u = cl.to_numpy(), d["Unit"].to_numpy()
        x = lab[codes] == 1
        nh, nt = [], []
        for _ in range(N_SHUFFLE_RECORDS):
            xp = rng.permutation(lab)[codes] == 1
            nh.append(half_cover(u, xp)); nt.append(top_share(u, xp))
        rows.append({"measure": label, "n_flagged": int(x.sum()), "n_coded": len(d),
                     "units_supplying_any": int(pd.Series(u[x]).nunique()),
                     "units_for_half": half_cover(u, x), "shuffled_mean": np.mean(nh),
                     "shuffled_lo": np.percentile(nh, 2.5), "shuffled_hi": np.percentile(nh, 97.5),
                     "top10_share": top_share(u, x), "top10_shuffled_mean": np.mean(nt),
                     "top10_shuffled_lo": np.percentile(nt, 2.5), "top10_shuffled_hi": np.percentile(nt, 97.5)})
    c_all = sizes
    rows.append({"measure": "all records", "n_flagged": len(g), "units_supplying_any": len(c_all),
                 "units_for_half": int(((c_all.cumsum() / c_all.sum()) < 0.5).sum() + 1),
                 "top10_share": 100 * c_all.head(10).sum() / c_all.sum()})
    c_art = per_unit["new_artists"].sort_values(ascending=False)
    rows.append({"measure": "all artists", "n_flagged": len(fs), "units_supplying_any": len(c_art),
                 "units_for_half": int(((c_art.cumsum() / c_art.sum()) < 0.5).sum() + 1),
                 "top10_share": 100 * c_art.head(10).sum() / c_art.sum()})
    conc = pd.DataFrame(rows)
    conc.to_csv(out("rev_artist_concentration.csv"), index=False)

    # 3. Volume versus artist-level diversity, by unit size class.
    all_units = pd.DataFrame({"n_records": sizes}).join(per_unit.drop(columns="n_records"), how="left").fillna(0)
    classes = [("1-9 records", all_units["n_records"] < 10),
               ("10-99 records", (all_units["n_records"] >= 10) & (all_units["n_records"] < 100)),
               ("100+ records", all_units["n_records"] >= 100),
               ("ten largest units", all_units["n_records"].rank(ascending=False, method="first") <= 10)]
    rec_f, rec_nw = g.groupby("Unit")["is_F"].sum(), g.groupby("Unit")["is_NW"].sum()
    srows = []
    for label, m in classes:
        s = all_units[m]
        srows.append({"class": label, "units": int(m.sum()), "pct_records": pct(s["n_records"].sum(), len(g)),
                      "pct_f_records": pct(rec_f.reindex(s.index).sum(), g["is_F"].sum()),
                      "pct_nw_records": pct(rec_nw.reindex(s.index).sum(), g["is_NW"].sum()),
                      "pct_new_artists": pct(s["new_artists"].sum(), len(fs)),
                      "pct_new_f_artists": pct(s["new_f"].sum(), fs["F"].sum()),
                      "pct_new_nw_artists": pct(s["new_nw"].sum(), fs["NW"].sum()),
                      "new_f_artists": int(s["new_f"].sum()), "new_nw_artists": int(s["new_nw"].sum())})
    big = all_units[all_units["n_records"] >= 100]
    for col in ("new_artists", "new_f", "new_nw"):
        srows.append({"class": f"Spearman rho, records vs {col} (100+ units)",
                      "units": len(big), "pct_records": spearmanr(big["n_records"], big[col]).statistic})
    pd.DataFrame(srows).to_csv(out("rev_artist_size_classes.csv"), index=False)

    # 4. Leave-one-unit-out artist-level contributions (artists held only through one unit).
    A = cons.groupby("cid").agg(gender=("gender", "first"), origin=("origin", "first"),
                                units=("unit", lambda s: frozenset(s)))
    F, GV = A["gender"] == "F", A["gender"].isin(["F", "M"])
    NW, OV = A["origin"] == "Non-Western", A["origin"].notna()
    sole = A[A["units"].map(len) == 1].copy()
    sole["unit"] = sole["units"].map(lambda s: next(iter(s)))
    lrows = []
    for u, s in sole.groupby("unit"):
        f, gv = (s["gender"] == "F").sum(), s["gender"].isin(["F", "M"]).sum()
        nw, ov = (s["origin"] == "Non-Western").sum(), s["origin"].notna().sum()
        lrows.append({"unit": u, "n_records": int(sizes[u]), "sole_artists": len(s), "sole_f": int(f),
                      "sole_nw": int(nw), "sole_origin_uncoded": int(len(s) - ov),
                      "art_contrib_f": pct(F.sum(), GV.sum()) - pct(F.sum() - f, GV.sum() - gv),
                      "art_contrib_nw": pct(NW.sum(), OV.sum()) - pct(NW.sum() - nw, OV.sum() - ov)})
    loo = pd.DataFrame(lrows).set_index("unit")
    loo.sort_values("n_records", ascending=False).to_csv(out("rev_artist_unit_contributions.csv"))

    # 5. Rothschild bound: artists without a nationality treated as non-Western.
    brows = []
    for label, sel in (("Judith Rothschild Foundation", ROTHSCHILD[:1]), ("Both Rothschild Foundation strings", ROTHSCHILD)):
        sel = set(sel)
        in_unit = A["units"].map(lambda s: bool(s & sel))
        only = A["units"].map(lambda s: s <= sel)
        unc = in_unit & ~OV
        nw_all, ov_all = NW.sum() + unc.sum(), OV.sum() + unc.sum()
        keep = ~only
        nw_wo, ov_wo = (NW & keep).sum() + (unc & keep).sum(), (OV & keep).sum() + (unc & keep).sum()
        brows.append({"units": label, "artists": int(in_unit.sum()), "artists_without_nationality": int(unc.sum()),
                      "sole_source_without_nationality": int((unc & only).sum()),
                      "contribution_coded_only_pp": pct(NW.sum(), OV.sum()) - pct((NW & keep).sum(), (OV & keep).sum()),
                      "museum_share_if_non_western": pct(nw_all, ov_all), "share_without_units": pct(nw_wo, ov_wo),
                      "contribution_if_non_western_pp": pct(nw_all, ov_all) - pct(nw_wo, ov_wo),
                      "contribution_if_western_pp": pct(NW.sum(), OV.sum() + unc.sum())
                      - pct((NW & keep).sum(), (OV & keep).sum() + (unc & keep).sum())})
    bound = pd.DataFrame(brows)
    bound.to_csv(out("rev_rothschild_origin_bound.csv"), index=False)

    # 6. New artists by decade and the breadth of their sources.
    fs["decade"] = (fs["year"] // 10) * 10
    drows = []
    for dec, s in fs.groupby("decade"):
        tf, tn = s.loc[s["F"], "unit"].value_counts(), s.loc[s["NW"], "unit"].value_counts()
        drows.append({"decade": int(dec), "new_artists": len(s), "new_f": int(s["F"].sum()),
                      "pct_f": pct(s["F"].sum(), s["GV"].sum()), "units_with_new_f": len(tf),
                      "top_unit_f": tf.index[0] if len(tf) else "", "top_unit_f_n": int(tf.iloc[0]) if len(tf) else 0,
                      "new_nw": int(s["NW"].sum()), "pct_nw": pct(s["NW"].sum(), s["OV"].sum()),
                      "units_with_new_nw": len(tn), "top_unit_nw": tn.index[0] if len(tn) else "",
                      "top_unit_nw_n": int(tn.iloc[0]) if len(tn) else 0})
    pd.DataFrame(drows).to_csv(out("rev_new_artists_by_decade.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(conc.round(1).to_string()); print(pd.DataFrame(srows).round(2).to_string()); print(bound.round(2).to_string())
    print(loo.reindex(loo["art_contrib_nw"].abs().sort_values(ascending=False).index).head(6).round(2).to_string())
    print(loo.reindex(loo["art_contrib_f"].abs().sort_values(ascending=False).index).head(6).round(2).to_string())


if __name__ == "__main__":
    main()

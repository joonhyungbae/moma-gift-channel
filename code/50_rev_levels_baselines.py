# -*- coding: utf-8 -*-
"""
Revision analyses A1-A3 (reviewer 1, comments 1-3).

A1  Three levels of measurement for every donor unit and for the six cases:
    records, distinct artists, artists first entering the collection.
    Record-level and artist-level counterfactuals for the gift channel.
A2  Dependence among records: artist-cluster bootstrap intervals and shares
    with the largest artist cluster removed. Donor-level association between
    volume and gap, with permutation p-values and donor-bootstrap intervals.
A3  Gap from baseline versus contribution to the aggregate, and a
    department x decade matched baseline.

Outputs (output/):
  rev_units.csv                 one row per donor unit, all measures
  rev_six_cases.csv             the six cases, all measures
  rev_museum.csv                museum-wide gift-channel baselines
  rev_joint_counterfactual.csv  removing the six cases, record and artist level
  rev_volume_gap_association.csv
  rev_suberas.csv               sub-era profile with artist counts
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from donor_bio_utils import CASE_ORDER, detect_suberas
from rev_utils import (
    CASE_LABEL, N_BOOT, RNG_SEED, artist_frame, artist_names, artist_shares, cluster_bootstrap,
    constituents, drop_top_cluster, load_collection, load_gifts, matched_expectation, out, pct,
    record_shares,
)

MIN_N = 100


def first_entry_table(collection, names):
    """Earliest acquisition date of every artist across all channels."""
    cons = constituents(collection, names)
    cons = cons[~cons["placeholder"]]
    cons["date"] = collection["DateAcquired"].reindex(cons["rec"]).to_numpy()
    cons["year"] = collection["YearAcquired"].reindex(cons["rec"]).to_numpy()
    first = cons.groupby("cid").agg(first_date=("date", "min"), first_year=("year", "min"),
                                    n_records_all=("rec", "nunique"))
    return first


def unit_measures(sub, cons_sub, gifts_rest, museum, first, cons_all_gift_counts, with_boot=True):
    row = record_shares(sub)
    years = sub["YearAcquired"]
    yc = years.value_counts()
    row.update({
        "era_start": int(years.min()), "era_end": int(years.max()),
        "era_years": int(years.max() - years.min()),
        "peak_year": int(yc.idxmax()), "peak_year_share": pct(yc.max(), len(sub)),
        "top_dept": sub["Department"].value_counts().idxmax(),
        "top_dept_share": pct(sub["Department"].value_counts().max(), len(sub)),
        "n_co_donor": int(sub["co_donor"].sum()), "n_fund_led": int(sub["fund_led"].sum()),
        "n_by_exchange": int(sub["by_exchange"].sum()),
    })
    # Level 2: distinct artists.
    arts = artist_frame(cons_sub)
    row.update(artist_shares(arts))
    rec_per_artist = arts["n_records"].to_numpy(dtype=float)
    if len(rec_per_artist):
        shares = rec_per_artist / rec_per_artist.sum()
        row["effective_n_artists"] = 1.0 / np.sum(shares ** 2)
        row["top_artist"] = arts["n_records"].idxmax()
        row["top_artist_name"] = arts.loc[row["top_artist"], "name"]
        row["top_artist_share"] = pct(rec_per_artist.max(), len(sub))
    row["placeholder_records"] = int(cons_sub.loc[cons_sub["placeholder"], "rec"].nunique())
    row["unidentified_records"] = int(cons_sub.loc[cons_sub["unidentified"], "rec"].nunique())
    # Level 3: artists first entering the collection through this unit, and
    # artists held in the gift channel only through this unit.
    dates = sub["DateAcquiredTS"].reindex(cons_sub["rec"]).to_numpy()
    c = cons_sub[~cons_sub["placeholder"]].copy()
    c["date"] = sub["DateAcquiredTS"].reindex(c["rec"]).to_numpy()
    c["first_date"] = c["cid"].map(first["first_date"])
    intro = arts.loc[arts.index.intersection(c.loc[c["date"] == c["first_date"], "cid"].unique())]
    intro_s = artist_shares(intro)
    row.update({"intro_n": intro_s["n_artists"], "intro_f": intro_s["art_f"],
                "intro_g_valid": intro_s["art_g_valid"], "intro_pct_f": intro_s["art_pct_f"],
                "intro_nw": intro_s["art_nw"], "intro_o_valid": intro_s["art_o_valid"],
                "intro_pct_nw": intro_s["art_pct_nw"]})
    own = c.groupby("cid")["rec"].nunique()
    sole = arts.loc[own.index[own.to_numpy() == cons_all_gift_counts.reindex(own.index).to_numpy()]]
    row["sole_n"] = int(len(sole))
    row["sole_f"] = int((sole["gender"] == "F").sum())
    row["sole_nw"] = int((sole["origin"] == "Non-Western").sum())
    # Gap versus contribution (record level).
    rest = record_shares(gifts_rest)
    row["gap_f"] = row["pct_f"] - museum["pct_f"]
    row["gap_nw"] = row["pct_nw"] - museum["pct_nw"]
    row["contrib_f"] = museum["pct_f"] - rest["pct_f"]
    row["contrib_nw"] = museum["pct_nw"] - rest["pct_nw"]
    row["share_of_records"] = pct(row["n"], museum["n"])
    row["share_of_f_records"] = pct(row["f"], museum["f"])
    row["share_of_nw_records"] = pct(row["nw"], museum["nw"])
    # Matched baseline.
    exp = matched_expectation(sub, gifts_rest)
    row.update(exp)
    row["gap_f_matched"] = row["pct_f"] - exp["exp_f"]
    row["gap_nw_matched"] = row["pct_nw"] - exp["exp_nw"]
    # Dependence among records.
    row.update(drop_top_cluster(sub))
    if with_boot:
        row.update(cluster_bootstrap(sub))
    return row


def association(units, xcol, ycol, n_perm=10000, seed=RNG_SEED):
    d = units[[xcol, ycol]].dropna()
    if len(d) < 8:
        return {"n_units": len(d)}
    rho = spearmanr(d[xcol], d[ycol]).statistic
    rng = np.random.default_rng(seed)
    y = d[ycol].to_numpy()
    perm = np.array([spearmanr(d[xcol], rng.permutation(y)).statistic for _ in range(n_perm)])
    p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1)
    boot = []
    arr = d.to_numpy()
    for _ in range(2000):
        s = arr[rng.integers(0, len(arr), len(arr))]
        boot.append(spearmanr(s[:, 0], s[:, 1]).statistic)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"n_units": len(d), "spearman_rho": rho, "perm_p": p, "boot_lo": lo, "boot_hi": hi}


def main():
    gifts = load_gifts()
    names = artist_names()
    collection = load_collection()
    date_by_obj = collection.set_index("ObjectID")["DateAcquired"]
    gifts["DateAcquiredTS"] = gifts["ObjectID"].map(date_by_obj[~date_by_obj.index.duplicated()])
    first = first_entry_table(collection, names)

    cons = constituents(gifts, names)
    cons["unit"] = gifts["Unit"].reindex(cons["rec"]).to_numpy()
    cons["case"] = gifts["CaseDonor"].reindex(cons["rec"]).to_numpy()
    gift_counts = cons[~cons["placeholder"]].groupby("cid")["rec"].nunique()

    museum = record_shares(gifts)
    museum_art = artist_shares(artist_frame(cons))
    mrow = {**museum, **museum_art, "n_units": gifts["Unit"].nunique(),
            "n_mixed_gender": int((gifts["Gender_Grouped"] == "Mixed").sum()),
            "n_uncoded_gender": int((gifts["Gender_Grouped"] == "Other/Unknown").sum()),
            "n_mixed_origin": int((gifts["GeoDetail"] == "Mixed").sum()),
            "n_uncoded_origin": int(gifts["GeoDetail"].isna().sum()),
            "n_co_donor": int(gifts["co_donor"].sum()), "n_fund_led": int(gifts["fund_led"].sum()),
            "n_by_exchange": int(gifts["by_exchange"].sum())}
    pd.DataFrame([mrow]).to_csv(out("rev_museum.csv"), index=False)

    sizes = gifts["Unit"].value_counts()
    big = sizes[sizes >= MIN_N].index.tolist()
    case_units = {c: (CASE_LABEL[c] if c != "Rockefeller lineage" else "Rockefeller lineage") for c in CASE_ORDER}
    for u in case_units.values():
        if u not in big:
            big.append(u)

    rows = []
    for u in big:
        mask = gifts["Unit"] == u
        row = unit_measures(gifts[mask], cons[cons["unit"] == u], gifts[~mask], museum, first, gift_counts)
        row["unit"] = u
        row["is_case"] = u in case_units.values()
        rows.append(row)
    units = pd.DataFrame(rows).set_index("unit")
    # Artist-level gap from the museum-wide artist-level share.
    units["art_gap_f"] = units["art_pct_f"] - museum_art["art_pct_f"]
    units["art_gap_nw"] = units["art_pct_nw"] - museum_art["art_pct_nw"]
    units.sort_values("n", ascending=False).to_csv(out("rev_units.csv"))
    six = units.loc[[case_units[c] for c in CASE_ORDER]]
    six.to_csv(out("rev_six_cases.csv"))

    # Artist-level counterfactual: gift-channel artist shares without each case / all six.
    cf_rows = []
    def art_cf(label, mask_rec):
        keep = cons[~cons["rec"].isin(gifts.index[mask_rec])]
        a = artist_shares(artist_frame(keep))
        r = record_shares(gifts[~mask_rec])
        cf_rows.append({"removed": label, "n_removed": int(mask_rec.sum()),
                        "rec_pct_f": r["pct_f"], "rec_delta_f": r["pct_f"] - museum["pct_f"],
                        "rec_pct_nw": r["pct_nw"], "rec_delta_nw": r["pct_nw"] - museum["pct_nw"],
                        "n_artists": a["n_artists"],
                        "artists_lost": museum_art["n_artists"] - a["n_artists"],
                        "art_pct_f": a["art_pct_f"], "art_delta_f": a["art_pct_f"] - museum_art["art_pct_f"],
                        "art_pct_nw": a["art_pct_nw"], "art_delta_nw": a["art_pct_nw"] - museum_art["art_pct_nw"]})
    cf_rows.append({"removed": "(none)", "n_removed": 0, "rec_pct_f": museum["pct_f"], "rec_pct_nw": museum["pct_nw"],
                    "n_artists": museum_art["n_artists"], "art_pct_f": museum_art["art_pct_f"],
                    "art_pct_nw": museum_art["art_pct_nw"]})
    for c in CASE_ORDER:
        art_cf(case_units[c], (gifts["Unit"] == case_units[c]).to_numpy())
    art_cf("All six", gifts["Unit"].isin(case_units.values()).to_numpy())
    # Reference point: the same number of records removed from the long tail
    # (all units with fewer than 100 records), repeated draws.
    tail_idx = np.flatnonzero((~gifts["Unit"].isin(big)).to_numpy())
    n_six = int(gifts["Unit"].isin(case_units.values()).sum())
    rng = np.random.default_rng(RNG_SEED)
    draws = []
    for _ in range(200):
        m = np.zeros(len(gifts), dtype=bool)
        m[rng.choice(tail_idx, size=min(n_six, len(tail_idx)), replace=False)] = True
        keep = cons[~cons["rec"].isin(gifts.index[m])]
        a = artist_shares(artist_frame(keep))
        draws.append((museum_art["n_artists"] - a["n_artists"], a["art_pct_f"], a["art_pct_nw"]))
    draws = np.array(draws)
    cf_rows.append({"removed": f"Random {n_six} long-tail records (mean of 200 draws)", "n_removed": n_six,
                    "artists_lost": draws[:, 0].mean(), "art_pct_f": draws[:, 1].mean(),
                    "art_delta_f": draws[:, 1].mean() - museum_art["art_pct_f"],
                    "art_pct_nw": draws[:, 2].mean(),
                    "art_delta_nw": draws[:, 2].mean() - museum_art["art_pct_nw"]})
    pd.DataFrame(cf_rows).to_csv(out("rev_joint_counterfactual.csv"), index=False)

    # Where do artists come from? Distribution of distinct artists by unit size.
    art_unit = cons[~cons["placeholder"]].groupby("cid")["unit"].agg(lambda s: set(s))
    only_big = art_unit.map(lambda s: s <= set(big))
    only_tail = art_unit.map(lambda s: not (s & set(big)))
    af = artist_frame(cons)
    src = []
    for label, m in (("Only through units with 100+ records", only_big),
                     ("Only through units with <100 records", only_tail),
                     ("Through both", ~(only_big | only_tail))):
        a = artist_shares(af.loc[m.index[m]])
        src.append({"source": label, **a})
    tail_rec = record_shares(gifts[~gifts["Unit"].isin(big)])
    big_rec = record_shares(gifts[gifts["Unit"].isin(big)])
    src.append({"source": "records: units with 100+ records", **big_rec})
    src.append({"source": "records: units with <100 records", **tail_rec})
    pd.DataFrame(src).to_csv(out("rev_artist_sources.csv"), index=False)

    # Volume versus gap, donor units with at least MIN_N records.
    pop = units[units["n"] >= MIN_N].copy()
    pop["log_n"] = np.log10(pop["n"])
    assoc = []
    for label, col, valid_col, min_valid in (
        ("records: |gap| female", "gap_f", "g_valid", 30),
        ("records: |gap| non-Western", "gap_nw", "o_valid", 30),
        ("records: |matched gap| female", "gap_f_matched", "g_valid", 30),
        ("records: |matched gap| non-Western", "gap_nw_matched", "o_valid", 30),
        ("artists: |gap| female", "art_gap_f", "art_g_valid", 10),
        ("artists: |gap| non-Western", "art_gap_nw", "art_o_valid", 10),
        ("records: |contribution| female", "contrib_f", "g_valid", 30),
        ("records: |contribution| non-Western", "contrib_nw", "o_valid", 30),
    ):
        d = pop[pop[valid_col] >= min_valid].copy()
        d["y"] = d[col].abs()
        assoc.append({"measure": label, **association(d, "n", "y")})
    pd.DataFrame(assoc).to_csv(out("rev_volume_gap_association.csv"), index=False)

    # Sub-eras for the six cases, with artist counts per sub-era.
    sub_rows = []
    for c in CASE_ORDER:
        d = gifts[gifts["Unit"] == case_units[c]]
        for start, end, _ in detect_suberas(d["YearAcquired"].value_counts().sort_index()):
            s = d[d["YearAcquired"].between(start, end)]
            cs = cons[cons["rec"].isin(s.index)]
            a = artist_frame(cs)
            top = s["Cluster"].value_counts()
            sub_rows.append({"case": case_units[c], "start": start, "end": end, **record_shares(s),
                             "n_artists": len(a), "top_cluster_share": pct(top.iloc[0], len(s)),
                             "n_fund_led": int(s["fund_led"].sum()), "n_co_donor": int(s["co_donor"].sum())})
    pd.DataFrame(sub_rows).to_csv(out("rev_suberas.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print("MUSEUM", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in mrow.items()})
    show = ["n", "pct_f", "g_valid", "pct_nw", "o_valid", "n_artists", "art_pct_f", "art_g_valid", "art_pct_nw",
            "art_o_valid", "effective_n_artists", "top_artist_share", "intro_n", "intro_f", "intro_nw", "sole_n"]
    print(six[show].round(1).to_string())
    show2 = ["gap_f", "contrib_f", "exp_f", "gap_f_matched", "gap_nw", "contrib_nw", "exp_nw", "gap_nw_matched",
             "pct_f_wo_top", "pct_nw_wo_top", "f_lo", "f_hi", "nw_lo", "nw_hi", "n_clusters"]
    print(six[show2].round(2).to_string())
    print(pd.DataFrame(cf_rows).round(2).to_string())
    print(pd.DataFrame(src).round(2).to_string())
    print(pd.DataFrame(assoc).round(3).to_string())
    print(pd.DataFrame(sub_rows).round(1).to_string())


if __name__ == "__main__":
    main()

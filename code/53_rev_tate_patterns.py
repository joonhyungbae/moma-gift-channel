# -*- coding: utf-8 -*-
"""
Revision analysis A8 (reviewer 1 comment 6).

Tests whether the donor *patterns* documented at MoMA recur at the Tate,
instead of asking whether the same named donors appear there. For donor units
with at least 100 gift records at each museum the script compares

  1. concentration of the gift channel,
  2. how common event-style and single-artist gifts are,
  3. the association between volume and the gap from the museum's baseline,
  4. the record-level versus artist-level effect of removing the six units
     with the largest absolute record-level contributions,
  5. how many units the six fixed-threshold signatures classify.

Tate caveats carried into the paper: origin is coded from place of birth
(MoMA: nationality), the public Tate file ends in 2013, and channel-aggregate
credit lines (Turner Bequest and similar) are excluded as in the submitted
boundary check.

Outputs (output/): rev_tate_units.csv, rev_tate_vs_moma_patterns.csv
"""

import importlib.util
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from coding import BASELINE_WESTERN_REGIONS, INSTITUTIONAL_DONOR_RE, TRAILING_CLAUSE_RE, UNIDENTIFIED_ARTIST_RE
from config import PROJECT_ROOT
from donor_bio_utils import canonicalize_donors
from rev_utils import RNG_SEED, out, pct
from tate_utils import extract_tate_donor_from_creditline

TATE_DIR = os.path.join(PROJECT_ROOT, "data", "tate")

# Country as written in Tate's placeOfBirth (last comma-separated element).
COUNTRY_REGION = {
    "North America": ["United States", "Canada", "Staten Island", "Charlotte Amalie"],
    "Western Europe": [
        "United Kingdom", "France", "Deutschland", "Italia", "Éire", "België", "Nederland", "Schweiz", "España",
        "Österreich", "Sverige", "Portugal", "Ellás", "Danmark", "Norge", "Suomi", "Luxembourg", "Ísland", "Malta",
        "London", "Bristol", "Saint Hélier", "Isle of Man", "Solothurn", "Braintree", "Egremont", "Auteuil",
        "Kensington", "Liverpool", "Charlieu", "Plymouth", "Perth", "Epsom", "Wimbledon", "Blackheath",
        "Bermondsey", "Douglas", "Stoke on Trent", "Stockholm", "Melmerby", "Beckington", "Edinburgh",
        "Hertfordshire", "Rochdale", "Canterbury",
    ],
    "Australia and New Zealand": ["Australia", "New Zealand"],
    "Eastern Europe and former USSR": [
        "Polska", "Rossiya", "Ukrayina", "Ceská Republika", "România", "Magyarország", "Hrvatska", "Slovenija",
        "Belarus", "Lietuva", "Jugoslavija", "Latvija", "Slovenská Republika", "Bosna i Hercegovina", "Schlesien",
        "Niederschlesien", "Bulgaria", "Makedonija", "Eesti", "Armenia", "Moldova", "Shqipëria",
    ],
    "Latin America and Caribbean": [
        "Brasil", "Argentina", "México", "Colombia", "Cuba", "Venezuela", "Chile", "Perú", "Guyana", "Jamaica",
        "Barbados", "Bahamas", "Costa Rica", "Panamá", "Montserrat", "Nicaragua",
    ],
    "East Asia": ["Nihon", "Zhonghua", "Taehan Min'guk", "Choson Minjujuui In'min Konghwaguk", "Chung-hua Min-kuo"],
    "South and Southeast Asia": [
        "Bharat", "Pakistan", "Sri Lanka", "Viet Nam", "Singapore", "Indonesia", "Bangladesh", "Malaysia",
        "Myanmar", "Pilipinas", "Lao", "Prathet Thai",
    ],
    "Middle East and North Africa": [
        "Yisra'el", "Îran", "Al-Lubnan", "Misr", "Türkiye", "Al-Jaza'ir", "Suriyah", "Al-‘Iraq", "As-Sudan", "Tunis",
    ],
    "Sub-Saharan Africa": [
        "South Africa", "Mauritius", "Uganda", "Cameroun", "Bénin", "Tanzania", "Zimbabwe", "Kenya", "Nigeria",
        "Zambia",
    ],
    "Oceania (other)": ["Samoa"],
}
COUNTRY_TO_REGION = {c: r for r, cs in COUNTRY_REGION.items() for c in cs}


def load_channel_filter():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "39_phase_l_tate_channel_filter.py")
    spec = importlib.util.spec_from_file_location("tate_channel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.is_channel_aggregate


def load_tate_gifts():
    w = pd.read_csv(os.path.join(TATE_DIR, "artwork_data.csv"), low_memory=False)
    proc = pd.read_csv(os.path.join(TATE_DIR, "processed_tate_data.csv"), usecols=["Department"])
    assert len(proc) == len(w), "processed and raw Tate files are not row-aligned"
    w["dept"] = proc["Department"].to_numpy()
    a = pd.read_csv(os.path.join(TATE_DIR, "artist_data.csv"))
    w["year"] = pd.to_numeric(w["acquisitionYear"], errors="coerce")
    cl = w["creditLine"].fillna("")
    w = w[cl.str.match(r"^(Presented|Bequeathed|Given) by", case=False)].copy()
    # Same clean-up as the MoMA parser: drop 'through ...', 'in honour of ...',
    # 'by exchange' and similar clauses that describe the transfer, not the donor.
    w["DonorRaw"] = w["creditLine"].apply(extract_tate_donor_from_creditline).map(
        lambda v: (TRAILING_CLAUSE_RE.sub("", str(v)).strip(" .;") or np.nan) if pd.notna(v) else v)
    w = w[w["DonorRaw"].notna() & w["year"].notna()].copy()
    w["Unit"] = canonicalize_donors(w["DonorRaw"])
    w = w[w["Unit"].notna()].copy()
    is_channel = load_channel_filter()
    w["channel"] = w["Unit"].map(is_channel)
    a = a.set_index("id")
    country = a["placeOfBirth"].dropna().str.split(",").str[-1].str.strip()
    region = country.map(COUNTRY_TO_REGION)
    unmapped = sorted(set(country[region.isna()]))
    if unmapped:
        print("Unmapped Tate countries:", unmapped)
    w["gender"] = w["artistId"].map(a["gender"])
    w["region"] = w["artistId"].map(region)
    w["is_F"] = (w["gender"] == "Female").astype(int)
    w["g_valid"] = w["gender"].isin(["Female", "Male"]).astype(int)
    w["is_NW"] = (w["region"].notna() & ~w["region"].isin(BASELINE_WESTERN_REGIONS)).astype(int)
    w["o_valid"] = w["region"].notna().astype(int)
    w["cid"] = w["artistId"].astype(str)
    w["unidentified"] = w["artist"].fillna("").str.contains(UNIDENTIFIED_ARTIST_RE).astype(int)
    return w.reset_index(drop=True)


def unit_table(df, pairs, min_n=100):
    """Unit measures. df holds one row per record; pairs holds one row per
    (record, artist) with artist-level flags a_f, a_gv, a_nw, a_ov."""
    museum_f, museum_nw = pct(df["is_F"].sum(), df["g_valid"].sum()), pct(df["is_NW"].sum(), df["o_valid"].sum())
    arts = pairs.groupby("cid").agg(f=("a_f", "max"), gv=("a_gv", "max"), nw=("a_nw", "max"), ov=("a_ov", "max"))
    pair_groups = dict(tuple(pairs.groupby("Unit")))
    m_art_f, m_art_nw = pct(arts["f"].sum(), arts["gv"].sum()), pct(arts["nw"].sum(), arts["ov"].sum())
    rows = []
    sizes = df["Unit"].value_counts()
    tot = df[["is_F", "g_valid", "is_NW", "o_valid"]].sum()
    for u in sizes[sizes >= min_n].index:
        g = df[df["Unit"] == u]
        yc = g["year"].value_counts()
        pg = pair_groups.get(u)
        ac = pg["cid"].value_counts() if pg is not None else pd.Series(dtype=int)
        a = arts.loc[ac.index]
        s = g[["is_F", "g_valid", "is_NW", "o_valid"]].sum()
        pf, pn = pct(s["is_F"], s["g_valid"]), pct(s["is_NW"], s["o_valid"])
        dc = g["dept"].value_counts() if "dept" in g else pd.Series([len(g)])
        rows.append({
            "unit": u, "n": len(g), "era_years": int(g["year"].max() - g["year"].min()),
            "top_dept_share": pct(dc.iloc[0], len(g)),
            "unidentified_records": int(g["unidentified"].sum()) if "unidentified" in g else 0,
            "institutional": bool(INSTITUTIONAL_DONOR_RE.search(str(u))),
            "peak_year_share": pct(yc.max(), len(g)),
            "top_artist_share": pct(ac.max(), len(g)) if len(ac) else 0.0,
            "n_artists": len(ac), "g_valid": int(s["g_valid"]), "o_valid": int(s["o_valid"]),
            "pct_f": pf, "pct_nw": pn, "gap_f": pf - museum_f, "gap_nw": pn - museum_nw,
            "contrib_f": museum_f - pct(tot["is_F"] - s["is_F"], tot["g_valid"] - s["g_valid"]),
            "contrib_nw": museum_nw - pct(tot["is_NW"] - s["is_NW"], tot["o_valid"] - s["o_valid"]),
            "art_g_valid": int(a["gv"].sum()), "art_o_valid": int(a["ov"].sum()),
            "art_gap_f": pct(a["f"].sum(), a["gv"].sum()) - m_art_f,
            "art_gap_nw": pct(a["nw"].sum(), a["ov"].sum()) - m_art_nw,
        })
    base = {"n": len(df), "units": int(df["Unit"].nunique()), "pct_f": museum_f, "pct_nw": museum_nw,
            "n_artists": len(arts), "art_pct_f": m_art_f, "art_pct_nw": m_art_nw}
    return pd.DataFrame(rows).set_index("unit"), base, arts


def perm_spearman(x, y, n_perm=10000, seed=RNG_SEED):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 8:
        return np.nan, np.nan, len(d)
    rho = spearmanr(d["x"], d["y"]).statistic
    rng = np.random.default_rng(seed)
    yv = d["y"].to_numpy()
    perm = np.array([spearmanr(d["x"], rng.permutation(yv)).statistic for _ in range(n_perm)])
    return rho, (np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1), len(d)


def pattern_summary(label, df, pairs, units, base, arts):
    sizes = df["Unit"].value_counts()
    k1 = max(1, int(round(0.01 * len(sizes))))
    row = {"museum": label, "gift_records": base["n"], "donor_units": base["units"],
           "units_100plus": len(units),
           "top_1pct_share_of_records": pct(sizes.head(k1).sum(), base["n"]),
           "share_of_records_in_100plus_units": pct(units["n"].sum(), base["n"]),
           "record_pct_f": base["pct_f"], "record_pct_nw": base["pct_nw"],
           "artist_pct_f": base["art_pct_f"], "artist_pct_nw": base["art_pct_nw"],
           "units_peak_year_over_90": pct((units["peak_year_share"] > 90).sum(), len(units)),
           "units_peak_year_over_50": pct((units["peak_year_share"] > 50).sum(), len(units)),
           "units_top_artist_over_90": pct((units["top_artist_share"] > 90).sum(), len(units)),
           "units_top_artist_over_50": pct((units["top_artist_share"] > 50).sum(), len(units)),
           "median_peak_year_share": units["peak_year_share"].median(),
           "median_top_artist_share": units["top_artist_share"].median()}
    for key, col, valid, mv in (("f", "gap_f", "g_valid", 30), ("nw", "gap_nw", "o_valid", 30)):
        d = units[units[valid] >= mv]
        rho, p, n = perm_spearman(d["n"], d[col].abs())
        row.update({f"vol_gap_rho_{key}": rho, f"vol_gap_p_{key}": p, f"vol_gap_n_{key}": n})
    for key, col, valid, mv in (("f", "art_gap_f", "art_g_valid", 10), ("nw", "art_gap_nw", "art_o_valid", 10)):
        d = units[units[valid] >= mv]
        rho, p, n = perm_spearman(d["n"], d[col].abs())
        row.update({f"vol_artgap_rho_{key}": rho, f"vol_artgap_p_{key}": p, f"vol_artgap_n_{key}": n})
    # Six largest absolute record-level contributions on each dimension: record versus artist level.
    for key, flag, valid, ccol in (("f", "is_F", "g_valid", "contrib_f"), ("nw", "is_NW", "o_valid", "contrib_nw")):
        top6 = units[ccol].abs().sort_values(ascending=False).head(6).index
        m = df["Unit"].isin(top6)
        rec_delta = pct(df.loc[~m, flag].sum(), df.loc[~m, valid].sum()) - pct(df[flag].sum(), df[valid].sum())
        aflag, avalid = ("a_f", "a_gv") if key == "f" else ("a_nw", "a_ov")
        a_wo = pairs[~pairs["Unit"].isin(top6)].groupby("cid").agg(x=(aflag, "max"), v=(avalid, "max"))
        art_delta = pct(a_wo["x"].sum(), a_wo["v"].sum()) - (base["art_pct_f"] if key == "f" else base["art_pct_nw"])
        row.update({f"top6_{key}_units": "; ".join(top6), f"top6_{key}_share_of_records": pct(m.sum(), len(df)),
                    f"top6_{key}_abs_contrib_sum": units.loc[top6, ccol].abs().sum(),
                    f"top6_{key}_record_delta": rec_delta, f"top6_{key}_artist_delta": art_delta,
                    f"top6_{key}_artists_lost_pct": pct(len(arts) - len(a_wo), len(arts))})
    # Simple signatures that need no department field.
    row["sig_single_artist_pct"] = pct((units["top_artist_share"] > 90).sum(), len(units))
    row["sig_event_pct"] = pct((units["peak_year_share"] > 90).sum(), len(units))
    row["sig_long_era_pct"] = pct((units["era_years"] >= 60).sum(), len(units))
    row["sig_none_pct"] = pct(((units["top_artist_share"] <= 90) & (units["peak_year_share"] <= 90)
                               & (units["era_years"] < 60)).sum(), len(units))
    return row


def moma_tables():
    """MoMA gift records and (record, artist) pairs in the same form as Tate."""
    from rev_utils import artist_names, constituents, load_gifts
    g = load_gifts().rename(columns={"YearAcquired": "year"})
    cons_all = constituents(g, artist_names())
    cons = cons_all[~cons_all["placeholder"]]
    pairs = pd.DataFrame({
        "Unit": g["Unit"].reindex(cons["rec"]).to_numpy(), "cid": cons["cid"].to_numpy(),
        "a_f": (cons["gender"] == "F").astype(int).to_numpy(),
        "a_gv": cons["gender"].isin(["F", "M"]).astype(int).to_numpy(),
        "a_nw": (cons["origin"] == "Non-Western").astype(int).to_numpy(),
        "a_ov": cons["origin"].notna().astype(int).to_numpy()})
    unid = cons_all.groupby("rec")["unidentified"].max().reindex(g.index).fillna(False).astype(int)
    g["unidentified"] = unid.to_numpy()
    g["dept"] = g["Department"]
    return g[["Unit", "year", "dept", "unidentified", "is_F", "g_valid", "is_NW", "o_valid"]], pairs


def tate_pairs(t):
    return pd.DataFrame({"Unit": t["Unit"].to_numpy(), "cid": t["cid"].to_numpy(), "a_f": t["is_F"].to_numpy(),
                         "a_gv": t["g_valid"].to_numpy(), "a_nw": t["is_NW"].to_numpy(), "a_ov": t["o_valid"].to_numpy()})


def main():
    tate_all = load_tate_gifts()
    tate = tate_all[~tate_all["channel"]].copy()
    tp = tate_pairs(tate)
    t_units, t_base, t_arts = unit_table(tate, tp)
    t_units.sort_values("n", ascending=False).to_csv(out("rev_tate_units.csv"))
    moma, mp = moma_tables()
    m_units, m_base, m_arts = unit_table(moma, mp)
    rows = [pattern_summary("MoMA", moma, mp, m_units, m_base, m_arts),
            pattern_summary("Tate", tate, tp, t_units, t_base, t_arts)]
    # Lower threshold as well, since the Tate's named-donor gift channel is smaller.
    t50, _, _ = unit_table(tate, tp, min_n=50)
    m50, _, _ = unit_table(moma, mp, min_n=50)
    rows.append(pattern_summary("MoMA (units with 50+ records)", moma, mp, m50, m_base, m_arts))
    rows.append(pattern_summary("Tate (units with 50+ records)", tate, tp, t50, t_base, t_arts))
    # The six pattern rules of script 52, applied with the same thresholds to both museums.
    spec = importlib.util.spec_from_file_location(
        "typology", os.path.join(os.path.dirname(os.path.abspath(__file__)), "52_rev_typology.py"))
    typ = importlib.util.module_from_spec(spec); spec.loader.exec_module(typ)
    rule_rows = []
    for label, units in (("MoMA", m_units), ("Tate", t_units), ("MoMA (units with 50+ records)", m50),
                         ("Tate (units with 50+ records)", t50)):
        matched, status = typ.classify(units, typ.DEFAULT)
        r = {"museum": label, "units": len(units), "one pattern": int((status == "clean").sum()),
             "several": int((status == "ambiguous").sum()), "none": int((status == "unclassified").sum())}
        for ty in typ.TYPES:
            r[ty] = int(matched.map(lambda m: ty in m).sum())
        rule_rows.append(r)
        if label == "Tate":
            t_units.assign(patterns=matched.map("; ".join)).sort_values("n", ascending=False).to_csv(out("rev_tate_units.csv"))
    rules = pd.DataFrame(rule_rows).set_index("museum")
    rules.to_csv(out("rev_tate_rules.csv"))
    print(rules.to_string())
    res = pd.DataFrame(rows).set_index("museum").T
    res.to_csv(out("rev_tate_vs_moma_patterns.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 200)
    print("Tate gift records with named donor:", len(tate_all), "| channel aggregates removed:", int(tate_all["channel"].sum()))
    for k, r in res.iterrows():
        print(k, "|", " | ".join(str(round(v, 2)) if isinstance(v, float) else str(v)[:110] for v in r))
    print(t_units.sort_values("n", ascending=False).round(1).head(30).to_string())


if __name__ == "__main__":
    main()

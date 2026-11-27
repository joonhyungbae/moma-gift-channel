# -*- coding: utf-8 -*-
"""
Revision analyses A4-A6 and the case-selection audit.

A4  Gender coding: denominators, records that cannot be coded, where they sit
    (department, decade, donor), what kind of maker they name, and how far the
    female share can move under bounding assumptions (reviewer 1 comment 5,
    reviewer 2 comment 3).
A5  Regional coding: region composition and the Western / non-Western share
    under the baseline rule and five alternative rules (reviewer 1 comment 5,
    reviewer 2 comment 4).
A6  Rockefeller family-line aggregation: the lineage taken apart (reviewer 2
    comment 6).
SEL The six-axis selection rule re-applied to the corrected data.

Outputs (output/): rev_gender_*.csv, rev_region_*.csv, rev_rockefeller.csv,
rev_selection_audit.csv
"""

import re

import numpy as np
import pandas as pd

from coding import (
    REGIONS, SCHEME_LABELS, SCHEMES, gender_token, origin_record, split_paren_tokens,
)
from donor_bio_utils import CASE_ORDER
from rev_utils import (
    CASE_LABEL, artist_frame, artist_names, artist_shares, constituents, load_gifts, out, pct,
    record_shares,
)

FIRM_RE = re.compile(
    r"\b(inc|co|corp|corporation|company|ltd|gmbh|associates|architects|architecture|studio|studios|"
    r"design|designs|group|collective|workshop|works|manufactur\w*|partners|office|bureau|atelier|"
    r"brothers|bros|sons|press|editions|industries|laborator\w*|factory|team)\b|&",
    re.IGNORECASE,
)


def case_units():
    return {c: (CASE_LABEL[c] if c != "Rockefeller lineage" else "Rockefeller lineage") for c in CASE_ORDER}


# ----------------------------------------------------------------- gender
def gender_tables(gifts, cons):
    def breakdown(group):
        vc = group["Gender_Grouped"].value_counts()
        n = len(group)
        f, m = vc.get("Female", 0), vc.get("Male", 0)
        mixed, unc = vc.get("Mixed", 0), vc.get("Other/Unknown", 0)
        return pd.Series({
            "n": n, "female": f, "male": m, "mixed": mixed, "uncoded": unc,
            "pct_uncoded": pct(unc, n), "pct_mixed": pct(mixed, n),
            "pct_f_coded": pct(f, f + m),
            # Bounds: every uncoded and mixed record counted as not-female / as female.
            "pct_f_lower": pct(f, n), "pct_f_upper": pct(f + mixed + unc, n),
        })

    rows = [breakdown(gifts).rename("All donor-identified gifts")]
    for d, g in gifts.groupby("Department"):
        rows.append(breakdown(g).rename(f"Department: {d}"))
    for dec, g in gifts.groupby("Decade"):
        rows.append(breakdown(g).rename(f"Acquired {int(dec)}s"))
    for c, u in case_units().items():
        rows.append(breakdown(gifts[gifts["Unit"] == u]).rename(f"Case: {u}"))
    table = pd.DataFrame(rows)
    table.to_csv(out("rev_gender_denominators.csv"))

    # Which donors hold the uncoded records?
    unc = gifts[gifts["Gender_Grouped"] == "Other/Unknown"]
    by_unit = unc["Unit"].value_counts()
    tot = gifts["Unit"].value_counts()
    conc = pd.DataFrame({"uncoded": by_unit, "n": tot.reindex(by_unit.index)})
    conc["pct_of_unit"] = 100 * conc["uncoded"] / conc["n"]
    conc["pct_of_all_uncoded"] = 100 * conc["uncoded"] / len(unc)
    conc["cum_pct_of_all_uncoded"] = conc["pct_of_all_uncoded"].cumsum()
    conc.head(25).to_csv(out("rev_gender_uncoded_by_donor.csv"))

    # What kind of maker do the uncoded records name?
    c = cons[cons["rec"].isin(unc.index)].copy()
    def kind(row):
        if row["unidentified"]:
            return "Unidentified maker"
        if row["placeholder"]:
            return "Group label (e.g. Various Artists)"
        name = row["name"] if isinstance(row["name"], str) else ""
        if FIRM_RE.search(name):
            return "Firm, studio or collective"
        if row["gender"] == "NB":
            return "Non-binary (recorded)"
        return "Named maker, gender not recorded"
    c["kind"] = c.apply(kind, axis=1)
    order = ["Unidentified maker", "Group label (e.g. Various Artists)", "Firm, studio or collective",
             "Named maker, gender not recorded", "Non-binary (recorded)"]
    rec_kind = c.groupby("rec")["kind"].agg(lambda s: next(k for k in order if k in set(s)))
    no_cons = len(unc) - rec_kind.shape[0]
    kinds = rec_kind.value_counts().reindex(order).fillna(0).astype(int)
    kinds["No constituent recorded"] = no_cons
    kt = pd.DataFrame({"records": kinds, "pct": 100 * kinds / len(unc)})
    kt.to_csv(out("rev_gender_uncoded_kinds.csv"))
    for dept in gifts["Department"].unique():
        idx = unc.index[unc["Department"] == dept]
        vc = rec_kind.reindex(idx).fillna("No constituent recorded").value_counts()
        kt[dept] = vc.reindex(kt.index).fillna(0).astype(int)
    kt.to_csv(out("rev_gender_uncoded_kinds.csv"))

    # Alternative treatments of multi-artist records.
    toks = gifts["Gender"].map(lambda s: [gender_token(t) for t in split_paren_tokens(s)])
    n_f = toks.map(lambda t: sum(x == "F" for x in t))
    n_m = toks.map(lambda t: sum(x == "M" for x in t))
    coded = (n_f + n_m) > 0
    alt = []
    def add(label, num, den):
        alt.append({"rule": label, "numerator": float(num), "denominator": float(den), "pct_f": pct(num, den)})
    s = record_shares(gifts)
    add("Main rule: single-gender records, women-only / (women-only + men-only)", s["f"], s["g_valid"])
    add("Fractional: mean share of coded constituents who are women", (n_f[coded] / (n_f + n_m)[coded]).sum(), coded.sum())
    add("Submitted rule (mixed records counted as women-only)", (n_f > 0).sum(), coded.sum())
    add("Lower bound: women-only / all records", s["f"], len(gifts))
    add("Upper bound: (women-only + mixed + uncoded) / all records",
        s["f"] + (gifts["Gender_Grouped"].isin(["Mixed", "Other/Unknown"])).sum(), len(gifts))
    arts = artist_shares(artist_frame(cons))
    add("Distinct artists: women / gender-coded artists", arts["art_f"], arts["art_g_valid"])
    pd.DataFrame(alt).to_csv(out("rev_gender_alternative_rules.csv"), index=False)
    return table, conc, kt, pd.DataFrame(alt)


# ----------------------------------------------------------------- region
def region_tables(gifts, cons):
    units = case_units()
    # Composition by region, records and artists.
    rec_rows, art_rows = {}, {}
    groups = {"All donor-identified gifts": gifts.index}
    groups.update({u: gifts.index[gifts["Unit"] == u] for u in units.values()})
    for label, idx in groups.items():
        g = gifts.loc[idx]
        vc = g["Region"].value_counts(dropna=False)
        row = {r: int(vc.get(r, 0)) for r in REGIONS}
        row["Mixed regions"] = int(vc.get("Mixed", 0))
        row["Not coded"] = int(g["Region"].isna().sum())
        row["n"] = len(g)
        rec_rows[label] = row
        a = artist_frame(cons[cons["rec"].isin(idx)])
        avc = a["region"].value_counts(dropna=False)
        arow = {r: int(avc.get(r, 0)) for r in REGIONS}
        arow["Not coded"] = int(a["region"].isna().sum())
        arow["n"] = len(a)
        art_rows[label] = arow
    pd.DataFrame(rec_rows).T.to_csv(out("rev_region_composition_records.csv"))
    pd.DataFrame(art_rows).T.to_csv(out("rev_region_composition_artists.csv"))

    # Schemes.
    rows = []
    contrib_rows = []
    six_mask = gifts["Unit"].isin(units.values())
    sizes = gifts["Unit"].value_counts()
    big_units = sizes[sizes >= 100].index
    for scheme in SCHEMES:
        origin = gifts["Nationality"].map(lambda s: origin_record(s, scheme))
        valid = origin.isin(["Western", "Non-Western"])
        nw = origin == "Non-Western"
        base = pct(nw.sum(), valid.sum())
        row = {"scheme": SCHEME_LABELS[scheme], "museum_pct_nw": base, "museum_nw": int(nw.sum()),
               "museum_valid": int(valid.sum())}
        wo = pct((nw & ~six_mask).sum(), (valid & ~six_mask).sum())
        row["without_six_pct_nw"] = wo
        row["six_contribution_pp"] = base - wo
        row["six_share_of_nw_records"] = pct((nw & six_mask).sum(), nw.sum())
        for c, u in units.items():
            m = gifts["Unit"] == u
            row[f"{u} | pct_nw"] = pct((nw & m).sum(), (valid & m).sum())
            row[f"{u} | contribution_pp"] = base - pct((nw & ~m).sum(), (valid & ~m).sum())
        rows.append(row)
        # Largest single-unit contributions under this scheme.
        grp = pd.DataFrame({"nw": nw, "valid": valid, "unit": gifts["Unit"]}).groupby("unit")[["nw", "valid"]].sum()
        grp = grp.loc[grp.index.intersection(big_units)]
        tot_nw, tot_valid = nw.sum(), valid.sum()
        grp["contribution_pp"] = base - 100 * (tot_nw - grp["nw"]) / (tot_valid - grp["valid"])
        top = grp.reindex(grp["contribution_pp"].abs().sort_values(ascending=False).index).head(8)
        for u, r in top.iterrows():
            contrib_rows.append({"scheme": SCHEME_LABELS[scheme], "unit": u, "nw": int(r["nw"]),
                                 "valid": int(r["valid"]), "contribution_pp": r["contribution_pp"]})
    schemes = pd.DataFrame(rows)
    schemes.to_csv(out("rev_region_schemes.csv"), index=False)
    pd.DataFrame(contrib_rows).to_csv(out("rev_region_scheme_top_contributors.csv"), index=False)
    return pd.DataFrame(rec_rows).T, pd.DataFrame(art_rows).T, schemes, pd.DataFrame(contrib_rows)


# ------------------------------------------------------------- Rockefeller
def rockefeller_table(gifts, cons):
    lin = gifts[gifts["Unit"] == "Rockefeller lineage"]
    museum = record_shares(gifts)
    parts = {
        "Rockefeller lineage (all strings)": lin.index,
        "Abby Aldrich Rockefeller, all": lin.index[lin["Donor"] == "Abby Aldrich Rockefeller"],
        "Other family members": lin.index[lin["Donor"] != "Abby Aldrich Rockefeller"],
    }
    rows = []
    for label, idx in parts.items():
        g = gifts.loc[idx]
        s = record_shares(g)
        a = artist_shares(artist_frame(cons[cons["rec"].isin(idx)]))
        rest = record_shares(gifts.drop(index=idx))
        yc = g["YearAcquired"].value_counts()
        rows.append({"part": label, **s, **a, "era_start": int(g["YearAcquired"].min()),
                     "era_end": int(g["YearAcquired"].max()),
                     "peak_year": int(yc.idxmax()), "peak_year_share": pct(yc.max(), len(g)),
                     "contrib_f": museum["pct_f"] - rest["pct_f"], "contrib_nw": museum["pct_nw"] - rest["pct_nw"],
                     "n_strings": g["Donor"].nunique()})
    strings = lin.groupby("Donor").agg(n=("Donor", "size"), start=("YearAcquired", "min"), end=("YearAcquired", "max"))
    strings.sort_values("n", ascending=False).to_csv(out("rev_rockefeller_strings.csv"))
    tab = pd.DataFrame(rows)
    tab.to_csv(out("rev_rockefeller.csv"), index=False)
    return tab, strings


# ----------------------------------------------------------- selection audit
from coding import INSTITUTIONAL_DONOR_RE as INSTITUTIONAL_RE  # noqa: E402


def selection_audit(gifts):
    units = pd.read_csv(out("rev_units.csv"), index_col=0)
    pop = units[units["n"] >= 100].copy()
    # Longevity without family aggregation, for reference.
    strings = gifts.groupby("Donor").agg(n=("Donor", "size"), start=("YearAcquired", "min"), end=("YearAcquired", "max"))
    strings = strings[strings["n"] >= 100]
    strings["era_years"] = strings["end"] - strings["start"]
    pop["institutional"] = [bool(INSTITUTIONAL_RE.search(u)) for u in pop.index]
    axes = [
        ("Volume (records)", pop["n"], False, None),
        ("Longevity (active era, years; family line aggregated)", pop["era_years"], False, None),
        ("Non-Western share, records (30+ coded)", pop["pct_nw"].where(pop["o_valid"] >= 30), False, None),
        ("Female share, records (30+ coded)", pop["pct_f"].where(pop["g_valid"] >= 30), False, None),
        ("Non-Western share, artists (10+ coded)", pop["art_pct_nw"].where(pop["art_o_valid"] >= 10), False, None),
        ("Female share, artists (10+ coded)", pop["art_pct_f"].where(pop["art_g_valid"] >= 10), False, None),
        ("Institutional donor with more than 90% of records in one year (records)",
         pop["n"].where(pop["institutional"] & (pop["peak_year_share"] > 90)), False, None),
        ("Departmental spread (lowest top-department share)", pop["top_dept_share"], True, None),
    ]
    rows = []
    for label, series, ascending, _ in axes:
        top = series.dropna().sort_values(ascending=ascending).head(5)
        for rank, (u, v) in enumerate(top.items(), start=1):
            rows.append({"axis": label, "rank": rank, "unit": u, "value": v, "n": int(pop.loc[u, "n"]),
                         "is_case": bool(pop.loc[u, "is_case"])})
    top_str = strings.sort_values("era_years", ascending=False).head(5)
    for rank, (u, r) in enumerate(top_str.iterrows(), start=1):
        rows.append({"axis": "Longevity (single credit-line strings, no aggregation)", "rank": rank, "unit": u,
                     "value": r["era_years"], "n": int(r["n"]), "is_case": False})
    tab = pd.DataFrame(rows)
    tab.to_csv(out("rev_selection_audit.csv"), index=False)
    return tab


def main():
    gifts = load_gifts()
    cons = constituents(gifts, artist_names())
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 60)
    g1, g2, g3, g4 = gender_tables(gifts, cons)
    print(g1.round(1).to_string()); print(g2.head(12).round(1).to_string()); print(g3.round(1).to_string()); print(g4.round(2).to_string())
    r1, r2, r3, r4 = region_tables(gifts, cons)
    print(r1.to_string()); print(r2.to_string()); print(r3.round(2).T.to_string()); print(r4.round(2).to_string())
    k1, k2 = rockefeller_table(gifts, cons)
    print(k1.round(2).to_string()); print(k2.sort_values("n", ascending=False).to_string())
    print(selection_audit(gifts).round(1).to_string())


if __name__ == "__main__":
    main()

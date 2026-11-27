# -*- coding: utf-8 -*-
"""
Shared loaders and measures for the revision analyses (scripts 50-54).

Units of analysis
  record   one accessioned object in the donor-identified gift sample
  artist   one MoMA constituent (ConstituentID); placeholder constituents
           ('Unidentified photographer', 'Various Artists') are not artists
  cluster  the set of constituents credited on a record; records sharing a
           cluster are not independent, so resampling is done by cluster
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding import PLACEHOLDER_ARTIST_RE, UNIDENTIFIED_ARTIST_RE, explode_constituents  # noqa: E402
from config import MAIN_DEPARTMENTS, PROJECT_ROOT, get_output_path, get_processed_path  # noqa: E402
from donor_bio_utils import CASE_ORDER, assign_case, load_moma_donor_gifts  # noqa: E402

CASE_LABEL = {
    "Kleiner": "Kleiner, Bell & Co.",
    "Rockefeller lineage": "Rockefeller lineage",
    "Jean Pigozzi": "Jean Pigozzi",
    "Peter J. Cohen": "Peter J. Cohen",
    "Judith Rothschild Foundation": "Judith Rothschild Foundation",
    "Agnes Gund": "Agnes Gund",
}
RNG_SEED = 20261121
N_BOOT = 5000


def out(name):
    return get_output_path(name)


def pct(num, den):
    return 100.0 * num / den if den else np.nan


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_gifts(exclude_fund_led=False, moma_nationality=False, merge_rothschild=False, **sample):
    """Donor-identified gift records with analytic unit and cluster columns.

    Keyword arguments go to load_moma_donor_gifts (include_collections,
    include_exchange) for the sample-definition check; the default is the main
    analysis. exclude_fund_led drops records whose credit line opens with a
    museum fund or purchase formula; moma_nationality uses MoMA's nationality
    field without the Wikidata fill."""
    df = load_moma_donor_gifts(**sample)
    if exclude_fund_led:
        df = df[~df["fund_led"]].copy()
    if moma_nationality:
        from coding import origin_record
        df["Nationality"] = df["Nationality_MoMA"]
        geo = df["Nationality"].apply(origin_record)
        df["GeographicOrigin"] = geo.where(geo != "Mixed")
    df = assign_case(df.reset_index(drop=True))
    # Work: the accession number before its part suffix ('401.2003.1-16' -> '401.2003').
    df["Work"] = df["AccessionNumber"].astype(str).str.extract(r"^(\d+\.\d{4})", expand=False).fillna(
        pd.Series("rec:" + df.index.astype(str), index=df.index))
    # Analytic unit: the Rockefeller family strings form one unit, every other
    # donor string is its own unit.
    df["Unit"] = df["CaseDonor"].where(df["CaseDonor"] == "Rockefeller lineage", df["Donor"])
    df.loc[df["CaseDonor"] == "Kleiner", "Unit"] = "Kleiner, Bell & Co."
    if merge_rothschild:
        df.loc[df["Unit"] == "Judith Rothschild Foundation Contemporary Drawings Collection", "Unit"] = "Judith Rothschild Foundation"
    df["Decade"] = (df["YearAcquired"] // 10) * 10
    df["Cluster"] = df["ConstituentID"].astype(str).str.replace(r"\s+", "", regex=True)
    missing = df["ConstituentID"].isna()
    df.loc[missing, "Cluster"] = "noid:" + df.loc[missing, "Artist"].fillna("").astype(str)
    df["is_F"] = (df["Gender_Grouped"] == "Female").astype(int)
    df["g_valid"] = df["Gender_Grouped"].isin(["Female", "Male"]).astype(int)
    df["is_NW"] = (df["GeographicOrigin"] == "Non-Western").astype(int)
    df["o_valid"] = df["GeographicOrigin"].notna().astype(int)
    return df.reset_index(drop=True)


def load_collection():
    """All 1929-2024 main-department records (any channel), for first-entry dating."""
    cols = ["ObjectID", "Artist", "ConstituentID", "Gender", "Nationality", "DateAcquired",
            "YearAcquired", "Department", "AcquisitionType", "CreditLine"]
    df = pd.read_csv(get_processed_path(), usecols=cols, low_memory=False)
    df["YearAcquired"] = pd.to_numeric(df["YearAcquired"], errors="coerce")
    df = df[df["YearAcquired"].between(1929, 2024) & df["Department"].isin(MAIN_DEPARTMENTS)].copy()
    df["DateAcquired"] = pd.to_datetime(df["DateAcquired"], errors="coerce")
    return df.reset_index(drop=True)


def artist_names():
    path = os.path.join(PROJECT_ROOT, "data", "Artists_full.csv")
    a = pd.read_csv(path, dtype={"ConstituentID": str})
    return a.set_index("ConstituentID")["DisplayName"]


def constituents(df, names=None):
    """Long (record, constituent) table for df, with placeholder flag."""
    cons = explode_constituents(df)
    if names is None:
        names = artist_names()
    cons["name"] = cons["cid"].map(names)
    cons["placeholder"] = cons["name"].fillna("").str.contains(PLACEHOLDER_ARTIST_RE)
    cons["unidentified"] = cons["name"].fillna("").str.contains(UNIDENTIFIED_ARTIST_RE)
    return cons


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------


def record_shares(df):
    f, gv = int(df["is_F"].sum()), int(df["g_valid"].sum())
    nw, ov = int(df["is_NW"].sum()), int(df["o_valid"].sum())
    return {"n": int(len(df)), "f": f, "g_valid": gv, "pct_f": pct(f, gv),
            "nw": nw, "o_valid": ov, "pct_nw": pct(nw, ov)}


def artist_frame(cons):
    """One row per distinct non-placeholder artist."""
    a = cons[~cons["placeholder"]].groupby("cid").agg(
        gender=("gender", "first"), origin=("origin", "first"), region=("region", "first"),
        name=("name", "first"), n_records=("rec", "nunique"))
    return a


def artist_table(cons):
    """One row per non-placeholder artist: gender/origin flags and number of gift records."""
    c = cons[~cons["placeholder"]]
    a = c.groupby("cid").agg(gender=("gender", "first"), origin=("origin", "first"), n_rec=("rec", "nunique"))
    a["F"] = (a["gender"] == "F").astype(float); a["GV"] = a["gender"].isin(["F", "M"]).astype(float)
    a["NW"] = (a["origin"] == "Non-Western").astype(float); a["OV"] = a["origin"].notna().astype(float)
    return a


def removed_fraction(cons, rec_index, artists):
    """Share of each artist's gift records that lie in rec_index (0 to 1), aligned to artists."""
    c = cons[~cons["placeholder"]]
    hit = c[c["rec"].isin(rec_index)].groupby("cid")["rec"].nunique()
    return (hit / artists["n_rec"]).reindex(artists.index).fillna(0.0)


def frac_artist_contribution(artists, frac, nw=None, ov=None):
    """Artist-level contribution of the removed records, artists divided among donors.

    Every artist counts once in the artist-level share, divided among the
    donors in proportion to the gift records each gave. Removing records
    removes the matching fraction of each artist. Returns (female, non-Western)
    contributions in points: share with the records minus share without them.
    nw / ov override the origin flags (for bounds on uncoded artists)."""
    nw = artists["NW"] if nw is None else nw
    ov = artists["OV"] if ov is None else ov
    w = 1.0 - frac
    f_all = pct(artists["F"].sum(), artists["GV"].sum()); f_wo = pct((w * artists["F"]).sum(), (w * artists["GV"]).sum())
    n_all = pct(nw.sum(), ov.sum()); n_wo = pct((w * nw).sum(), (w * ov).sum())
    return f_all - f_wo, n_all - n_wo


def artist_shares(artists):
    g = artists["gender"].isin(["F", "M"])
    o = artists["origin"].notna()
    return {"n_artists": int(len(artists)),
            "art_f": int((artists["gender"] == "F").sum()), "art_g_valid": int(g.sum()),
            "art_pct_f": pct((artists["gender"] == "F").sum(), g.sum()),
            "art_nw": int((artists["origin"] == "Non-Western").sum()), "art_o_valid": int(o.sum()),
            "art_pct_nw": pct((artists["origin"] == "Non-Western").sum(), o.sum())}


def cluster_bootstrap(df, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile intervals for record shares, resampling artist clusters.

    The gift sample is a complete enumeration, so these are not sampling
    intervals for a wider population. They show how far a donor's share moves
    when the set of artists in the gift is perturbed, which is the relevant
    dependence when many records share one artist.
    """
    agg = df.groupby("Cluster")[["is_F", "g_valid", "is_NW", "o_valid"]].sum().to_numpy(dtype=float)
    k = len(agg)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    sums = agg[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = 100 * sums[:, 0] / sums[:, 1]
        pn = 100 * sums[:, 2] / sums[:, 3]
    q = lambda x: tuple(np.nanpercentile(x, [2.5, 97.5])) if np.isfinite(x).any() else (np.nan, np.nan)
    return {"n_clusters": k, "f_lo": q(pf)[0], "f_hi": q(pf)[1], "nw_lo": q(pn)[0], "nw_hi": q(pn)[1]}


def drop_top_cluster(df, placeholder_ids=frozenset()):
    """Record shares after removing every record that credits the donor's
    largest identified artist (the constituent on the most records).

    Placeholder constituents (unidentified makers, group labels) carry no codes
    and are skipped when the largest artist is chosen."""
    if df.empty:
        return {}
    ids = df["Cluster"].astype(str).map(lambda c: set(re.findall(r"\d+", c)) - set(placeholder_ids))
    counts = pd.Series([i for s_ in ids for i in s_]).value_counts()
    if counts.empty:
        s = record_shares(df)
        return {"top_cluster_n": 0, "top_cluster_share": 0.0, "pct_f_wo_top": s["pct_f"], "g_valid_wo_top": s["g_valid"],
                "pct_nw_wo_top": s["pct_nw"], "o_valid_wo_top": s["o_valid"]}
    top = counts.index[0]
    has_top = ids.map(lambda s_: top in s_)
    s = record_shares(df[~has_top])
    return {"top_cluster_n": int(has_top.sum()), "top_cluster_share": pct(has_top.sum(), len(df)),
            "pct_f_wo_top": s["pct_f"], "g_valid_wo_top": s["g_valid"],
            "pct_nw_wo_top": s["pct_nw"], "o_valid_wo_top": s["o_valid"]}


DOMINANT_SHARE = 0.20  # a unit holding more than this share of a reference cell is left out of that cell


def matched_expectation(df_unit, df_ref):
    """Share expected for a unit if, within each department x decade cell, its
    records had the composition of other donors' gifts in that cell. Units that
    hold more than DOMINANT_SHARE of a cell's coded records are left out of that
    cell's reference, so that one large gift does not set the expectation."""
    res = {}
    for flag, valid, key in (("is_F", "g_valid", "f"), ("is_NW", "o_valid", "nw")):
        u = df_unit[df_unit[valid] == 1]
        r = df_ref[df_ref[valid] == 1]
        if "Unit" in r.columns and len(r):
            cu = r.groupby(["Department", "Decade", "Unit"]).size()
            share = cu / cu.groupby(level=[0, 1]).transform("sum")
            dom = share[share > DOMINANT_SHARE].index
            if len(dom):
                key_ = pd.MultiIndex.from_frame(r[["Department", "Decade", "Unit"]])
                r = r[~key_.isin(dom)]
        if u.empty:
            res[f"exp_{key}"] = np.nan
            continue
        cell = r.groupby(["Department", "Decade"])[flag].agg(["sum", "count"])
        dept = r.groupby("Department")[flag].agg(["sum", "count"])
        overall = r[flag].mean()
        w = u.groupby(["Department", "Decade"]).size()
        exp = 0.0
        for (d, dec), n in w.items():
            if (d, dec) in cell.index and cell.loc[(d, dec), "count"] >= 30:
                p = cell.loc[(d, dec), "sum"] / cell.loc[(d, dec), "count"]
            elif d in dept.index and dept.loc[d, "count"] > 0:
                p = dept.loc[d, "sum"] / dept.loc[d, "count"]
            else:
                p = overall
            exp += n * p
        res[f"exp_{key}"] = 100 * exp / len(u)
    return res

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


def load_gifts():
    """Donor-identified gift records with analytic unit and cluster columns."""
    df = assign_case(load_moma_donor_gifts())
    # Analytic unit: the Rockefeller family strings form one unit, every other
    # donor string is its own unit.
    df["Unit"] = df["CaseDonor"].where(df["CaseDonor"] == "Rockefeller lineage", df["Donor"])
    df.loc[df["CaseDonor"] == "Kleiner", "Unit"] = "Kleiner, Bell & Co."
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


def matched_expectation(df_unit, df_ref):
    """Share expected for a unit if, within each department x decade cell, its
    records had the composition of all other donors' gifts in that cell."""
    res = {}
    for flag, valid, key in (("is_F", "g_valid", "f"), ("is_NW", "o_valid", "nw")):
        u = df_unit[df_unit[valid] == 1]
        r = df_ref[df_ref[valid] == 1]
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

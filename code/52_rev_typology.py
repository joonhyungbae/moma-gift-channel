# -*- coding: utf-8 -*-
"""
Revision analysis A7 (reviewer 2 comment 1; reviewer 1 comment 2).

Applies the six gift-pattern signatures as fixed-threshold rules to every
donor unit with at least 100 records other than the six anchors, varies the
thresholds, and asks whether the donor feature space has cluster structure.
The submitted rules used record-level binomial intervals; the revised rules
use fixed thresholds on quantities that do not assume independent records.

Outputs (output/): rev_typology_classification.csv, rev_typology_summary.csv,
rev_typology_threshold_sensitivity.csv, rev_typology_clustering.csv
"""

import itertools
import re

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from rev_utils import RNG_SEED, out, pct

from coding import INSTITUTIONAL_DONOR_RE as INSTITUTIONAL_RE  # noqa: E402

DEFAULT = {
    "min_volume": 250,        # volume contributor: at least this many records
    "dept_single": 90.0,      # top-department share at or above: single-department gift
    "near_gap": 5.0,          # neither record share exceeds the museum-wide share by more than this
    "max_top_artist": 20.0,   # volume contributor: no artist above this share of records
    "era_years": 60,          # founding lineage: active era at least this long
    "top_artist": 90.0,       # single-artist: one artist above this share of records
    "placeholder": 50.0,      # anonymous archive: records by unidentified makers above this share
    "peak_year": 90.0,        # institutional mass gift: peak-year share above this
    "dept_spread": 60.0,      # cross-departmental: top-department share at or below
    "artist_gap": 10.0,       # ... and artist-level share above museum-wide by this much
    "min_artists": 30,        # ... over at least this many coded artists
}

ANCHOR_TYPE = {
    "Kleiner, Bell & Co.": "Volume contributor", "Rockefeller lineage": "Founding lineage",
    "Jean Pigozzi": "Single-artist block", "Peter J. Cohen": "Anonymous-archive donor",
    "Judith Rothschild Foundation": "Institutional mass-gift", "Agnes Gund": "Cross-departmental diversifying donor",
}

TYPES = ["Volume contributor", "Founding lineage", "Single-artist block", "Anonymous-archive donor",
         "Institutional mass-gift", "Cross-departmental diversifying donor"]


def apply_rules(r, t):
    m = []
    ph_share = pct(r["unidentified_records"], r["n"])
    if (r["n"] >= t["min_volume"] and r["top_dept_share"] >= t["dept_single"] and r["top_artist_share"] < t["max_top_artist"]
            and ph_share < t["placeholder"]
            and (pd.isna(r["gap_f"]) or r["gap_f"] <= t["near_gap"])
            and (pd.isna(r["gap_nw"]) or r["gap_nw"] <= t["near_gap"])
            and r["peak_year_share"] <= t["peak_year"]):
        m.append(TYPES[0])
    if r["era_years"] >= t["era_years"] and r["gap_f"] < 0 and r["gap_nw"] < 0:
        m.append(TYPES[1])
    if r["top_artist_share"] > t["top_artist"]:
        m.append(TYPES[2])
    if ph_share > t["placeholder"] and r["top_dept_share"] >= t["dept_single"]:
        m.append(TYPES[3])
    if r["institutional"] and r["peak_year_share"] > t["peak_year"]:
        m.append(TYPES[4])
    div_f = r["art_g_valid"] >= t["min_artists"] and r["art_gap_f"] >= t["artist_gap"]
    div_nw = r["art_o_valid"] >= t["min_artists"] and r["art_gap_nw"] >= t["artist_gap"]
    if r["top_dept_share"] <= t["dept_spread"] and (div_f or div_nw):
        m.append(TYPES[5])
    return m


def classify(units, t):
    matched = units.apply(lambda r: apply_rules(r, t), axis=1)
    status = matched.map(lambda m: "clean" if len(m) == 1 else ("unclassified" if not m else "ambiguous"))
    return matched, status


def main():
    units = pd.read_csv(out("rev_units.csv"), index_col=0)
    units = units[units["n"] >= 100].copy()
    units["institutional"] = [bool(INSTITUTIONAL_RE.search(u)) for u in units.index]
    units["top_artist_share"] = units["top_artist_share"].fillna(0.0)

    matched, status = classify(units, DEFAULT)
    units["types"] = matched.map(lambda m: "; ".join(m))
    units["status"] = status
    cols = ["n", "is_case", "status", "types", "era_years", "peak_year_share", "top_dept_share", "top_artist_share",
            "unidentified_records", "n_artists", "gap_f", "gap_nw", "art_gap_f", "art_gap_nw", "institutional"]
    units[cols].sort_values("n", ascending=False).to_csv(out("rev_typology_classification.csv"))

    anchors = units[units["is_case"]]
    others = units[~units["is_case"]]
    summ = [{"metric": "units with 100+ records", "value": len(units)},
            {"metric": "anchors", "value": len(anchors)},
            {"metric": "non-anchor units", "value": len(others)}]
    for s in ("clean", "ambiguous", "unclassified"):
        summ.append({"metric": f"non-anchor {s}", "value": int((others["status"] == s).sum())})
    for ty in TYPES:
        summ.append({"metric": f"type count (non-anchor, any match): {ty}",
                     "value": int(others["types"].str.contains(re.escape(ty)).sum())})
        summ.append({"metric": f"type count (non-anchor, clean): {ty}",
                     "value": int(((others["types"] == ty)).sum())})
    for u, r in anchors.iterrows():
        summ.append({"metric": f"anchor {u}", "value": r["types"] or "(no rule matched)"})
    pd.DataFrame(summ).to_csv(out("rev_typology_summary.csv"), index=False)

    # Threshold sensitivity: move one threshold at a time.
    grid = {"min_volume": [150, 250, 500, 1000], "dept_single": [80, 85, 90, 95], "near_gap": [2.5, 5, 10], "era_years": [40, 50, 60, 70],
            "top_artist": [50, 70, 80, 90, 95], "placeholder": [30, 50, 70], "peak_year": [70, 80, 90, 95],
            "dept_spread": [50, 60, 70], "artist_gap": [5, 10, 15], "max_top_artist": [10, 20, 30]}
    sens = []
    for key, values in grid.items():
        for v in values:
            t = dict(DEFAULT); t[key] = v
            m, s = classify(others, t)
            am, _ = classify(anchors, t)
            own = sum(ANCHOR_TYPE[u] in am.loc[u] for u in anchors.index)
            sens.append({"threshold": key, "value": v, "default": DEFAULT[key] == v, "anchors_own_pattern": own,
                         "clean": int((s == "clean").sum()), "ambiguous": int((s == "ambiguous").sum()),
                         "unclassified": int((s == "unclassified").sum()),
                         "types_populated": len({x for mm in m for x in mm})})
    pd.DataFrame(sens).to_csv(out("rev_typology_threshold_sensitivity.csv"), index=False)

    # Cluster structure of the donor feature space (all units with 100+ records).
    feats = pd.DataFrame({
        "log_n": np.log10(units["n"]), "era_years": units["era_years"], "peak_year_share": units["peak_year_share"],
        "top_dept_share": units["top_dept_share"], "top_artist_share": units["top_artist_share"],
        "unidentified_share": 100 * units["unidentified_records"] / units["n"],
        "pct_f": units["pct_f"], "pct_nw": units["pct_nw"],
    })
    feats = feats.fillna(feats.median())
    x = StandardScaler().fit_transform(feats)
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=50, random_state=RNG_SEED).fit(x)
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(x)
        rows.append({"k": k, "kmeans_silhouette": silhouette_score(x, km.labels_),
                     "ward_silhouette": silhouette_score(x, ag.labels_),
                     "kmeans_smallest_cluster": int(np.bincount(km.labels_).min())})
    # Reference: silhouette of the same procedure on feature-wise shuffled data
    # (marginals kept, joint structure destroyed).
    rng = np.random.default_rng(RNG_SEED)
    null = {k: [] for k in range(2, 11)}
    for _ in range(100):
        xs = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        for k in range(2, 11):
            null[k].append(silhouette_score(xs, KMeans(n_clusters=k, n_init=5, random_state=0).fit(xs).labels_))
    for r in rows:
        r["shuffled_mean"] = float(np.mean(null[r["k"]]))
        r["shuffled_p95"] = float(np.percentile(null[r["k"]], 95))
    clus = pd.DataFrame(rows)
    # Agreement between the rule-based types and the k=6 partition.
    km6 = KMeans(n_clusters=6, n_init=50, random_state=RNG_SEED).fit(x)
    units["cluster6"] = km6.labels_
    clean = units[units["status"] == "clean"]
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(clean["types"], clean["cluster6"]) if len(clean) > 5 else np.nan
    clus["ari_rules_vs_k6_clean_units"] = ari
    clus.to_csv(out("rev_typology_clustering.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70)
    print(pd.DataFrame(summ).to_string())
    print(pd.DataFrame(sens).to_string())
    print(clus.round(3).to_string())
    print(units[cols].sort_values("n", ascending=False).round(1).to_string())


if __name__ == "__main__":
    main()

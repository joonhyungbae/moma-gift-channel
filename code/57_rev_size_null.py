# -*- coding: utf-8 -*-
"""
Revision analysis A2b (reviewer 1 comment 2): does a donor's gift depart from
the gift channel more than a random gift of the same structure would?

Null. The donor's gift keeps its own structure: the same number of artist
clusters and the same number of coded records in each. Every cluster is given
the label (woman / not, non-Western / not) of an artist cluster drawn at random
from other donors' gifts. In the matched version the draw comes from the same
department and acquisition decade as the donor's cluster; in the pooled version
from the whole channel. A gift by one artist is therefore one draw, and a gift
spread over many artists is many.

Test. The donor's observed record share is compared with 10,000 null shares.
The p-value is two-sided (twice the smaller tail, so departures below the
expected share count as much as departures above it). Across units the
Benjamini-Hochberg procedure controls the false discovery rate at 5%.

Outputs (output/): rev_null_units.csv, rev_null_summary.csv
"""

import importlib.util
import os

import numpy as np
import pandas as pd

from coding import origin_record
from rev_utils import RNG_SEED, load_gifts, out

N_DRAWS = 10000
MIN_RECORDS, MIN_CODED, MIN_POOL = 100, 30, 20
FDR = 0.05


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out_ = np.empty_like(q)
    out_[order] = np.minimum(q, 1.0)
    return out_


def run(df, unit_col, cluster_col, dept_col, flag, valid, label, share_name, rng):
    d = df[df[valid] == 1].copy()
    d["decade"] = (d["year"] // 10) * 10
    sizes = df[unit_col].value_counts()
    p_channel = d[flag].mean()
    # One row per (cluster, department, decade, unit) with the cluster's label.
    cu = d.groupby([cluster_col, dept_col, "decade", unit_col]).agg(n=(flag, "size"), lab=(flag, "max")).reset_index()
    # Pools: distinct clusters present in a cell / department / channel, with label.
    cell_pool = cu.groupby([dept_col, "decade", cluster_col])["lab"].max().reset_index()
    dept_pool = cu.groupby([dept_col, cluster_col])["lab"].max().reset_index()
    all_pool = cu.groupby([cluster_col])["lab"].max().reset_index()
    cell_groups = {k: v for k, v in cell_pool.groupby([dept_col, "decade"])}
    dept_groups = {k: v for k, v in dept_pool.groupby(dept_col)}
    rows = []
    for unit, own in cu.groupby(unit_col):
        n_coded = int(own["n"].sum())
        if sizes.get(unit, 0) < MIN_RECORDS or n_coded < MIN_CODED:
            continue
        own_clusters = set(own[cluster_col])
        # The donor's clusters, each placed in the cell where the donor has most of its records.
        oc = own.sort_values("n", ascending=False).groupby(cluster_col).agg(
            n=("n", "sum"), lab=("lab", "max"), dept=(dept_col, "first"), decade=("decade", "first")).reset_index()
        obs = float((oc["n"] * oc["lab"]).sum() / oc["n"].sum())
        w = oc["n"].to_numpy(dtype=float) / oc["n"].sum()
        res = {"museum": label, "share": share_name, "unit": unit, "records": int(sizes[unit]), "coded": n_coded,
               "clusters": len(oc), "obs_share": 100 * obs, "channel_share": 100 * p_channel}
        cache = {}
        for version in ("matched", "pooled"):
            q = np.empty(len(oc))
            for i, r in enumerate(oc.itertuples()):
                key = (version, r.dept, r.decade) if version == "matched" else ("pooled",)
                if key not in cache:
                    cand = None
                    if version == "matched":
                        c = cell_groups.get((r.dept, r.decade))
                        if c is not None:
                            c = c[~c[cluster_col].isin(own_clusters)]
                            if len(c) >= MIN_POOL:
                                cand = c
                        if cand is None:
                            c = dept_groups.get(r.dept)
                            if c is not None:
                                c = c[~c[cluster_col].isin(own_clusters)]
                                if len(c) >= MIN_POOL:
                                    cand = c
                    if cand is None:
                        cand = all_pool[~all_pool[cluster_col].isin(own_clusters)]
                    cache[key] = float(cand["lab"].mean())
                q[i] = cache[key]
            sims = (rng.random((N_DRAWS, len(oc))) < q) @ w
            p_up = (1 + np.sum(sims >= obs - 1e-12)) / (N_DRAWS + 1)
            p_dn = (1 + np.sum(sims <= obs + 1e-12)) / (N_DRAWS + 1)
            res.update({f"{version}_expected": 100 * float(q @ w), f"{version}_p": min(1.0, 2 * min(p_up, p_dn)),
                        f"{version}_dir": "up" if obs > float(q @ w) else "down"})
        rows.append(res)
    t = pd.DataFrame(rows)
    for version in ("matched", "pooled"):
        t[f"{version}_q"] = bh(t[f"{version}_p"])
        t[f"{version}_sig"] = t[f"{version}_q"] < FDR
    return t


def load_tate():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "53_rev_tate_patterns.py")
    spec = importlib.util.spec_from_file_location("tate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = mod.load_tate_gifts()
    return t[~t["channel"]].copy()


def main():
    rng = np.random.default_rng(RNG_SEED)
    g = load_gifts().rename(columns={"YearAcquired": "year"})
    alt = g["Nationality"].map(lambda s: origin_record(s, "east_europe_western"))
    g["is_NW_alt"] = (alt == "Non-Western").astype(int)
    g["o_valid_alt"] = alt.isin(["Western", "Non-Western"]).astype(int)
    t = load_tate()
    parts = [
        run(g, "Unit", "Cluster", "Department", "is_F", "g_valid", "MoMA", "female", rng),
        run(g, "Unit", "Cluster", "Department", "is_NW", "o_valid", "MoMA", "non-Western", rng),
        run(g, "Unit", "Cluster", "Department", "is_NW_alt", "o_valid_alt", "MoMA", "non-Western (Eastern Europe as Western)", rng),
        run(t, "Unit", "cid", "dept", "is_F", "g_valid", "Tate", "female", rng),
        run(t, "Unit", "cid", "dept", "is_NW", "o_valid", "Tate", "non-Western", rng),
    ]
    units = pd.concat(parts, ignore_index=True)
    units.to_csv(out("rev_null_units.csv"), index=False)
    rows = []
    for (museum, share), s in units.groupby(["museum", "share"], sort=False):
        r = {"museum": museum, "share": share, "units": len(s)}
        for v in ("matched", "pooled"):
            sig = s[s[f"{v}_sig"]]
            r[f"{v}_up"] = int((sig[f"{v}_dir"] == "up").sum())
            r[f"{v}_down"] = int((sig[f"{v}_dir"] == "down").sum())
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(out("rev_null_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
    print(summary.to_string())
    show = units[units["matched_sig"]].sort_values(["museum", "share", "records"], ascending=[True, True, False])
    print(show[["museum", "share", "unit", "records", "clusters", "obs_share", "matched_expected", "matched_q", "matched_dir",
                "pooled_q"]].round(3).to_string())


if __name__ == "__main__":
    main()

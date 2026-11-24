# -*- coding: utf-8 -*-
"""
Revision analysis A2b (reviewer 1 comment 2): size-matched random-gift null.

A donor's gap from the gift-channel baseline is compared with the gaps of
random 'gifts' of the same size drawn from the channel. Random gifts are built
artist cluster by artist cluster (all records of a sampled cluster are taken
together) until they reach the donor's number of coded records, so the null
keeps the dependence among records of one artist. For each donor unit with at
least 100 records and 30 coded records the script reports the observed
absolute gap, the 95th percentile of the random gaps, and their ratio to the
mean random gap. The same is done for the Tate.

Outputs (output/): rev_size_null_units.csv, rev_size_null_summary.csv
"""

import importlib.util
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rev_utils import RNG_SEED, load_gifts, out

N_DRAWS = 1000
MIN_RECORDS, MIN_CODED = 100, 30


def null_table(df, unit_col, cluster_col, flag, valid, label, rng):
    d = df[df[valid] == 1]
    p0 = d[flag].mean()
    cl = d.groupby(cluster_col)[flag].agg(["sum", "count"]).to_numpy(dtype=float)
    k = len(cl)
    sizes = df[unit_col].value_counts()
    rows = []
    for unit, s in d.groupby(unit_col):
        n = len(s)
        if sizes.get(unit, 0) < MIN_RECORDS or n < MIN_CODED:
            continue
        obs = abs(s[flag].mean() - p0)
        sims = np.empty(N_DRAWS)
        for i in range(N_DRAWS):
            idx = rng.permutation(k)
            cs = np.cumsum(cl[idx, 1])
            j = int(np.searchsorted(cs, n)) + 1
            pick = cl[idx[:j]]
            sims[i] = abs(pick[:, 0].sum() / pick[:, 1].sum() - p0)
        rows.append({"museum": label, "dimension": flag, "unit": unit, "records": int(sizes[unit]), "coded": n,
                     "obs_gap": 100 * obs, "null_p95": 100 * np.percentile(sims, 95),
                     "null_mean": 100 * sims.mean(), "ratio": obs / sims.mean() if sims.mean() > 0 else np.nan,
                     "exceeds": bool(obs > np.percentile(sims, 95))})
    return pd.DataFrame(rows)


def load_tate():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "53_rev_tate_patterns.py")
    spec = importlib.util.spec_from_file_location("tate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = mod.load_tate_gifts()
    return t[~t["channel"]].copy()


def main():
    rng = np.random.default_rng(RNG_SEED)
    g = load_gifts()
    t = load_tate()
    parts = []
    for flag, valid in (("is_F", "g_valid"), ("is_NW", "o_valid")):
        parts.append(null_table(g, "Unit", "Cluster", flag, valid, "MoMA", rng))
        parts.append(null_table(t, "Unit", "cid", flag, valid, "Tate", rng))
    units = pd.concat(parts, ignore_index=True)
    units.to_csv(out("rev_size_null_units.csv"), index=False)
    rows = []
    for (museum, dim), s in units.groupby(["museum", "dimension"]):
        rho = spearmanr(s["records"], s["ratio"]).statistic if len(s) >= 8 else np.nan
        rho_raw = spearmanr(s["records"], s["obs_gap"]).statistic if len(s) >= 8 else np.nan
        rows.append({"museum": museum, "dimension": dim, "units": len(s), "exceed_p95": int(s["exceeds"].sum()),
                     "median_ratio": s["ratio"].median(), "rho_size_ratio": rho, "rho_size_rawgap": rho_raw})
    summary = pd.DataFrame(rows)
    summary.to_csv(out("rev_size_null_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary.round(2).to_string())
    print(units[units["exceeds"]].sort_values(["museum", "dimension", "records"], ascending=[True, True, False])
          .round(1).to_string())


if __name__ == "__main__":
    main()

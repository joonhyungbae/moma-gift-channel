# -*- coding: utf-8 -*-
"""
Revision analysis A2b (reviewer 1 comment 2): does a donor's gift depart from
the gift channel more than a random gift of the same structure would?

Null. The donor's gift keeps its own structure: the same artist clusters and
the same number of coded records in each. Every cluster is given the label
(woman / not, non-Western / not) of a record drawn at random from other donors'
gifts. In the matched version the record comes from the same department and
acquisition decade as the donor's records of that cluster; in the pooled
version from the whole channel. A gift by one artist is therefore one draw,
and a gift spread over many artists is many.

Because records (not artists) are drawn, the expected share of a random gift
is the record share of other donors' gifts: in the matched version it equals
the matched baseline of script 50 (same cells, same 30-record fallback to the
department), and in the pooled version the channel share without the donor.

Test. The donor's observed record share is compared with 100,000 null shares.
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

N_DRAWS = 100000
CHUNK = 10000
MIN_RECORDS, MIN_CODED, MIN_CELL = 100, 30, 30
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
    tot_s, tot_n = float(d[flag].sum()), len(d)
    cell_all = d.groupby([dept_col, "decade"])[flag].agg(["sum", "count"])
    dept_all = d.groupby(dept_col)[flag].agg(["sum", "count"])
    rows = []
    for unit, own in d.groupby(unit_col):
        if sizes.get(unit, 0) < MIN_RECORDS or len(own) < MIN_CODED:
            continue
        own_cell = own.groupby([dept_col, "decade"])[flag].agg(["sum", "count"])
        own_dept = own.groupby(dept_col)[flag].agg(["sum", "count"])
        p_pooled = (tot_s - own[flag].sum()) / (tot_n - len(own))

        def other_share(dept, dec):
            """Share among other donors' coded records in the cell (department if the cell is thin)."""
            s = cell_all.loc[(dept, dec), "sum"] - own_cell.loc[(dept, dec), "sum"]
            n = cell_all.loc[(dept, dec), "count"] - own_cell.loc[(dept, dec), "count"]
            if n >= MIN_CELL:
                return s / n
            s = dept_all.loc[dept, "sum"] - own_dept.loc[dept, "sum"]
            n = dept_all.loc[dept, "count"] - own_dept.loc[dept, "count"]
            return s / n if n > 0 else p_pooled

        cells = own.groupby([cluster_col, dept_col, "decade"]).agg(n=(flag, "size"), lab=(flag, "max")).reset_index()
        cells["p"] = [other_share(a, b) for a, b in zip(cells[dept_col], cells["decade"])]
        cells["np"] = cells["n"] * cells["p"]
        oc = cells.groupby(cluster_col).agg(n=("n", "sum"), lab=("lab", "max"), np_=("np", "sum"))
        n = oc["n"].to_numpy(dtype=float)
        w = n / n.sum()
        obs = float((n * oc["lab"].to_numpy()).sum() / n.sum())
        res = {"museum": label, "share": share_name, "unit": unit, "records": int(sizes[unit]), "coded": int(len(own)),
               "clusters": len(oc), "obs_share": 100 * obs, "channel_share": 100 * tot_s / tot_n}
        for version, q in (("matched", oc["np_"].to_numpy() / n), ("pooled", np.full(len(oc), p_pooled))):
            n_up = n_dn = 0
            for _ in range(N_DRAWS // CHUNK):
                sims = (rng.random((CHUNK, len(oc))) < q) @ w
                n_up += int(np.sum(sims >= obs - 1e-12)); n_dn += int(np.sum(sims <= obs + 1e-12))
            p_up, p_dn = (1 + n_up) / (N_DRAWS + 1), (1 + n_dn) / (N_DRAWS + 1)
            exp = float(q @ w)
            res.update({f"{version}_expected": 100 * exp, f"{version}_p": min(1.0, 2 * min(p_up, p_dn)),
                        f"{version}_dir": "up" if obs > exp else "down"})
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
        both = s[s["matched_sig"] & s["pooled_sig"] & (s["matched_dir"] == s["pooled_dir"])]
        r["both_up"] = int((both["matched_dir"] == "up").sum()); r["both_down"] = int((both["matched_dir"] == "down").sum())
        for v in ("matched", "pooled"):
            sig = s[s[f"{v}_sig"]]
            r[f"{v}_up"] = int((sig[f"{v}_dir"] == "up").sum())
            r[f"{v}_down"] = int((sig[f"{v}_dir"] == "down").sum())
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(out("rev_null_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
    print(summary.to_string())
    show = units[units["matched_sig"] & units["pooled_sig"] & (units["matched_dir"] == units["pooled_dir"])]
    show = show.sort_values(["museum", "share", "matched_dir", "records"], ascending=[True, True, False, False])
    print(show[["museum", "share", "unit", "records", "clusters", "obs_share", "matched_expected", "pooled_expected",
                "matched_q", "pooled_q", "matched_dir"]].round(3).to_string())


if __name__ == "__main__":
    main()

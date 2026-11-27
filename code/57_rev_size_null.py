# -*- coding: utf-8 -*-
"""
Revision analysis A2b (reviewer 1 comment 2), supplementary check: how does a
donor's record share compare with random gifts that keep the gift's structure?

Structure kept. Records are grouped by their first-named maker. A random gift
keeps the donor's makers and the number of coded records by each, and gives
every maker one label (woman / not, non-Western / not).

Four designs, because "same structure" does not fix where the label comes from:
  record draw  the label of a record drawn at random from other donors' gifts,
               so the expected share is other donors' share of records;
  artist draw  the label of a maker drawn at random from other donors' gifts,
               so the expected share is other donors' share of makers;
each either matched (same department and acquisition decade as the donor's
records of that maker, department-wide when the cell is thin) or pooled (whole
channel). The matched record draw is centred on the matched baseline of
script 50.

For every unit the script also reports what the check can detect: the number of
makers, the effective number (inverse Herfindahl of record weights), and the
smallest two-sided p-value attainable in each direction (the probability that
every maker draws the same label, doubled). A unit whose smallest attainable
p-value is 0.05 or more cannot be flagged in that direction whatever its share.

p-values are two-sided over 100,000 draws; Benjamini-Hochberg adjusted values
are reported beside the unadjusted ones.

Outputs (output/): rev_null_units.csv, rev_null_summary.csv
"""

import importlib.util
import os

import numpy as np
import pandas as pd

from coding import origin_record
from rev_utils import DOMINANT_SHARE, RNG_SEED, load_gifts, out

N_DRAWS = 100000
CHUNK = 10000
MIN_RECORDS, MIN_CODED, MIN_CELL = 100, 30, 30
ALPHA = 0.05
DESIGNS = ("rec_matched", "rec_pooled", "art_matched", "art_pooled")


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out_ = np.empty_like(q)
    out_[order] = np.minimum(q, 1.0)
    return out_


def run(df, unit_col, maker_col, dept_col, flag, valid, label, share_name, rng):
    d = df[df[valid] == 1].copy()
    d["decade"] = (d["year"] // 10) * 10
    # One label per maker within a unit: a maker whose records carry both labels is split.
    d["cl"] = d[maker_col].astype(str) + "|" + d[flag].astype(str)
    sizes = df[unit_col].value_counts()
    tot_s, tot_n = float(d[flag].sum()), len(d)
    cell_all = d.groupby([dept_col, "decade"])[flag].agg(["sum", "count"])
    dept_all = d.groupby(dept_col)[flag].agg(["sum", "count"])
    # Makers present in each cell / department / channel, with the units that hold them.
    mk = d.groupby([dept_col, "decade", "cl", unit_col])[flag].max().reset_index()
    rows = []
    for unit, own in d.groupby(unit_col):
        if sizes.get(unit, 0) < MIN_RECORDS or len(own) < MIN_CODED:
            continue
        own_cell = own.groupby([dept_col, "decade"])[flag].agg(["sum", "count"])
        own_dept = own.groupby(dept_col)[flag].agg(["sum", "count"])
        rec_pooled = (tot_s - own[flag].sum()) / (tot_n - len(own))
        oth = mk[mk[unit_col] != unit]
        oth_cell = oth.drop_duplicates([dept_col, "decade", "cl"]).groupby([dept_col, "decade"])[flag].agg(["sum", "count"])
        oth_dept = oth.drop_duplicates([dept_col, "cl"]).groupby(dept_col)[flag].agg(["sum", "count"])
        oth_all = oth.drop_duplicates("cl")[flag]
        art_pooled = float(oth_all.mean())

        # Reference: other units' records, leaving out any unit that holds more than
        # DOMINANT_SHARE of a cell (as in rev_utils.matched_expectation).
        ref = d[d[unit_col] != unit]
        cu = ref.groupby([dept_col, "decade", unit_col]).size()
        dom = (cu / cu.groupby(level=[0, 1]).transform("sum"))
        dom = dom[dom > DOMINANT_SHARE].index
        if len(dom):
            ref = ref[~pd.MultiIndex.from_frame(ref[[dept_col, "decade", unit_col]]).isin(dom)]
        ref_cell = ref.groupby([dept_col, "decade"])[flag].agg(["sum", "count"])
        ref_dept = ref.groupby(dept_col)[flag].agg(["sum", "count"])

        def rec_share(dept, dec):
            if (dept, dec) in ref_cell.index and ref_cell.loc[(dept, dec), "count"] >= MIN_CELL:
                return ref_cell.loc[(dept, dec), "sum"] / ref_cell.loc[(dept, dec), "count"]
            if dept in ref_dept.index and ref_dept.loc[dept, "count"] > 0:
                return ref_dept.loc[dept, "sum"] / ref_dept.loc[dept, "count"]
            return rec_pooled

        def art_share(dept, dec):
            if (dept, dec) in oth_cell.index and oth_cell.loc[(dept, dec), "count"] >= MIN_CELL:
                return oth_cell.loc[(dept, dec), "sum"] / oth_cell.loc[(dept, dec), "count"]
            if dept in oth_dept.index and oth_dept.loc[dept, "count"] > 0:
                return oth_dept.loc[dept, "sum"] / oth_dept.loc[dept, "count"]
            return art_pooled

        cells = own.groupby(["cl", dept_col, "decade"]).agg(n=(flag, "size"), lab=(flag, "max")).reset_index()
        cells["rp"] = [rec_share(a, b) for a, b in zip(cells[dept_col], cells["decade"])]
        cells["ap"] = [art_share(a, b) for a, b in zip(cells[dept_col], cells["decade"])]
        cells["nrp"], cells["nap"] = cells["n"] * cells["rp"], cells["n"] * cells["ap"]
        oc = cells.groupby("cl").agg(n=("n", "sum"), lab=("lab", "max"), nrp=("nrp", "sum"), nap=("nap", "sum"))
        n = oc["n"].to_numpy(dtype=float)
        w = n / n.sum()
        obs = float((n * oc["lab"].to_numpy()).sum() / n.sum())
        res = {"museum": label, "share": share_name, "unit": unit, "records": int(sizes[unit]), "coded": int(len(own)),
               "makers": len(oc), "effective_makers": float(1.0 / np.sum(w ** 2)), "obs_share": 100 * obs}
        qs = {"rec_matched": oc["nrp"].to_numpy() / n, "rec_pooled": np.full(len(oc), rec_pooled),
              "art_matched": oc["nap"].to_numpy() / n, "art_pooled": np.full(len(oc), art_pooled)}
        for design in DESIGNS:
            q = np.clip(qs[design], 0.0, 1.0)
            n_up = n_dn = 0
            for _ in range(N_DRAWS // CHUNK):
                sims = (rng.random((CHUNK, len(oc))) < q) @ w
                n_up += int(np.sum(sims >= obs - 1e-12)); n_dn += int(np.sum(sims <= obs + 1e-12))
            p_up, p_dn = (1 + n_up) / (N_DRAWS + 1), (1 + n_dn) / (N_DRAWS + 1)
            exp = float(q @ w)
            with np.errstate(divide="ignore"):
                min_dn = min(1.0, 2 * float(np.exp(np.sum(np.log1p(-q)))))   # every maker draws 'not'
                min_up = min(1.0, 2 * float(np.exp(np.sum(np.log(q)))))      # every maker draws the label
            res.update({f"{design}_expected": 100 * exp, f"{design}_p": min(1.0, 2 * min(p_up, p_dn)),
                        f"{design}_dir": "up" if obs > exp else "down",
                        f"{design}_min_p_down": min_dn, f"{design}_min_p_up": min_up})
        rows.append(res)
    t = pd.DataFrame(rows)
    for design in DESIGNS:
        t[f"{design}_q"] = bh(t[f"{design}_p"])
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
    g["Maker"] = g["Cluster"].str.split(",").str[0]
    alt = g["Nationality"].map(lambda s: origin_record(s, "east_europe_western"))
    g["is_NW_alt"] = (alt == "Non-Western").astype(int)
    g["o_valid_alt"] = alt.isin(["Western", "Non-Western"]).astype(int)
    t = load_tate()
    parts = [
        run(g, "Unit", "Maker", "Department", "is_F", "g_valid", "MoMA", "female", rng),
        run(g, "Unit", "Maker", "Department", "is_NW", "o_valid", "MoMA", "non-Western", rng),
        run(g, "Unit", "Maker", "Department", "is_NW_alt", "o_valid_alt", "MoMA", "non-Western (Eastern Europe as Western)", rng),
        run(t, "Unit", "cid", "dept", "is_F", "g_valid", "Tate", "female", rng),
        run(t, "Unit", "cid", "dept", "is_NW", "o_valid", "Tate", "non-Western", rng),
    ]
    units = pd.concat(parts, ignore_index=True)
    units.to_csv(out("rev_null_units.csv"), index=False)
    rows = []
    for (museum, share), s in units.groupby(["museum", "share"], sort=False):
        for design in DESIGNS:
            up, dn = s[f"{design}_dir"] == "up", s[f"{design}_dir"] == "down"
            rows.append({
                "museum": museum, "share": share, "design": design, "units": len(s),
                "raw_up": int((up & (s[f"{design}_p"] < ALPHA)).sum()), "raw_down": int((dn & (s[f"{design}_p"] < ALPHA)).sum()),
                "bh_up": int((up & (s[f"{design}_q"] < ALPHA)).sum()), "bh_down": int((dn & (s[f"{design}_q"] < ALPHA)).sum()),
                "cannot_flag_down": int((s[f"{design}_min_p_down"] >= ALPHA).sum()),
                "cannot_flag_up": int((s[f"{design}_min_p_up"] >= ALPHA).sum()),
            })
        for r_ in rows[-len(DESIGNS):]:
            r_["median_effective_makers"] = float(s["effective_makers"].median())
            r_["units_under_10_makers"] = int((s["makers"] < 10).sum())
            r_["units_one_maker"] = int((s["makers"] == 1).sum())
    summary = pd.DataFrame(rows)
    summary.to_csv(out("rev_null_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
    print(summary.to_string())


if __name__ == "__main__":
    main()

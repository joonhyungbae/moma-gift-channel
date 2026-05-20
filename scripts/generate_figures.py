"""Generate Figures 1-3 for MoMA paper. Needs data/processed_moma_data.csv. Writes to output/."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")
CSV = os.path.join(DATA_DIR, "processed_moma_data.csv")

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 11, "figure.dpi": 150, "savefig.bbox": "tight"})


def decade(y):
    if pd.isna(y):
        return np.nan
    return (int(float(y)) // 10) * 10


def wilson_ci(s, n):
    if n == 0:
        return np.nan, np.nan
    z, p = 1.96, s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, c - m), min(1, c + m)


def extract_donor(cl):
    if pd.isna(cl):
        return np.nan
    s = str(cl).strip().lower()
    for phrase in ("gift of the ", "bequest of the ", "gift of ", "bequest of "):
        i = s.find(phrase)
        if i >= 0:
            r = s[i + len(phrase):].strip()
            for sep in (",", "\n"):
                if sep in r:
                    r = r.split(sep)[0].strip()
            return r if r else np.nan
    return np.nan


def fig1():
    usecols = [c for c in ["AcquisitionType", "Gender_Grouped", "YearAcquired"] if c in pd.read_csv(CSV, nrows=0).columns]
    if len(usecols) < 3:
        return
    reader = pd.read_csv(CSV, usecols=usecols, chunksize=50000, engine="python", on_bad_lines="skip")
    rows = []
    for ch in reader:
        ch = ch[ch["AcquisitionType"].isin(["Purchase", "Gift"])]
        ch["Decade"] = ch["YearAcquired"].apply(decade)
        ch = ch[ch["Decade"].notna()]
        ch["F"] = (ch["Gender_Grouped"] == "Female").astype(int)
        for (dec, typ), g in ch.groupby(["Decade", "AcquisitionType"]):
            rows.append({"Decade": dec, "Type": typ, "N": len(g), "F": g["F"].sum()})
    df = pd.DataFrame(rows).groupby(["Decade", "Type"], as_index=False).agg({"N": "sum", "F": "sum"})
    df["Pct"] = 100 * df["F"] / df["N"]
    df["Lo"] = df.apply(lambda r: 100 * wilson_ci(r["F"], r["N"])[0], axis=1)
    df["Hi"] = df.apply(lambda r: 100 * wilson_ci(r["F"], r["N"])[1], axis=1)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for typ in ["Purchase", "Gift"]:
        s = df[df["Type"] == typ].sort_values("Decade")
        if len(s):
            ax.plot(s["Decade"], s["Pct"], "o-", label=typ, markersize=4)
            ax.fill_between(s["Decade"], s["Lo"], s["Hi"], alpha=0.25)
    ax.set_xlabel("Acquisition decade (year)")
    ax.set_ylabel("% works by female artists")
    ax.set_title("Figure 1: Female artist share by acquisition type and decade")
    ax.legend()
    ax.set_ylim(0, None)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "fig1_female_by_decade.pdf"))
    plt.close(fig)
    print("Saved fig1_female_by_decade.pdf")


def fig2():
    usecols = [c for c in ["AcquisitionType", "GeographicOrigin", "YearAcquired"] if c in pd.read_csv(CSV, nrows=0).columns]
    if len(usecols) < 3:
        return
    reader = pd.read_csv(CSV, usecols=usecols, chunksize=50000, engine="python", on_bad_lines="skip")
    rows = []
    for ch in reader:
        ch = ch[ch["AcquisitionType"].isin(["Purchase", "Gift"])]
        ch = ch[ch["GeographicOrigin"].isin(["Western", "Non-Western"])]
        ch["Decade"] = ch["YearAcquired"].apply(decade)
        ch = ch[ch["Decade"].notna()]
        ch["NW"] = (ch["GeographicOrigin"] == "Non-Western").astype(int)
        for (dec, typ), g in ch.groupby(["Decade", "AcquisitionType"]):
            rows.append({"Decade": dec, "Type": typ, "N": len(g), "NW": g["NW"].sum()})
    df = pd.DataFrame(rows).groupby(["Decade", "Type"], as_index=False).agg({"N": "sum", "NW": "sum"})
    df["Pct"] = 100 * df["NW"] / df["N"]
    df["Lo"] = df.apply(lambda r: 100 * wilson_ci(r["NW"], r["N"])[0], axis=1)
    df["Hi"] = df.apply(lambda r: 100 * wilson_ci(r["NW"], r["N"])[1], axis=1)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for typ in ["Purchase", "Gift"]:
        s = df[df["Type"] == typ].sort_values("Decade")
        if len(s):
            ax.plot(s["Decade"], s["Pct"], "o-", label=typ, markersize=4)
            ax.fill_between(s["Decade"], s["Lo"], s["Hi"], alpha=0.25)
    ax.set_xlabel("Acquisition decade (year)")
    ax.set_ylabel("% works by non-Western artists")
    ax.set_title("Figure 2: Non-Western artist share by acquisition type and decade")
    ax.legend()
    ax.set_ylim(0, None)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "fig2_nonwestern_by_decade.pdf"))
    plt.close(fig)
    print("Saved fig2_nonwestern_by_decade.pdf")


def fig3():
    usecols = [c for c in ["AcquisitionType", "CreditLine"] if c in pd.read_csv(CSV, nrows=0).columns]
    if "CreditLine" not in usecols:
        return
    reader = pd.read_csv(CSV, usecols=usecols, chunksize=50000, engine="python", on_bad_lines="skip")
    cnt = Counter()
    for ch in reader:
        g = ch[ch["AcquisitionType"] == "Gift"].copy()
        g["Donor"] = g["CreditLine"].apply(extract_donor)
        g = g[g["Donor"].notna()]
        for d, n in g["Donor"].value_counts().items():
            cnt[d] += n
    if not cnt:
        return
    counts = np.array(sorted(cnt.values(), reverse=True), dtype=float)
    n_d, tot = len(counts), counts.sum()
    cum_d = np.arange(1, n_d + 1, dtype=float) / n_d
    cum_g = np.cumsum(counts) / tot
    top1_n = max(1, int(np.ceil(n_d * 0.01)))
    top1_sh = counts[:top1_n].sum() / tot
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0] + list(cum_d), [0] + list(cum_g), "b-", lw=1.5, label="Lorenz curve")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Equality")
    ax.axvline(0.01, color="gray", linestyle=":")
    ax.axhline(top1_sh, color="gray", linestyle=":")
    ax.scatter([0.01], [top1_sh], color="red", s=40, label="Top 1%% = %.1f%%" % (100 * top1_sh))
    ax.set_xlabel("Cumulative share of donors")
    ax.set_ylabel("Cumulative share of gifted works")
    ax.set_title("Figure 3: Donor concentration (Lorenz curve)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "fig3_lorenz.pdf"))
    plt.close(fig)
    print("Saved fig3_lorenz.pdf")


def main():
    if not os.path.isfile(CSV):
        print("Missing:", CSV, "- run 01_data_preprocessing.py first.", file=sys.stderr)
        sys.exit(1)
    fig1()
    fig2()
    fig3()
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()

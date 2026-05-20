# -*- coding: utf-8 -*-
"""
Experiment 16c: publication-ready figure suite for the six donor biographies.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAIN_DEPARTMENTS, get_output_path
from donor_bio_utils import (
    CASE_ORDER,
    CASE_SHORT,
    CASE_SLUGS,
    assign_case,
    demographic_stats,
    ensure_output_dir,
    load_moma_donor_gifts,
)
from experiment_utils import append_log


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GRAY = ["#111111", "#3b3b3b", "#666666", "#8a8a8a", "#b0b0b0", "#d0d0d0"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "P", "X"]
HATCHES = ["", "///", "\\\\\\", "...", "xx"]


def save_dual(fig, stem):
    pdf = get_output_path(f"{stem}.pdf")
    eps = get_output_path(f"{stem}.eps")
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(eps, bbox_inches="tight")
    plt.close(fig)
    return pdf, eps


def fig_timeline(profile):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y_positions = np.arange(len(profile))
    for i, (_, row) in enumerate(profile.iterrows()):
        ax.barh(
            i,
            row["active_era_end"] - row["active_era_start"],
            left=row["active_era_start"],
            height=0.48,
            color=GRAY[i],
            edgecolor="black",
            linewidth=0.7,
        )
        ax.plot(row["peak_year"], i, marker="|", color="black", markersize=12, markeredgewidth=1.5)
        ax.text(row["active_era_end"] + 0.8, i, f"N={int(row['n_gifts'])}", va="center", fontsize=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([CASE_SHORT[d] for d in profile["donor"]])
    ax.set_xlim(1929, 2024)
    ax.set_xlabel("Acquisition year")
    ax.set_title("Six donor active eras and peak years")
    ax.grid(axis="x", color="0.85", linewidth=0.6)
    ax.invert_yaxis()
    return save_dual(fig, "fig_donor_era_timeline")


def fig_dept_footprint(footprint):
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    labels = [CASE_SHORT[d] for d in footprint["donor"]]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    dept_colors = ["#111111", "#4a4a4a", "#777777", "#a0a0a0", "#c9c9c9"]
    for i, dept in enumerate(MAIN_DEPARTMENTS):
        counts = footprint[f"{dept}_n"].to_numpy(dtype=float)
        ax.bar(
            x,
            counts,
            bottom=bottom,
            label=dept,
            color=dept_colors[i],
            edgecolor="black",
            linewidth=0.4,
            hatch=HATCHES[i],
        )
        bottom += counts
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Gift count")
    ax.set_title("Department footprint of six donor cases")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    return save_dual(fig, "fig_donor_dept_footprint")


def fig_demographic_gap(gap):
    label_offsets = {
        "Kleiner":         {"xytext": (24, 22),  "ha": "left",   "va": "bottom", "leader": True},
        "Rockefeller":     {"xytext": (-24, -22),"ha": "right",  "va": "top",    "leader": True},
        "Pigozzi":         {"xytext": (16, -2),  "ha": "left",   "va": "center", "leader": False},
        "Cohen":           {"xytext": (-14, 10), "ha": "right",  "va": "bottom", "leader": False},
        "Rothschild Fdn.": {"xytext": (16, -2),  "ha": "left",   "va": "center", "leader": False},
        "Gund":            {"xytext": (14, 6),   "ha": "left",   "va": "bottom", "leader": False},
    }

    # Per-donor CIs (on the share) come from the biography profile; the gap is
    # share - baseline, so the CI half-widths transfer directly to the gap point.
    prof = pd.read_csv(get_output_path("donor_biography_profile.csv"))
    ci = {
        r["donor"]: (
            r["pct_female"], r["female_ci_lower"], r["female_ci_upper"],
            r["pct_nonwest"], r["nonwest_ci_lower"], r["nonwest_ci_upper"],
        )
        for _, r in prof.iterrows()
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, (_, row) in enumerate(gap.iterrows()):
        size = 30 + np.sqrt(row["n_gifts"]) * 7
        x, y = row["female_gap_pp"], row["nonwest_gap_pp"]
        if row["donor"] in ci:
            fpct, flo, fhi, npct, nlo, nhi = ci[row["donor"]]
            xerr = [[max(0.0, fpct - flo)], [max(0.0, fhi - fpct)]]
            yerr = [[max(0.0, npct - nlo)], [max(0.0, nhi - npct)]]
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                fmt="none", ecolor="0.6", elinewidth=0.8,
                capsize=2, capthick=0.8, zorder=2,
            )
        ax.scatter(
            x, y, s=size,
            color=GRAY[i], edgecolor="black",
            marker=MARKERS[i], linewidth=0.8, zorder=3,
        )
        label = CASE_SHORT[row["donor"]]
        opts = label_offsets[label]
        arrowprops = (
            dict(arrowstyle="-", color="0.4", linewidth=0.6, shrinkA=0, shrinkB=4)
            if opts["leader"] else None
        )
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=opts["xytext"],
            ha=opts["ha"], va=opts["va"],
            fontsize=8.5, zorder=4,
            arrowprops=arrowprops,
        )
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.axvline(0, color="0.45", linewidth=0.8)
    ax.set_xlabel("Female share gap from baseline (pp)")
    ax.set_ylabel("Non-Western share gap from baseline (pp)")
    ax.set_title("Donor demographic gaps from museum-wide gift baseline")
    ax.grid(color="0.88", linewidth=0.6)
    ax.margins(x=0.12, y=0.12)
    return save_dual(fig, "fig_donor_demographic_gap")


def fig_cumulative(profile):
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for i, (_, row) in enumerate(profile.iterrows()):
        slug = row["slug"]
        cum = pd.read_csv(get_output_path(f"cumulative_{slug}.csv"))
        ax.plot(
            cum["year"],
            cum["cumulative_gifts"],
            label=CASE_SHORT[row["donor"]],
            color=GRAY[i],
            linestyle=LINESTYLES[i],
            linewidth=1.8,
        )
    ax.set_xlim(1929, 2024)
    ax.set_xlabel("Acquisition year")
    ax.set_ylabel("Cumulative gifts")
    ax.set_title("Cumulative gifts by donor case")
    ax.legend(frameon=False, ncol=2)
    ax.grid(color="0.88", linewidth=0.6)
    return save_dual(fig, "fig_donor_cumulative")


def condition_stats(base_df):
    cases = [("Actual", pd.Series(False, index=base_df.index))]
    for donor in CASE_ORDER:
        cases.append((f"-{CASE_SHORT[donor]}", base_df["CaseDonor"] == donor))
    cases.append(("-All six", base_df["CaseDonor"].isin(CASE_ORDER)))
    rows = []
    for label, remove_mask in cases:
        d = base_df.loc[~remove_mask]
        stats = demographic_stats(d)
        rows.append(
            {
                "condition": label,
                "n": int(len(d)),
                "female_pct": stats["pct_female"],
                "female_ci_lower": stats["female_ci_lower"],
                "female_ci_upper": stats["female_ci_upper"],
                "nonwest_pct": stats["pct_nonwest"],
                "nonwest_ci_lower": stats["nonwest_ci_lower"],
                "nonwest_ci_upper": stats["nonwest_ci_upper"],
            }
        )
    return pd.DataFrame(rows)


def fig_joint_counterfactual(base_df):
    cf = condition_stats(base_df)
    x = np.arange(len(cf))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.1, 4.4))
    female_yerr = np.vstack(
        [
            cf["female_pct"] - cf["female_ci_lower"],
            cf["female_ci_upper"] - cf["female_pct"],
        ]
    )
    nonwest_yerr = np.vstack(
        [
            cf["nonwest_pct"] - cf["nonwest_ci_lower"],
            cf["nonwest_ci_upper"] - cf["nonwest_pct"],
        ]
    )
    ax.bar(
        x - width / 2,
        cf["female_pct"],
        width,
        yerr=female_yerr,
        label="Female",
        color="#4f4f4f",
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
    )
    ax.bar(
        x + width / 2,
        cf["nonwest_pct"],
        width,
        yerr=nonwest_yerr,
        label="Non-Western",
        color="#b7b7b7",
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        capsize=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cf["condition"], rotation=30, ha="right")
    ax.set_ylabel("Institutional gift share (%)")
    ax.set_title("Institutional demographic shares under donor-removal counterfactuals")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    return save_dual(fig, "fig_six_donor_jointcf")


def main():
    ensure_output_dir()
    profile = pd.read_csv(get_output_path("donor_biography_profile.csv"))
    profile["order"] = profile["donor"].map({donor: i for i, donor in enumerate(CASE_ORDER)})
    profile = profile.sort_values("order").drop(columns=["order"])
    footprint = pd.read_csv(get_output_path("donor_dept_footprint.csv"))
    footprint["order"] = footprint["donor"].map({donor: i for i, donor in enumerate(CASE_ORDER)})
    footprint = footprint.sort_values("order").drop(columns=["order"])
    gap = pd.read_csv(get_output_path("donor_demographic_gap.csv"))
    gap["order"] = gap["donor"].map({donor: i for i, donor in enumerate(CASE_ORDER)})
    gap = gap.sort_values("order").drop(columns=["order"])
    base_df = assign_case(load_moma_donor_gifts())

    outputs = []
    for maker in [
        lambda: fig_timeline(profile),
        lambda: fig_dept_footprint(footprint),
        lambda: fig_demographic_gap(gap),
        lambda: fig_cumulative(profile),
        lambda: fig_joint_counterfactual(base_df),
    ]:
        outputs.extend(maker())

    append_log(
        "Experiment 16c -- Donor figure suite",
        [
            "Saved figures: " + "; ".join(outputs),
            "All figures were written as PDF and EPS with grayscale-friendly encodings.",
        ],
    )
    print(f"Completed Experiment 16c | files={len(outputs)}")


if __name__ == "__main__":
    main()

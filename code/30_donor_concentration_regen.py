# -*- coding: utf-8 -*-
"""
Experiment 11 wrapper: regenerate donor-concentration table under the patched
donor parser.

The existing 05_donor_concentration.py is still the canonical pipeline script,
but it does not write the exact LaTeX table that the manuscript now inputs.
This wrapper computes the main-department 1930-2024 gift sample and writes:

  - output/donor_concentration_regen.csv
  - output/table_concentration_regen.tex
  - output/table4_concentration.csv
  - output/donor_concentration_summary.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAIN_DEPARTMENTS, get_output_path
from experiment_utils import append_log, load_base_data, safe_pct
from utils import extract_donor_from_gift, gini_coefficient


def latex_escape(value):
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def donor_stats(names):
    vc = pd.Series(names).dropna().astype(str).value_counts()
    if len(vc) == 0:
        return {
            "N_gifts": 0,
            "N_donors": 0,
            "Gini": np.nan,
            "Top1_n_donors": 0,
            "Top1_share_pct": np.nan,
            "Top_donor_name": "",
            "Top_donor_share_pct": np.nan,
        }
    top1_n = max(1, int(np.ceil(len(vc) * 0.01)))
    return {
        "N_gifts": int(vc.sum()),
        "N_donors": int(len(vc)),
        "Gini": round(float(gini_coefficient(vc.values)), 3),
        "Top1_n_donors": int(top1_n),
        "Top1_share_pct": round(float(100 * vc.head(top1_n).sum() / vc.sum()), 1),
        "Top_donor_name": vc.index[0],
        "Top_donor_share_pct": round(float(100 * vc.iloc[0] / vc.sum()), 1),
    }


def write_table(rows):
    lines = [
        "\\begin{table}",
        "\\centering",
        "\\tbl{Donor concentration metrics by department (gift channel)}",
        "{\\begin{tabular}{lccc}",
        "\\toprule",
        "Department & Gini & Top 1\\% share & Top donor share \\\\",
        "\\midrule",
    ]
    for r in rows:
        top_donor = "---"
        if r["Top_donor_name"]:
            top_donor = f"{r['Top_donor_share_pct']:.1f}\\% ({latex_escape(r['Top_donor_name'])})"
        lines.append(
            f"{latex_escape(r['Department'])} & {r['Gini']:.3f} & "
            f"{r['Top1_share_pct']:.1f}\\% & {top_donor} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\footnotesize{Donors are extracted from gift credit lines after filtering generic role labels such as architect, designer, and manufacturer. The sample is donor-identified gifts in the main departments, 1930-2024.}",
            "\\label{tab:concentration}",
            "\\end{table}",
        ]
    )
    with open(get_output_path("table_concentration_regen.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df = load_base_data(require_birthyear=False)
    gifts = df[df["AcquisitionType"] == "Gift"].copy()
    gifts["Donor"] = gifts["CreditLine"].apply(extract_donor_from_gift)
    gifts = gifts[gifts["Donor"].notna()].copy()

    overall = donor_stats(gifts["Donor"])
    vc = gifts["Donor"].value_counts()
    top_donors = set(vc.head(overall["Top1_n_donors"]).index)
    excl = gifts[~gifts["Donor"].isin(top_donors)].copy()
    female_valid = excl["Gender_Grouped"].isin(["Female", "Male"])
    geo_valid = excl["GeographicOrigin"].notna()
    overall.update(
        {
            "excl_n": int(len(excl)),
            "excl_pct_female": round(
                safe_pct((excl.loc[female_valid, "Gender_Grouped"] == "Female").sum(), female_valid.sum()), 2
            ),
            "excl_pct_nw": round(
                safe_pct((excl.loc[geo_valid, "GeographicOrigin"] == "Non-Western").sum(), geo_valid.sum()), 2
            ),
        }
    )

    rows = []
    for dept in MAIN_DEPARTMENTS:
        stats = donor_stats(gifts.loc[gifts["Department"] == dept, "Donor"])
        rows.append({"Department": dept, **stats})

    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("donor_concentration_regen.csv"), index=False)

    canonical = out[
        [
            "Department",
            "Gini",
            "Top1_share_pct",
            "Top_donor_share_pct",
            "Top_donor_name",
        ]
    ].rename(
        columns={
            "Top1_share_pct": "Top 1% share",
            "Top_donor_share_pct": "Top donor share",
            "Top_donor_name": "Top donor name",
        }
    )
    canonical["Top 1% share"] = canonical["Top 1% share"].map(lambda x: f"{x:.1f}%")
    canonical["Top donor share"] = canonical["Top donor share"].map(lambda x: f"{x:.1f}%")
    canonical.to_csv(get_output_path("table4_concentration.csv"), index=False)

    pd.DataFrame(
        [
            {
                "top1_n_donors": overall["Top1_n_donors"],
                "top1_share_pct": overall["Top1_share_pct"],
                "n_gifts": overall["N_gifts"],
                "excl_n": overall["excl_n"],
                "excl_pct_female": overall["excl_pct_female"],
                "excl_pct_nw": overall["excl_pct_nw"],
            }
        ]
    ).to_csv(get_output_path("donor_concentration_summary.csv"), index=False)

    write_table(rows)

    if any(str(r["Top_donor_name"]).lower() == "architect" for r in rows):
        raise RuntimeError("Patched donor parser still returned 'architect' as a top donor.")

    append_log(
        "Experiment 11 -- Donor concentration regen",
        [
            f"Saved CSV: {get_output_path('donor_concentration_regen.csv')}",
            f"Refreshed canonical CSV: {get_output_path('table4_concentration.csv')}",
            f"Refreshed summary CSV: {get_output_path('donor_concentration_summary.csv')}",
            f"Saved TEX: {get_output_path('table_concentration_regen.tex')}",
            f"Overall donor-identified gifts={overall['N_gifts']:,}; unique donors={overall['N_donors']:,}; top 1% n={overall['Top1_n_donors']}; top 1% share={overall['Top1_share_pct']:.1f}%.",
            f"Excluding top 1% leaves N={overall['excl_n']:,}, female={overall['excl_pct_female']:.2f}%, non-Western={overall['excl_pct_nw']:.2f}%.",
            f"Architecture & Design top donor is {rows[0]['Top_donor_name']} ({rows[0]['Top_donor_share_pct']:.1f}%).",
        ],
    )
    print(
        "Completed 30_donor_concentration_regen.py | top1=%s donors, %.1f%% | A&D top=%s"
        % (overall["Top1_n_donors"], overall["Top1_share_pct"], rows[0]["Top_donor_name"])
    )


if __name__ == "__main__":
    main()

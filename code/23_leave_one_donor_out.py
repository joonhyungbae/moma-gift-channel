# -*- coding: utf-8 -*-
"""
Experiment 4: Leave-one-mega-donor-out fragility analysis.

Sample:
  Gift records in main departments, 1930-2024. This is a descriptive
  concentration exercise rather than a regression sample.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_output_path
from experiment_utils import append_log, get_top_donors, load_base_data, safe_pct


def compute_rates(df):
    female_valid = df["Gender_Grouped"].isin(["Female", "Male"])
    geo_valid = df["GeographicOrigin"].notna()
    female_pct = safe_pct((df.loc[female_valid, "Gender_Grouped"] == "Female").sum(), female_valid.sum())
    nonwest_pct = safe_pct((df.loc[geo_valid, "GeographicOrigin"] == "Non-Western").sum(), geo_valid.sum())
    return female_pct, nonwest_pct


def main():
    df = load_base_data(require_birthyear=False)
    gifts = df[df["AcquisitionType"] == "Gift"].copy()
    donor_counts, gifts = get_top_donors(gifts, top_k=25)

    base_female, base_nonwest = compute_rates(gifts)
    rows = []
    for donor, n_donor in donor_counts.items():
        sub = gifts[gifts["Donor"] != donor].copy()
        female_pct, nonwest_pct = compute_rates(sub)
        donor_sub = gifts[gifts["Donor"] == donor].copy()
        depts = donor_sub["Department"].value_counts()
        top_depts = ", ".join(depts.head(2).index.tolist())
        for dept in depts.head(2).index.tolist():
            dept_base = gifts[gifts["Department"] == dept].copy()
            dept_sub = sub[sub["Department"] == dept].copy()
            d_female_base, d_nonwest_base = compute_rates(dept_base)
            d_female_sub, d_nonwest_sub = compute_rates(dept_sub)
            rows.append(
                {
                    "Donor": donor,
                    "Department_scope": dept,
                    "N_gifts_removed": int(n_donor),
                    "DeltaFemale_pp": round(female_pct - base_female, 3),
                    "DeltaNonWest_pp": round(nonwest_pct - base_nonwest, 3),
                    "Dept_DeltaFemale_pp": round(d_female_sub - d_female_base, 3),
                    "Dept_DeltaNonWest_pp": round(d_nonwest_sub - d_nonwest_base, 3),
                    "TopDepartments": top_depts,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(get_output_path("leave_one_donor_out.csv"), index=False)

    # sanity check for the two institutional foundations post-2000
    g2 = gifts[gifts["YearAcquired"] >= 2000].copy()
    inst = g2[g2["CreditLine"].fillna("").str.contains("Foundation", case=False, regex=False)].copy()
    patron = g2[~g2["CreditLine"].fillna("").str.contains("Foundation", case=False, regex=False)].copy()
    base_inst_nonwest = safe_pct((inst["GeographicOrigin"] == "Non-Western").sum(), inst["GeographicOrigin"].notna().sum())
    excl = inst[
        ~inst["CreditLine"].fillna("").str.contains(
            "Judith Rothschild Foundation|Roy Lichtenstein Foundation", case=False, regex=True
        )
    ].copy()
    excl_inst_nonwest = safe_pct((excl["GeographicOrigin"] == "Non-Western").sum(), excl["GeographicOrigin"].notna().sum())

    # figure
    fig_df = (
        out_df.groupby("Donor")[["DeltaFemale_pp", "DeltaNonWest_pp"]]
        .max()
        .reset_index()
        .assign(AbsMove=lambda x: x[["DeltaFemale_pp", "DeltaNonWest_pp"]].abs().max(axis=1))
        .sort_values("AbsMove", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(fig_df))
    ax.barh(y - 0.2, fig_df["DeltaFemale_pp"].abs(), height=0.4, label="|ΔFemale| pp")
    ax.barh(y + 0.2, fig_df["DeltaNonWest_pp"].abs(), height=0.4, label="|ΔNon-West| pp")
    ax.set_yticks(y)
    ax.set_yticklabels(fig_df["Donor"])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute percentage-point shift after removing donor")
    ax.set_title("Leave-one-donor-out fragility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(get_output_path("fig_donor_fragility.pdf"))

    header = "Donor & Top department(s) & $N$ removed & $\\Delta$Female pp & $\\Delta$Non-West pp \\\\"
    body = [header, "\\midrule"]
    for donor in fig_df["Donor"].tolist():
        row = out_df[out_df["Donor"] == donor].iloc[0]
        body.append(
            f"{row['Donor']} & {row['TopDepartments']} & {int(row['N_gifts_removed']):,} & "
            f"{row['DeltaFemale_pp']:.3f} & {row['DeltaNonWest_pp']:.3f} \\\\"
        )
    tex = (
        "\\begin{table}[p]\n\\centering\n"
        "\\caption{Leave-one-mega-donor-out fragility in gift-channel diversity metrics}\n"
        "\\label{tab:leave_one_donor_out}\n"
        "\\begin{tabular}{p{4.0cm}p{3.2cm}rrr}\n\\toprule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\footnotesize{Percentage-point shifts are computed relative to the institution-wide gift-channel baseline in "
        "main departments, 1930-2024. Positive values indicate that the institution-wide percentage rises when the donor is removed; "
        "negative values indicate that it falls.}\n\\end{table}"
    )
    with open(get_output_path("table_leave_one_donor_out.tex"), "w") as f:
        f.write(tex)

    top_move = fig_df.iloc[0]
    paragraph = (
        f"A leave-one-donor-out fragility analysis confirms that gift-channel diversity metrics are highly donor-dependent. "
        f"Removing the single most consequential donor in the top-twenty-five list shifts the institution-wide gift-channel "
        f"non-Western share by {top_move['DeltaNonWest_pp']:.2f} percentage points in absolute value, while the corresponding "
        f"female-share shift is {top_move['DeltaFemale_pp']:.2f} points. The same script reproduces the paper's post-2000 institutional-gift "
        f"sanity check: excluding the Judith Rothschild and Roy Lichtenstein foundations moves institutional non-Western share from "
        f"{base_inst_nonwest:.2f}\\% to {excl_inst_nonwest:.2f}\\%."
    )
    append_log(
        "Experiment 4 — Leave-one-donor-out fragility",
        [
            f"Saved CSV: {get_output_path('leave_one_donor_out.csv')}",
            f"Saved TEX: {get_output_path('table_leave_one_donor_out.tex')}",
            f"Saved FIG: {get_output_path('fig_donor_fragility.pdf')}",
            f"Sanity check institutional foundation shift: {base_inst_nonwest:.2f}% -> {excl_inst_nonwest:.2f}%",
        ],
        paper_paragraph=paragraph,
    )

    print(
        "Completed 23_leave_one_donor_out.py | donors=%d | top abs nonwest shift=%.3f pp"
        % (len(donor_counts), fig_df["DeltaNonWest_pp"].abs().max())
    )


if __name__ == "__main__":
    main()

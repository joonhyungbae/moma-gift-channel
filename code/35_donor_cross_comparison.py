# -*- coding: utf-8 -*-
"""
Experiment 16b: cross-donor comparative analysis for the six biography cases.
"""

import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAIN_DEPARTMENTS, get_output_path
from donor_bio_utils import (
    BASELINE_FEMALE,
    BASELINE_NONWEST,
    CASE_ORDER,
    assign_case,
    counterfactual_stats,
    demographic_stats,
    donor_artist_set,
    ensure_output_dir,
    fmt_float,
    load_moma_donor_gifts,
)
from experiment_utils import append_log


def write_overlay(case_df):
    rows = []
    for donor in CASE_ORDER:
        d = case_df[case_df["CaseDonor"] == donor]
        counts = d["YearAcquired"].value_counts()
        for year in range(1929, 2025):
            rows.append({"donor": donor, "year": year, "n_gifts": int(counts.get(year, 0))})
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("donor_era_overlay.csv"), index=False)
    return out


def write_dept_footprint(case_df):
    rows = []
    for donor in CASE_ORDER:
        d = case_df[case_df["CaseDonor"] == donor]
        row = {"donor": donor, "n_gifts": int(len(d))}
        counts = d["Department"].value_counts()
        for dept in MAIN_DEPARTMENTS:
            row[dept] = fmt_float(counts.get(dept, 0) / len(d) if len(d) else 0.0, 4)
            row[f"{dept}_n"] = int(counts.get(dept, 0))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("donor_dept_footprint.csv"), index=False)
    return out


def write_demographic_gap(profile):
    rows = []
    for _, row in profile.iterrows():
        rows.append(
            {
                "donor": row["donor"],
                "n_gifts": int(row["n_gifts"]),
                "pct_female": row["pct_female"],
                "female_gap_pp": fmt_float(row["pct_female"] - BASELINE_FEMALE),
                "pct_nonwest": row["pct_nonwest"],
                "nonwest_gap_pp": fmt_float(row["pct_nonwest"] - BASELINE_NONWEST),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("donor_demographic_gap.csv"), index=False)
    return out


def write_pairwise_overlap(case_df):
    artist_sets = {
        donor: donor_artist_set(case_df[case_df["CaseDonor"] == donor]) for donor in CASE_ORDER
    }
    rows = []
    for donor_a, donor_b in itertools.combinations(CASE_ORDER, 2):
        a = artist_sets[donor_a]
        b = artist_sets[donor_b]
        smaller = min(len(a), len(b))
        shared = len(a & b)
        rows.append(
            {
                "donor_a": donor_a,
                "donor_b": donor_b,
                "n_artists_a": int(len(a)),
                "n_artists_b": int(len(b)),
                "n_shared_artists": int(shared),
                "pct_of_smaller_pool": fmt_float(100 * shared / smaller if smaller else 0.0),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("six_donor_overlap.csv"), index=False)
    return out


def write_joint_counterfactual(base_df):
    remove_mask = base_df["CaseDonor"].isin(CASE_ORDER)
    actual = demographic_stats(base_df)
    without = demographic_stats(base_df.loc[~remove_mask])
    cf = counterfactual_stats(base_df, remove_mask)
    row = {
        "actual_n": int(len(base_df)),
        "without_six_n": int((~remove_mask).sum()),
        "six_removed_n": int(remove_mask.sum()),
        "actual_female_pct": fmt_float(actual["pct_female"]),
        "actual_female_ci_lower": fmt_float(actual["female_ci_lower"]),
        "actual_female_ci_upper": fmt_float(actual["female_ci_upper"]),
        "without_six_female_pct": fmt_float(without["pct_female"]),
        "without_six_female_ci_lower": fmt_float(without["female_ci_lower"]),
        "without_six_female_ci_upper": fmt_float(without["female_ci_upper"]),
        "delta_female_pp": fmt_float(cf["delta_female_pp"]),
        "actual_nonwest_pct": fmt_float(actual["pct_nonwest"]),
        "actual_nonwest_ci_lower": fmt_float(actual["nonwest_ci_lower"]),
        "actual_nonwest_ci_upper": fmt_float(actual["nonwest_ci_upper"]),
        "without_six_nonwest_pct": fmt_float(without["pct_nonwest"]),
        "without_six_nonwest_ci_lower": fmt_float(without["nonwest_ci_lower"]),
        "without_six_nonwest_ci_upper": fmt_float(without["nonwest_ci_upper"]),
        "delta_nonwest_pp": fmt_float(cf["delta_nonwest_pp"]),
    }
    out = pd.DataFrame([row])
    out.to_csv(get_output_path("six_donor_joint_counterfactual.csv"), index=False)
    return out


def coverage_stats(profile, case_df, base_df):
    all_years = set(range(1929, 2025))
    active_span_years = set()
    for _, row in profile.iterrows():
        active_span_years.update(range(int(row["active_era_start"]), int(row["active_era_end"]) + 1))
    active_span_years = active_span_years & all_years
    observed_case_years = set(case_df["YearAcquired"].dropna().astype(int).unique())
    observed_base_years = set(base_df["YearAcquired"].dropna().astype(int).unique())
    return {
        "active_span_years": len(active_span_years),
        "active_span_fraction": len(active_span_years) / len(all_years),
        "observed_case_years": len(observed_case_years),
        "observed_base_years": len(observed_base_years),
        "observed_fraction_of_base_gift_years": len(observed_case_years & observed_base_years) / len(observed_base_years),
    }


def main():
    ensure_output_dir()
    base_df = assign_case(load_moma_donor_gifts())
    case_df = base_df[base_df["CaseDonor"].isin(CASE_ORDER)].copy()
    profile = pd.read_csv(get_output_path("donor_biography_profile.csv"))

    overlay = write_overlay(case_df)
    footprint = write_dept_footprint(case_df)
    gap = write_demographic_gap(profile)
    overlap = write_pairwise_overlap(case_df)
    joint = write_joint_counterfactual(base_df)
    coverage = coverage_stats(profile, case_df, base_df)

    bullets = [
        f"Saved active-era overlay CSV: {get_output_path('donor_era_overlay.csv')} ({len(overlay):,} rows)",
        f"Saved department footprint matrix: {get_output_path('donor_dept_footprint.csv')}",
        f"Saved demographic gap matrix: {get_output_path('donor_demographic_gap.csv')}",
        f"Saved six-donor artist-overlap CSV: {get_output_path('six_donor_overlap.csv')} ({len(overlap):,} pairs)",
        f"Saved joint counterfactual CSV: {get_output_path('six_donor_joint_counterfactual.csv')}",
        f"Joint removal drops N from {int(joint.iloc[0]['actual_n']):,} to {int(joint.iloc[0]['without_six_n']):,}; female share shifts by {joint.iloc[0]['delta_female_pp']:+.2f} pp and non-Western share by {joint.iloc[0]['delta_nonwest_pp']:+.2f} pp.",
        f"Active-span coverage: {coverage['active_span_years']}/96 years ({100 * coverage['active_span_fraction']:.1f}%) in 1929-2024 have at least one of the six donors inside its active era.",
        f"Observed gift-year coverage: {coverage['observed_case_years']}/{coverage['observed_base_years']} donor-identified gift years ({100 * coverage['observed_fraction_of_base_gift_years']:.1f}%) include at least one gift from the six donors.",
    ]
    append_log("Experiment 16b -- Cross-donor comparative analysis", bullets)
    print(
        "Completed Experiment 16b | overlay=%s footprint=%s overlap=%s"
        % (len(overlay), footprint.shape, len(overlap))
    )


if __name__ == "__main__":
    main()

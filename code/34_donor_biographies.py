# -*- coding: utf-8 -*-
"""
Experiment 16a: six-donor deep biography analytics.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAIN_DEPARTMENTS, get_output_path
from donor_bio_utils import (
    CASE_ORDER,
    CASE_SHORT,
    CASE_SLUGS,
    assign_case,
    counterfactual_stats,
    demographic_stats,
    dept_counts_and_shares,
    detect_suberas,
    donor_artist_set,
    ensure_output_dir,
    fmt_float,
    footprint_type,
    load_moma_donor_gifts,
    matched_case_strings,
    pct_female,
    pct_nonwest,
    top_breakdown,
)
from experiment_utils import append_log


def pipe_dept_breakdown(rows):
    return "|".join(f"{dept}:{n} ({100 * share:.1f}%)" for dept, n, share in rows)


def top_artist_table(case_df, slug):
    group_cols = ["Artist"]
    table = (
        case_df.assign(Artist=case_df["Artist"].fillna("Unknown artist").astype(str))
        .groupby(group_cols, dropna=False)
        .agg(
            n_works=("Artist", "size"),
            pct_female=("Gender_Grouped", lambda s: pct_female(case_df.loc[s.index])),
            pct_nonwest=("GeographicOrigin", lambda s: pct_nonwest(case_df.loc[s.index])),
            median_birth_year=("ArtistBirthYear", "median"),
        )
        .reset_index()
        .sort_values(["n_works", "Artist"], ascending=[False, True])
        .head(20)
    )
    table["pct_female"] = table["pct_female"].map(lambda x: fmt_float(x))
    table["pct_nonwest"] = table["pct_nonwest"].map(lambda x: fmt_float(x))
    table["median_birth_year"] = table["median_birth_year"].map(lambda x: fmt_float(x, 1))
    path = get_output_path(f"donor_{slug}_top_artists.csv")
    table.to_csv(path, index=False)
    return table, path


def cumulative_table(case_df, slug):
    years = case_df["YearAcquired"].dropna().astype(int)
    if years.empty:
        out = pd.DataFrame(columns=["donor", "year", "n_gifts", "cumulative_gifts"])
    else:
        counts = years.value_counts().sort_index()
        full = counts.reindex(range(int(years.min()), int(years.max()) + 1), fill_value=0)
        out = pd.DataFrame(
            {
                "donor": case_df["CaseDonor"].iloc[0],
                "year": full.index.astype(int),
                "n_gifts": full.values.astype(int),
                "cumulative_gifts": full.cumsum().values.astype(int),
            }
        )
    path = get_output_path(f"cumulative_{slug}.csv")
    out.to_csv(path, index=False)
    return out, path


def subera_rows(case_name, case_df):
    counts = case_df["YearAcquired"].value_counts().sort_index()
    eras = detect_suberas(counts)
    rows = []
    for i, (start, end, _) in enumerate(eras, start=1):
        sub = case_df[case_df["YearAcquired"].between(start, end)]
        demo = demographic_stats(sub)
        rows.append(
            {
                "donor": case_name,
                "subera_id": i,
                "year_start": int(start),
                "year_end": int(end),
                "n_gifts": int(len(sub)),
                "pct_female": fmt_float(demo["pct_female"]),
                "female_ci_lower": fmt_float(demo["female_ci_lower"]),
                "female_ci_upper": fmt_float(demo["female_ci_upper"]),
                "pct_nonwest": fmt_float(demo["pct_nonwest"]),
                "nonwest_ci_lower": fmt_float(demo["nonwest_ci_lower"]),
                "nonwest_ci_upper": fmt_float(demo["nonwest_ci_upper"]),
            }
        )
    breakpoints = sorted({bp for _, _, bps in eras for bp in bps})
    return rows, breakpoints


def post_donor_drift(base_df, case_df, top_dept):
    years = case_df["YearAcquired"].dropna().astype(int)
    if years.empty:
        return {}
    end_year = int(years.max())
    if end_year + 5 > 2024:
        return {
            "post_drift_applicable": False,
            "post_drift_delta_female_pp": np.nan,
            "post_drift_delta_nonwest_pp": np.nan,
            "post_drift_pre_n": np.nan,
            "post_drift_post_n": np.nan,
        }
    dept_df = base_df[base_df["Department"] == top_dept]
    pre = dept_df[dept_df["YearAcquired"].between(end_year - 5, end_year - 1)]
    post = dept_df[dept_df["YearAcquired"].between(end_year + 1, end_year + 5)]
    female_pre = pct_female(pre)
    female_post = pct_female(post)
    nw_pre = pct_nonwest(pre)
    nw_post = pct_nonwest(post)
    return {
        "post_drift_applicable": True,
        "post_drift_pre_start": end_year - 5,
        "post_drift_pre_end": end_year - 1,
        "post_drift_post_start": end_year + 1,
        "post_drift_post_end": end_year + 5,
        "post_drift_pre_n": int(len(pre)),
        "post_drift_post_n": int(len(post)),
        "post_drift_female_pre5": fmt_float(female_pre),
        "post_drift_female_post5": fmt_float(female_post),
        "post_drift_delta_female_pp": fmt_float(female_post - female_pre)
        if pd.notna(female_post) and pd.notna(female_pre)
        else np.nan,
        "post_drift_nonwest_pre5": fmt_float(nw_pre),
        "post_drift_nonwest_post5": fmt_float(nw_post),
        "post_drift_delta_nonwest_pp": fmt_float(nw_post - nw_pre)
        if pd.notna(nw_post) and pd.notna(nw_pre)
        else np.nan,
    }


def profile_case(base_df, case_df, case_name):
    slug = CASE_SLUGS[case_name]
    years = case_df["YearAcquired"].dropna().astype(int)
    year_counts = years.value_counts().sort_values(ascending=False)
    dept_rows = dept_counts_and_shares(case_df)
    top_dept, top_dept_n, top_dept_share = max(dept_rows, key=lambda x: x[1])
    shares = [share for _, _, share in dept_rows]
    demo = demographic_stats(case_df)
    sub_rows, breakpoints = subera_rows(case_name, case_df)
    cumulative, cumulative_path = cumulative_table(case_df, slug)
    top_artists, top_artists_path = top_artist_table(case_df, slug)

    remove_mask = base_df["CaseDonor"] == case_name
    inst_cf = counterfactual_stats(base_df, remove_mask)
    topdept_cf = counterfactual_stats(base_df, remove_mask, base_df["Department"] == top_dept)
    drift = post_donor_drift(base_df, case_df, top_dept)

    row = {
        "donor": case_name,
        "donor_short": CASE_SHORT[case_name],
        "slug": slug,
        "n_gifts": int(len(case_df)),
        "active_era_start": int(years.min()) if not years.empty else np.nan,
        "active_era_end": int(years.max()) if not years.empty else np.nan,
        "active_era_median": fmt_float(years.median(), 1) if not years.empty else np.nan,
        "active_era_years": int(years.max() - years.min()) if not years.empty else np.nan,
        "peak_year": int(year_counts.index[0]) if not year_counts.empty else np.nan,
        "peak_year_n": int(year_counts.iloc[0]) if not year_counts.empty else 0,
        "changepoints": "|".join(map(str, breakpoints)),
        "n_suberas": len(sub_rows),
        "top_dept": top_dept,
        "top_dept_n": int(top_dept_n),
        "top_dept_share": fmt_float(top_dept_share, 3),
        "dept_footprint_type": footprint_type(shares),
        "dept_breakdown": pipe_dept_breakdown(dept_rows),
        "classification_top5": top_breakdown(case_df["Classification"], n=5),
        "medium_top5": top_breakdown(case_df["Medium"], n=5),
        "n_unique_artists": int(len(donor_artist_set(case_df))),
        "top_artist": top_artists.iloc[0]["Artist"] if not top_artists.empty else np.nan,
        "top_artist_n": int(top_artists.iloc[0]["n_works"]) if not top_artists.empty else 0,
        "cumulative_csv": cumulative_path,
        "top_artists_csv": top_artists_path,
    }
    for key, value in demo.items():
        row[key] = fmt_float(value) if "n_valid" not in key else int(value)
    for prefix, values in [("inst", inst_cf), ("topdept", topdept_cf)]:
        for key, value in values.items():
            row[f"{prefix}_{key}"] = fmt_float(value) if "n" not in key else int(value)
    row.update(drift)
    return row, sub_rows


def log_sentence(row):
    return (
        f"{row['donor']}: N={int(row['n_gifts'])}, {int(row['active_era_start'])}-"
        f"{int(row['active_era_end'])}, {row['dept_footprint_type']} footprint led by "
        f"{row['top_dept']} ({100 * row['top_dept_share']:.1f}%). Female={row['pct_female']:.1f}% "
        f"[{row['female_ci_lower']:.1f}, {row['female_ci_upper']:.1f}], non-Western={row['pct_nonwest']:.1f}% "
        f"[{row['nonwest_ci_lower']:.1f}, {row['nonwest_ci_upper']:.1f}]; removing this case shifts "
        f"institutional female share by {row['inst_delta_female_pp']:+.2f} pp and non-Western share by "
        f"{row['inst_delta_nonwest_pp']:+.2f} pp."
    )


def main():
    ensure_output_dir()
    base_df = assign_case(load_moma_donor_gifts())
    rows = []
    subera_all = []
    missing = []

    for case_name in CASE_ORDER:
        case_df = base_df[base_df["CaseDonor"] == case_name].copy()
        if case_df.empty:
            missing.append(case_name)
            continue
        row, sub_rows = profile_case(base_df, case_df, case_name)
        rows.append(row)
        subera_all.extend(sub_rows)

    if missing:
        raise RuntimeError(f"No records found for case donors: {missing}")

    profile = pd.DataFrame(rows)
    profile.to_csv(get_output_path("donor_biography_profile.csv"), index=False)
    suberas = pd.DataFrame(subera_all)
    suberas.to_csv(get_output_path("donor_subera_demographics.csv"), index=False)

    rockefeller_matches = matched_case_strings(base_df, "Rockefeller lineage")
    rockefeller_audit = "; ".join(f"{name}={int(count)}" for name, count in rockefeller_matches.items())

    bullets = [
        f"Saved profile CSV: {get_output_path('donor_biography_profile.csv')}",
        f"Saved sub-era CSV: {get_output_path('donor_subera_demographics.csv')} ({len(suberas):,} rows)",
        "Rockefeller lineage matched donor strings: " + rockefeller_audit,
    ]
    bullets.extend(log_sentence(row) for _, row in profile.iterrows())
    append_log("Experiment 16a -- Six-donor biography analytics", bullets)
    print(f"Completed Experiment 16a | donors={len(profile)} suberas={len(suberas)}")


if __name__ == "__main__":
    main()

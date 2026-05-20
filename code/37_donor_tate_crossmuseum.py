# -*- coding: utf-8 -*-
"""
Experiment 16d: Tate cross-museum check for the six MoMA donor cases.
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_output_path
from donor_bio_utils import (
    CASE_ORDER,
    TATE_GIFT_TYPES,
    demographic_stats,
    ensure_output_dir,
    fmt_float,
    load_tate_records,
)
from experiment_utils import append_log


PATTERNS = {
    "Kleiner": r"\bkleiner\b",
    "Rockefeller lineage": r"\brockefeller\b",
    "Jean Pigozzi": r"\bpigozzi\b|jean\s+pigozzi",
    "Peter J. Cohen": r"peter\s+j\.?\s+cohen|\bpeter\s+cohen\b",
    "Judith Rothschild Foundation": r"judith\s+rothschild|rothschild\s+foundation",
    "Agnes Gund": r"agnes\s+gund|\bgund\b",
}


def matched_string_summary(sub):
    if sub.empty:
        return ""
    labels = sub["Donor"].copy()
    labels = labels.where(labels.notna(), sub["creditLine"])
    labels = labels.fillna("").astype(str)
    vc = labels[labels.str.strip() != ""].value_counts().head(5)
    return "|".join(f"{name}:{int(count)}" for name, count in vc.items())


def row_for_match(tate_gifts, donor):
    pattern = PATTERNS[donor]
    credit = tate_gifts["creditLine"].fillna("").astype(str)
    parsed = tate_gifts["Donor"].fillna("").astype(str)
    mask = credit.str.contains(pattern, case=False, regex=True, na=False) | parsed.str.contains(
        pattern, case=False, regex=True, na=False
    )
    sub = tate_gifts[mask].copy()
    if sub.empty:
        return {
            "donor": donor,
            "tate_matched_string": "",
            "tate_n_gifts": 0,
            "tate_pct_female": np.nan,
            "tate_pct_nonwest": np.nan,
            "tate_active_era_start": np.nan,
            "tate_active_era_end": np.nan,
        }
    demo = demographic_stats(sub)
    years = pd.to_numeric(sub["YearAcquired"], errors="coerce").dropna().astype(int)
    return {
        "donor": donor,
        "tate_matched_string": matched_string_summary(sub),
        "tate_n_gifts": int(len(sub)),
        "tate_pct_female": fmt_float(demo["pct_female"]),
        "tate_pct_nonwest": fmt_float(demo["pct_nonwest"]),
        "tate_active_era_start": int(years.min()) if not years.empty else np.nan,
        "tate_active_era_end": int(years.max()) if not years.empty else np.nan,
    }


def main():
    ensure_output_dir()
    tate = load_tate_records()
    tate["YearAcquired"] = pd.to_numeric(tate["YearAcquired"], errors="coerce")
    tate_gifts = tate[tate["AcquisitionType"].isin(TATE_GIFT_TYPES) & tate["YearAcquired"].notna()].copy()

    rows = [row_for_match(tate_gifts, donor) for donor in CASE_ORDER]
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("six_donor_tate_overlap.csv"), index=False)

    found = out[out["tate_n_gifts"] > 0]
    if found.empty:
        found_text = "No Tate overlaps found for the six MoMA donor patterns."
    else:
        found_text = "; ".join(
            f"{row['donor']}={int(row['tate_n_gifts'])} gifts ({row['tate_active_era_start']}-{row['tate_active_era_end']})"
            for _, row in found.iterrows()
        )
    append_log(
        "Experiment 16d -- Tate cross-museum donor check",
        [
            f"Saved Tate overlap CSV: {get_output_path('six_donor_tate_overlap.csv')}",
            found_text,
        ],
    )
    print(f"Completed Experiment 16d | overlaps={len(found)}/6")


if __name__ == "__main__":
    main()

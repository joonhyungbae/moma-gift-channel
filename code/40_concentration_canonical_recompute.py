# -*- coding: utf-8 -*-
"""
Phase Q: recompute donor concentration on the canonical 49,740-record donor
gift sample used by the six-donor joint counterfactual.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_output_path
from donor_bio_utils import load_moma_donor_gifts
from experiment_utils import append_log, load_base_data
from utils import extract_donor_from_gift, gini_coefficient


def pct(value):
    return 100.0 * value


def fmt_pct(value, digits=2):
    return f"{value:.{digits}f}%"


def legacy_gift_only_sample():
    """Reconstruct the superseded Experiment 11 concentration base."""
    oldbase = load_base_data(require_birthyear=False)
    gifts = oldbase[oldbase["AcquisitionType"] == "Gift"].copy()
    gifts["Donor"] = gifts["CreditLine"].apply(extract_donor_from_gift)
    gifts = gifts[gifts["Donor"].notna()].copy()
    return gifts


def build_rows(canon, vc, legacy):
    total = len(canon)
    donors = int(canon["Donor"].nunique())
    top1_round_n = max(1, int(round(donors * 0.01)))
    top1_ceil_n = max(1, int(np.ceil(donors * 0.01)))
    top45_n = 45
    largest_name = vc.index[0]
    largest_n = int(vc.iloc[0])
    rows = [
        {
            "metric": "total_gifts",
            "value": str(total),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "total_donors",
            "value": str(donors),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "largest_donor_name",
            "value": largest_name,
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "largest_donor_n",
            "value": str(largest_n),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "largest_donor_share",
            "value": fmt_pct(pct(largest_n / total), 2),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "top1pct_n_donors",
            "value": str(top1_round_n),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "top1pct_raw_1pct_of_donors",
            "value": f"{donors * 0.01:.2f}",
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "top1pct_share",
            "value": fmt_pct(pct(vc.head(top1_round_n).sum() / total), 2),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "top1pct_n_donors_ceiling",
            "value": str(top1_ceil_n),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "top1pct_share_ceiling",
            "value": fmt_pct(pct(vc.head(top1_ceil_n).sum() / total), 2),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "top45_share",
            "value": fmt_pct(pct(vc.head(top45_n).sum() / total), 2),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "top20_share",
            "value": fmt_pct(pct(vc.head(20).sum() / total), 2),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "top20_n_gifts",
            "value": str(int(vc.head(20).sum())),
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "gini",
            "value": f"{gini_coefficient(vc.values):.4f}",
            "denominator_N": total,
            "donor_base": donors,
        },
        {
            "metric": "legacy_total_gifts",
            "value": str(len(legacy)),
            "denominator_N": len(legacy),
            "donor_base": int(legacy["Donor"].nunique()),
        },
        {
            "metric": "legacy_total_donors",
            "value": str(int(legacy["Donor"].nunique())),
            "denominator_N": len(legacy),
            "donor_base": int(legacy["Donor"].nunique()),
        },
        {
            "metric": "canonical_minus_legacy_n",
            "value": str(total - len(legacy)),
            "denominator_N": total,
            "donor_base": donors,
        },
    ]
    return pd.DataFrame(rows)


def reconciliation_text(canon, legacy):
    acq = canon["AcquisitionType"].value_counts()
    canonical_gift = int(acq.get("Gift", 0))
    nongift = len(canon) - canonical_gift
    legacy_n = len(legacy)
    gift_delta = canonical_gift - legacy_n
    return (
        "The 48,792 figure is a stale/superseded Gift-only run from code/30, not a single-donor-only subset: "
        f"legacy Gift-only N={legacy_n:,}; canonical Gift N={canonical_gift:,} ({gift_delta:+,} vs legacy after stricter normalization/filtering); "
        f"canonical adds {nongift:,} donor-identified non-Gift records "
        f"({int(acq.get('Partial Gift/Purchase', 0)):,} Partial Gift/Purchase, {int(acq.get('Bequest', 0)):,} Bequest, "
        f"{int(acq.get('Artist Gift', 0)):,} Artist Gift), yielding the net +{len(canon) - legacy_n:,} records."
    )


def main():
    canon = load_moma_donor_gifts()
    legacy = legacy_gift_only_sample()
    vc = canon["Donor"].value_counts()
    rows = build_rows(canon, vc, legacy)
    rows.to_csv(get_output_path("concentration_canonical_recompute.csv"), index=False)

    total = len(canon)
    donors = canon["Donor"].nunique()
    top1_n = int(round(donors * 0.01))
    largest_share = pct(vc.iloc[0] / total)
    top1_share = pct(vc.head(top1_n).sum() / total)
    top45_share = pct(vc.head(45).sum() / total)
    top20_share = pct(vc.head(20).sum() / total)
    recon = reconciliation_text(canon, legacy)

    append_log(
        "Phase Q -- Canonical donor concentration recompute",
        [
            f"Saved CSV: {get_output_path('concentration_canonical_recompute.csv')}",
            f"Canonical donor-identified gift sample: N={total:,}, donors={donors:,}.",
            f"Largest donor: {vc.index[0]} with {int(vc.iloc[0]):,} records ({largest_share:.2f}%).",
            f"Top 1% by rounded count: {top1_n} donors, {top1_share:.2f}% of records; top 45 donors hold {top45_share:.2f}%.",
            f"Top 20 donors hold {top20_share:.2f}% of records.",
            recon,
        ],
    )

    print(f"Largest single donor share: {largest_share:.2f}% ({vc.index[0]}, {int(vc.iloc[0]):,}/{total:,}).")
    print(f"Top 1% ({top1_n} donors) share: {top1_share:.2f}% ; top 45 share: {top45_share:.2f}%.")
    print(f"Top 20 donor share: {top20_share:.2f}% ({int(vc.head(20).sum()):,}/{total:,}).")
    print(f"Correct sample: N={total:,} donor-identified gift records; {donors:,} normalized donors.")
    print(recon)


if __name__ == "__main__":
    main()

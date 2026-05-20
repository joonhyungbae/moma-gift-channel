# -*- coding: utf-8 -*-
"""
Phase L: re-run the symmetric Tate boundary check after removing Tate
channel-label aggregates from the Tate-side donor pool.
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_output_path
from donor_bio_utils import CASE_ORDER, assign_case, demographic_stats, ensure_output_dir, fmt_float, load_moma_donor_gifts
from experiment_utils import append_log


def load_phasej_module():
    path = Path(__file__).with_name("38_phase_j_verifications.py")
    spec = importlib.util.spec_from_file_location("phasej", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASEJ = load_phasej_module()

ORIGINAL_TATE_EXTREMES = [
    "Turner Bequest",
    "Trustees of the Chantrey Bequest",
    "Alicia and León Ferrari",
    "Mrs Joan Highmore Blackhall and Dr R.B. McConnell",
    "American Fund for the Tate Gallery",
    "Tate Members",
]

CHANNEL_EXACT = {
    "Turner Bequest",
    "Trustees of the Chantrey Bequest",
    "American Fund for the Tate Gallery",
    "Tate Members",
    "Tate Patrons",
    "Tate Gallery Publications",
    "Friends of the Tate Gallery",
    "Tate International Council",
    "Institute of Contemporary Prints",
    "Art Fund",
    "National Art Collections Fund",
    "HM Government in lieu of inheritance tax and allocated to Tate",
    "HM Government in lieu of tax and allocated to the Tate Gallery",
    "HM Government in lieu of tax and allocated to Tate",
    "Ministry of Information",
}

CHANNEL_PATTERNS = [
    (re.compile(r"^trustees\s+of\s+.*\bbequest\b", re.IGNORECASE), "trustees administering a bequest instrument"),
    (re.compile(r"\bturner\s+bequest\b", re.IGNORECASE), "state-historical bequest instrument, not an individual donor string"),
    (re.compile(r"^[A-Z][A-Za-z .'-]+\s+Bequest$", re.IGNORECASE), "standalone bequest label without a named contemporary donor"),
    (re.compile(r"\bmembers\b", re.IGNORECASE), "membership-body label"),
    (re.compile(r"\bpatrons\b", re.IGNORECASE), "patron/membership acquisition body"),
    (re.compile(r"^friends\s+of\b", re.IGNORECASE), "friends/membership acquisition body"),
    (re.compile(r"\bfund\s+for\s+the\s+tate\b", re.IGNORECASE), "fundraising vehicle for Tate"),
    (re.compile(r"^american\s+fund\s+for\s+the\s+tate", re.IGNORECASE), "US fundraising vehicle for Tate"),
    (re.compile(r"^hm\s*government\b", re.IGNORECASE), "state allocation in lieu of tax"),
    (re.compile(r"^ministry\s+of\b", re.IGNORECASE), "government channel"),
    (re.compile(r"^tate\s+gallery\s+publications\b", re.IGNORECASE), "internal Tate publication/acquisition channel"),
    (re.compile(r"^tate\s+international\s+council\b", re.IGNORECASE), "council acquisition channel"),
    (re.compile(r"^institute\s+of\s+contemporary\s+prints\b", re.IGNORECASE), "print-distribution channel label without named donor"),
    (re.compile(r"^art\s+fund(?:\b|\s|\()", re.IGNORECASE), "art-fund channel label without named donor"),
    (re.compile(r"^patrons\s+of\s+new\s+art\b", re.IGNORECASE), "patron acquisition channel"),
]

INSTITUTION_RE = re.compile(
    r"Foundation|Ltd|Limited|Studio|Studios|Galleries|Gallery|Society|Press|"
    r"Committee|Council|Corporation|Company|Collection|Estate of|Inc\.",
    re.IGNORECASE,
)


def latex_escape(value):
    text = "" if pd.isna(value) else str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }.items():
        text = text.replace(old, new)
    return text


def channel_reason(name):
    if pd.isna(name):
        return None
    text = str(name).strip()
    if text in CHANNEL_EXACT:
        exact_reasons = {
            "Turner Bequest": "state-historical testamentary bequest accepted by the nation; no individual biography unit in the parsed label",
            "Trustees of the Chantrey Bequest": "trustees administering a bequest instrument",
            "American Fund for the Tate Gallery": "US fundraising vehicle for Tate, not a named donor biography",
            "Tate Members": "membership-body acquisition label",
            "Tate Gallery Publications": "internal Tate publication/acquisition channel",
            "Tate Patrons": "patron/membership acquisition body",
            "Friends of the Tate Gallery": "friends/membership acquisition body",
            "Tate International Council": "council acquisition channel",
            "Institute of Contemporary Prints": "print-distribution channel label without named donor",
            "Art Fund": "art-fund channel label without named donor",
            "National Art Collections Fund": "art-fund channel label without named donor",
            "Ministry of Information": "government channel",
        }
        return exact_reasons.get(text, "channel aggregate label")
    for pattern, reason in CHANNEL_PATTERNS:
        if pattern.search(text):
            # Preserve named people who gave through a channel, such as
            # "A. Acland Allen through the Art Fund" or "Rose and Chris Prater
            # through the Institute of Contemporary Prints"; these remain donor
            # strings with a named actor before the channel phrase.
            if re.search(r"\bthrough\s+the\s+(?:art fund|institute of contemporary prints)\b", text, re.IGNORECASE):
                prefix = re.split(r"\bthrough\b", text, flags=re.IGNORECASE)[0].strip()
                if len(prefix.split()) >= 2 and not re.search(r"^(art fund|institute)", prefix, re.IGNORECASE):
                    continue
            return reason
    return None


def is_channel_aggregate(name):
    return channel_reason(name) is not None


def classify_original_six():
    rows = []
    manual = {
        "Turner Bequest": (
            "channel aggregate",
            "The Tate credit line is 'Accepted by the nation as part of the Turner Bequest'; the parsed label is a state-historical bequest channel rather than a living donor biography.",
        ),
        "Trustees of the Chantrey Bequest": (
            "channel aggregate",
            "The label names trustees administering a bequest instrument, not the donor as an individual or foundation.",
        ),
        "Alicia and León Ferrari": (
            "individual donor",
            "Named individual/couple credit line: 'Presented by Alicia and León Ferrari'.",
        ),
        "Mrs Joan Highmore Blackhall and Dr R.B. McConnell": (
            "individual donor",
            "Named individual/couple credit line with personal titles.",
        ),
        "American Fund for the Tate Gallery": (
            "channel aggregate",
            "Fundraising vehicle for Tate; credit lines often add courtesy/private collector or acquisitions-committee details.",
        ),
        "Tate Members": (
            "channel aggregate",
            "Membership-body acquisition label, not a donor biography comparable to MoMA's six cases.",
        ),
    }
    for donor in ORIGINAL_TATE_EXTREMES:
        classification, rationale = manual[donor]
        rows.append({"donor_string": donor, "classification": classification, "rationale": rationale})
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("tate_channel_label_classification.csv"), index=False)
    return out


def donor_profile_for_selection(tate):
    dept_universe = sorted(tate["Department"].fillna("Unknown").astype(str).unique())
    baseline = demographic_stats(tate)
    rows = []
    for donor, group in tate.groupby("Donor"):
        n = len(group)
        years = group["YearAcquired"].dropna().astype(int)
        dept_counts = group["Department"].fillna("Unknown").astype(str).value_counts()
        shares = np.array([dept_counts.get(dept, 0) / n for dept in dept_universe], dtype=float)
        h = float(np.sum(shares**2))
        demo = demographic_stats(group)
        female_gap = demo["pct_female"] - baseline["pct_female"] if pd.notna(demo["pct_female"]) else np.nan
        nonwest_gap = demo["pct_nonwest"] - baseline["pct_nonwest"] if pd.notna(demo["pct_nonwest"]) else np.nan
        rows.append(
            {
                "donor_string": donor,
                "n_gifts": int(n),
                "era_start": int(years.min()) if not years.empty else np.nan,
                "era_end": int(years.max()) if not years.empty else np.nan,
                "era_span": int(years.max() - years.min()) if not years.empty else np.nan,
                "top_dept": dept_counts.index[0] if not dept_counts.empty else np.nan,
                "pct_female": demo["pct_female"],
                "pct_nonwest": demo["pct_nonwest"],
                "female_codable_n": int(demo["female_n_valid"]),
                "nonwest_codable_n": int(demo["nonwest_n_valid"]),
                "codable_n": int(max(demo["female_n_valid"], demo["nonwest_n_valid"])),
                "female_gap": female_gap,
                "nonwest_gap": nonwest_gap,
                "is_institutional": bool(INSTITUTION_RE.search(str(donor))),
                "dept_herfindahl": h,
                "dept_one_minus_herfindahl": 1 - h,
            }
        )
    return pd.DataFrame(rows), baseline


def select_tate_extremes_v2(profile):
    eligible = profile[profile["n_gifts"] >= 100].copy()
    selected = []
    used = set()

    def pick(axis, candidates, sort_cols, ascending):
        pool = candidates[~candidates["donor_string"].isin(used)].copy()
        if pool.empty:
            pool = candidates.copy()
        row = pool.sort_values(sort_cols, ascending=ascending, kind="mergesort").iloc[0].to_dict()
        row["axis"] = axis
        used.add(row["donor_string"])
        selected.append(row)

    pick("pure_volume", eligible, ["n_gifts", "donor_string"], [False, True])
    pick("active_era_length", eligible, ["era_span", "n_gifts", "donor_string"], [False, False, True])
    geo = eligible[eligible["nonwest_codable_n"] > 0]
    pick("geographic_origin_outlier", geo, ["nonwest_gap", "n_gifts", "donor_string"], [False, False, True])
    gender = eligible[eligible["female_codable_n"] > 0]
    pick("gender_outlier", gender, ["female_gap", "n_gifts", "donor_string"], [False, False, True])
    institutional = eligible[eligible["is_institutional"]]
    pick("largest_institutional", institutional, ["n_gifts", "donor_string"], [False, True])
    pick(
        "departmental_spread",
        eligible,
        ["dept_one_minus_herfindahl", "n_gifts", "donor_string"],
        [False, False, True],
    )
    cols = [
        "axis",
        "donor_string",
        "n_gifts",
        "era_start",
        "era_end",
        "top_dept",
        "pct_female",
        "pct_nonwest",
        "codable_n",
    ]
    out = pd.DataFrame(selected)[cols].copy()
    out["pct_female"] = out["pct_female"].map(lambda x: fmt_float(x))
    out["pct_nonwest"] = out["pct_nonwest"].map(lambda x: fmt_float(x))
    out.to_csv(get_output_path("tate_six_extreme_donors_v2.csv"), index=False)
    return out


def recurrence_against_moma(tate_extremes, moma, out_name):
    moma_counts = moma["Donor"].value_counts()
    moma_names = moma_counts.index.to_series().astype(str)
    rows = []
    for _, row in tate_extremes.iterrows():
        donor = str(row["donor_string"])
        exact = donor in moma_counts.index
        tokens = PHASEJ.diagnostic_tokens(donor)
        if tokens:
            pattern = "|".join(re.escape(tok) for tok in tokens)
            hits = moma_names[moma_names.str.contains(pattern, case=False, regex=True, na=False)]
        else:
            hits = pd.Series(dtype=str)
        rows.append(
            {
                "tate_donor": donor,
                "n_at_tate": int(row["n_gifts"]),
                "exact_match_at_moma": bool(exact),
                "exact_match_n_gifts": int(moma_counts.get(donor, 0)),
                "surname_substring_hits": int(len(hits)),
                "hit_strings": "|".join(f"{name} ({int(moma_counts.get(name, 0))})" for name in hits.head(10)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path(out_name), index=False)
    return out


def moma_to_filtered_tate_recurrence(filtered_tate):
    rows = []
    for donor in CASE_ORDER:
        pattern = PHASEJ.MOMA_TO_TATE_PATTERNS[donor]
        credit = filtered_tate["creditLine"].fillna("").astype(str)
        parsed = filtered_tate["Donor"].fillna("").astype(str)
        mask = credit.str.contains(pattern, case=False, regex=True, na=False) | parsed.str.contains(
            pattern, case=False, regex=True, na=False
        )
        sub = filtered_tate[mask]
        labels = sub["Donor"].fillna(sub["creditLine"]).fillna("").astype(str)
        vc = labels[labels.str.strip() != ""].value_counts().head(5)
        rows.append(
            {
                "moma_donor": donor,
                "tate_n_gifts_filtered": int(len(sub)),
                "tate_matched_string": "|".join(f"{name}:{int(count)}" for name, count in vc.items()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("moma_to_tate_channel_filtered_recurrence_v2.csv"), index=False)
    return out


def write_boundary_table_v2(original_class, tate_extremes, recurrence, moma_to_tate):
    channel_n = int((original_class["classification"] == "channel aggregate").sum())
    tate_exact = int(recurrence["exact_match_at_moma"].sum())
    tate_exact_gifts = int(recurrence["exact_match_n_gifts"].sum())
    tate_diag = int((recurrence["surname_substring_hits"] > 0).sum())
    mt_exact = int((moma_to_tate["tate_n_gifts_filtered"] > 0).sum())
    mt_gifts = int(moma_to_tate["tate_n_gifts_filtered"].sum())
    mt_word = "gift" if mt_gifts == 1 else "gifts"
    tate_word = "gift" if tate_exact_gifts == 1 else "gifts"
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\tbl{Symmetric cross-museum boundary check after Tate channel-label filtering}",
        r"{\begin{tabular}{lccc p{0.35\textwidth}}",
        r"\toprule",
        r"Direction & Selected cases & Exact recurrence & Diagnostic substring hits & Interpretation \\",
        r"\midrule",
        (
            f"MoMA extremes $\\rightarrow$ Tate & 6 & {mt_exact}/6 ({mt_gifts} {mt_word}) & -- & "
            f"{latex_escape('Rechecked against the channel-filtered Tate donor pool.')} \\\\"
        ),
        (
            f"Tate extremes v2 $\\rightarrow$ MoMA & 6 & {tate_exact}/6 ({tate_exact_gifts} {tate_word}) & "
            f"{tate_diag}/6 & {latex_escape(f'{channel_n}/6 original Tate extremes were channel aggregates and were excluded before reselection.')} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}}",
        r"\label{tab:symmetric_boundary_check_v2}",
        r"\end{table}",
    ]
    with open(get_output_path("symmetric_boundary_check_v2.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_summary_v2(original_class, tate_extremes, recurrence, moma_to_tate):
    channel = original_class[original_class["classification"] == "channel aggregate"]["donor_string"].tolist()
    tate_exact = int(recurrence["exact_match_at_moma"].sum())
    tate_gifts = int(recurrence["exact_match_n_gifts"].sum())
    mt_exact = int((moma_to_tate["tate_n_gifts_filtered"] > 0).sum())
    mt_gifts = int(moma_to_tate["tate_n_gifts_filtered"].sum())
    tate_word = "gift" if tate_gifts == 1 else "gifts"
    mt_word = "gift" if mt_gifts == 1 else "gifts"
    selected = "; ".join(tate_extremes["donor_string"].tolist())
    sentences = [
        f"Four of the original six Tate-extreme labels are channel aggregates rather than donor biographies: {', '.join(channel)}.",
        f"After filtering those channel labels and re-running the six-axis Tate selection, the selected donors are {selected}; exact recurrence at MoMA is {tate_exact}/6 ({tate_gifts} MoMA {tate_word}), while the MoMA-to-filtered-Tate direction remains {mt_exact}/6 ({mt_gifts} Tate {mt_word}).",
        "The institution-specific reading still holds, but the Phase J table should be replaced by the channel-filtered version because the original Tate-side selection mixed donor biographies with acquisition channels.",
    ]
    with open(get_output_path("symmetric_boundary_check_v2_summary.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(sentences) + "\n")


def main():
    ensure_output_dir()
    original_class = classify_original_six()
    moma = assign_case(load_moma_donor_gifts())
    tate = PHASEJ.load_tate_phase_j_gifts()
    tate["ChannelAggregate"] = tate["Donor"].apply(is_channel_aggregate)
    filtered = tate[~tate["ChannelAggregate"]].copy()
    profile, baseline = donor_profile_for_selection(filtered)
    tate_extremes = select_tate_extremes_v2(profile)
    recurrence = recurrence_against_moma(tate_extremes, moma, "tate_to_moma_recurrence_v2.csv")
    moma_to_tate = moma_to_filtered_tate_recurrence(filtered)
    write_boundary_table_v2(original_class, tate_extremes, recurrence, moma_to_tate)
    write_summary_v2(original_class, tate_extremes, recurrence, moma_to_tate)

    channel_n = int(tate["ChannelAggregate"].sum())
    append_log(
        "Phase L -- Tate channel-label re-verification",
        [
            f"Saved classification CSV: {get_output_path('tate_channel_label_classification.csv')}",
            f"Saved filtered Tate extremes: {get_output_path('tate_six_extreme_donors_v2.csv')}",
            f"Saved recurrence CSV: {get_output_path('tate_to_moma_recurrence_v2.csv')}",
            f"Saved updated symmetric table: {get_output_path('symmetric_boundary_check_v2.tex')}",
            f"Original Tate extremes classified as channel aggregates: {(original_class['classification'] == 'channel aggregate').sum()}/6.",
            f"Filtered Tate donor pool: {len(filtered):,} records, {filtered['Donor'].nunique():,} donors; removed {channel_n:,} channel-labeled records.",
            f"Tate v2 -> MoMA exact recurrence: {int(recurrence['exact_match_at_moma'].sum())}/6.",
            f"MoMA -> filtered Tate recurrence: {int((moma_to_tate['tate_n_gifts_filtered'] > 0).sum())}/6.",
        ],
    )

    print(
        "Phase L complete | original channels=%s/6 | filtered Tate records=%s | Tate->MoMA exact=%s/6 | MoMA->Tate exact=%s/6"
        % (
            int((original_class["classification"] == "channel aggregate").sum()),
            f"{len(filtered):,}",
            int(recurrence["exact_match_at_moma"].sum()),
            int((moma_to_tate["tate_n_gifts_filtered"] > 0).sum()),
        )
    )


if __name__ == "__main__":
    main()

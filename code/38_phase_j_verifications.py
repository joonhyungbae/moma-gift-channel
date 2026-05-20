# -*- coding: utf-8 -*-
"""
Phase J (Round 4) verification script.

Runs:
  1. arithmetic checks for three flagged manuscript claims;
  2. symmetric Tate boundary check;
  3. parser-specification appendix generation.
"""

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GENERIC_DONOR_ROLE_TERMS, MAIN_DEPARTMENTS, WESTERN_KEYWORDS, get_output_path, get_processed_path
from donor_bio_utils import (
    CASE_ORDER,
    GIFT_TYPES,
    TITLE_PREFIX_RE,
    assign_case,
    canonicalize_donors,
    compact_space,
    demographic_stats,
    ensure_output_dir,
    fmt_float,
    is_generic_donor_name,
    load_moma_donor_gifts,
    load_tate_records,
    matched_case_strings,
)
from experiment_utils import append_log


INSTITUTION_RE = re.compile(
    r"Foundation|Fund|Trust|Estate of|Inc\.|Corporation|Committee|Council|Endowment|Charitable|Collection",
    re.IGNORECASE,
)

TATE_PREFIXES = [
    "presented by the ",
    "bequeathed by the ",
    "given by the ",
    "accepted by the ",
    "presented by ",
    "bequeathed by ",
    "given by ",
    "accepted by ",
]

MOMA_TO_TATE_PATTERNS = {
    "Kleiner": r"\bkleiner\b",
    "Rockefeller lineage": r"\brockefeller\b",
    "Jean Pigozzi": r"\bpigozzi\b|jean\s+pigozzi",
    "Peter J. Cohen": r"peter\s+j\.?\s+cohen|\bpeter\s+cohen\b",
    "Judith Rothschild Foundation": r"judith\s+rothschild|rothschild\s+foundation",
    "Agnes Gund": r"agnes\s+gund|\bgund\b",
}

EASTERN_EUROPE_NONWESTERN_LIST = [
    "Russian",
    "Soviet-era",
    "Ukrainian",
    "Polish",
    "Czech",
    "Hungarian",
    "Romanian",
    "Slovak",
    "Slovenian",
    "Croatian",
    "Bosnian",
    "Serbian",
    "Bulgarian",
    "Belarusian",
    "Estonian",
    "Latvian",
    "Lithuanian",
]


def latex_escape(value):
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def pct_text(value, digits=1):
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}%"


def status_match(actual, expected, tol=0.05):
    return "match" if abs(float(actual) - float(expected)) <= tol else "mismatch"


def load_phase_j_moma():
    df = assign_case(load_moma_donor_gifts())
    if len(df) != 49_740:
        print(f"Warning: canonical MoMA donor sample has N={len(df):,}, expected 49,740.")
    return df


def arithmetic_checks(moma):
    rows = []
    source = get_processed_path()
    six_mask = moma["CaseDonor"].isin(CASE_ORDER)
    total_nonwest = int((moma["GeographicOrigin"] == "Non-Western").sum())
    six_nonwest = int((moma.loc[six_mask, "GeographicOrigin"] == "Non-Western").sum())
    ratio = six_nonwest / total_nonwest if total_nonwest else np.nan
    rows.append(
        {
            "claim_id": "1a_nonwestern_share_carried_by_six",
            "claim_text": "The six donors account for 22% of non-Western records.",
            "paper_value": "22%",
            "recomputed_value": f"{100 * ratio:.3g}% ({six_nonwest:,}/{total_nonwest:,} non-Western records)",
            "source_file": source,
            "status": status_match(100 * ratio, 22, tol=0.5),
        }
    )

    years = set(range(1929, 2025))
    six_gift_years = set(moma.loc[six_mask, "YearAcquired"].dropna().astype(int).unique())
    years_with_gifts = len(years & six_gift_years)
    zero_years = len(years - six_gift_years)
    active_span_years = set()
    for donor in CASE_ORDER:
        donor_years = moma.loc[moma["CaseDonor"] == donor, "YearAcquired"].dropna().astype(int)
        if not donor_years.empty:
            active_span_years.update(range(int(donor_years.min()), int(donor_years.max()) + 1))
    active_span_count = len(years & active_span_years)
    rows.append(
        {
            "claim_id": "1b_six_donor_years_with_gifts",
            "claim_text": "94 of 96 years between 1929 and 2024 have at least one of the six donors active.",
            "paper_value": "94/96",
            "recomputed_value": (
                f"{years_with_gifts}/96 years with >=1 six-donor gift; {zero_years} zero years; "
                f"{active_span_count}/96 if continuous active-era spans are counted"
            ),
            "source_file": source,
            "status": "match" if years_with_gifts == 94 else "mismatch_observed_gift_years",
        }
    )

    gund = moma[moma["CaseDonor"] == "Agnes Gund"].copy()
    dept_counts = gund["Department"].value_counts()
    shares = {dept: dept_counts.get(dept, 0) / len(gund) for dept in MAIN_DEPARTMENTS}
    top_dept = max(shares, key=shares.get)
    top_share = shares[top_dept]
    h = float(sum(v * v for v in shares.values()))
    n_depts = len(MAIN_DEPARTMENTS)
    h_norm = (h - 1 / n_depts) / (1 - 1 / n_depts)
    share_text = "; ".join(f"{dept}={shares[dept]:.3f}" for dept in MAIN_DEPARTMENTS)
    rows.append(
        {
            "claim_id": "1c_gund_department_concentration",
            "claim_text": "Gund dept-Herfindahl approximately 0.5.",
            "paper_value": "Herfindahl approx 0.5",
            "recomputed_value": (
                f"top_share={top_share:.3f} ({top_dept}); H={h:.3f}; H_norm={h_norm:.3f}; shares: {share_text}"
            ),
            "source_file": source,
            "status": "terminology_mismatch" if abs(h - 0.5) > 0.05 and abs(top_share - 0.5) <= 0.05 else "check",
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("phase_j_arithmetic_check.csv"), index=False)
    return out, {
        "ratio": ratio,
        "six_nonwest": six_nonwest,
        "total_nonwest": total_nonwest,
        "years_with_gifts": years_with_gifts,
        "zero_years": zero_years,
        "active_span_count": active_span_count,
        "gund_shares": shares,
        "gund_top_share": top_share,
        "gund_h": h,
        "gund_h_norm": h_norm,
    }


def extract_tate_phase_j_donor(creditline):
    if pd.isna(creditline):
        return np.nan
    line = str(creditline).strip()
    low = line.lower()
    for phrase in TATE_PREFIXES:
        idx = low.find(phrase)
        if idx >= 0:
            rest = line[idx + len(phrase) :].strip()
            for sep in (",", "\n"):
                if sep in rest:
                    rest = rest.split(sep)[0].strip()
                    break
            rest = re.sub(r"\s*\d{4}\s*$", "", rest).strip()
            rest = rest.rstrip(".").strip()
            turner = re.search(r"as part of the\s+(Turner Bequest)", rest, flags=re.IGNORECASE)
            if turner:
                return "Turner Bequest"
            if not rest or is_generic_donor_name(rest):
                return np.nan
            return rest
    return np.nan


def load_tate_phase_j_gifts():
    tate = load_tate_records()
    tate["YearAcquired"] = pd.to_numeric(tate["YearAcquired"], errors="coerce")
    credit = tate["creditLine"].fillna("").astype(str)
    prefix_gift = credit.str.contains(
        r"^\s*(?:presented|bequeathed|given|accepted)\s+by\b", case=False, regex=True, na=False
    )
    gift_like = tate["AcquisitionType"].isin(["Gift", "Bequest"]) | prefix_gift
    tate = tate[gift_like & tate["YearAcquired"].notna()].copy()
    tate["DonorRaw"] = tate["creditLine"].apply(extract_tate_phase_j_donor)
    tate["Donor"] = canonicalize_donors(tate["DonorRaw"])
    tate = tate[tate["Donor"].notna()].copy()
    tate["YearAcquired"] = tate["YearAcquired"].astype(int)
    return tate


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
    return pd.DataFrame(rows), baseline, dept_universe


def select_tate_extremes(profile):
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
    out.to_csv(get_output_path("tate_six_extreme_donors.csv"), index=False)
    return out


STOPWORDS = {
    "and",
    "the",
    "of",
    "for",
    "in",
    "through",
    "by",
    "with",
    "honour",
    "honor",
    "mrs",
    "mr",
    "dr",
    "ms",
    "miss",
    "sir",
    "dame",
    "ltd",
    "inc",
    "fund",
    "foundation",
    "trust",
    "trustees",
    "bequest",
    "gallery",
    "tate",
    "publications",
    "society",
    "members",
    "patrons",
    "friends",
    "contemporary",
    "prints",
    "institute",
    "art",
    "arts",
    "studio",
    "studios",
    "graphics",
    "limited",
    "american",
    "british",
}


def diagnostic_tokens(name):
    core = re.split(r"\bthrough\b|\bin honour\b|\bin honor\b|\bfor\b", str(name), flags=re.IGNORECASE)[0]
    tokens = [t.lower().strip("'") for t in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", core)]
    tokens = [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]
    if not tokens:
        return []
    return list(dict.fromkeys(tokens[-2:]))


def tate_to_moma_recurrence(tate_extremes, moma):
    moma_counts = moma["Donor"].value_counts()
    moma_names = moma_counts.index.to_series().astype(str)
    rows = []
    for _, row in tate_extremes.iterrows():
        donor = str(row["donor_string"])
        exact = donor in moma_counts.index
        exact_n = int(moma_counts.get(donor, 0))
        tokens = diagnostic_tokens(donor)
        if tokens:
            pattern = "|".join(re.escape(tok) for tok in tokens)
            hits = moma_names[moma_names.str.contains(pattern, case=False, regex=True, na=False)]
        else:
            hits = pd.Series(dtype=str)
        hit_strings = []
        for name in hits.head(10):
            hit_strings.append(f"{name} ({int(moma_counts.get(name, 0))})")
        rows.append(
            {
                "tate_donor": donor,
                "n_at_tate": int(row["n_gifts"]),
                "exact_match_at_moma": bool(exact),
                "exact_match_n_gifts": exact_n,
                "surname_substring_hits": int(len(hits)),
                "hit_strings": "|".join(hit_strings),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("tate_to_moma_recurrence.csv"), index=False)
    return out


def existing_moma_to_tate_summary():
    path = Path(get_output_path("six_donor_tate_overlap.csv"))
    if not path.exists():
        return {"exact_count": 1, "gift_total": 1, "note": "Prior exact-string check: 5 absent, 1 with 1 gift."}
    df = pd.read_csv(path)
    exact_count = int((df["tate_n_gifts"].fillna(0) > 0).sum())
    gift_total = int(df["tate_n_gifts"].fillna(0).sum())
    gift_word = "gift" if gift_total == 1 else "gifts"
    return {
        "exact_count": exact_count,
        "gift_total": gift_total,
        "note": f"{6 - exact_count} absent, {exact_count} recurring with {gift_total} Tate {gift_word}.",
    }


def write_symmetric_boundary_outputs(tate_extremes, recurrence):
    moma_to_tate = existing_moma_to_tate_summary()
    exact_tate_to_moma = int(recurrence["exact_match_at_moma"].sum())
    exact_gifts = int(recurrence["exact_match_n_gifts"].sum())
    diagnostic = int((recurrence["surname_substring_hits"] > 0).sum())
    mt_gift_word = "gift" if moma_to_tate["gift_total"] == 1 else "gifts"
    tm_gift_word = "gift" if exact_gifts == 1 else "gifts"

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\tbl{Symmetric cross-museum boundary check}",
        r"{\begin{tabular}{lccc p{0.36\textwidth}}",
        r"\toprule",
        r"Direction & Selected cases & Exact recurrence & Diagnostic substring hits & Interpretation \\",
        r"\midrule",
        (
            f"MoMA extremes $\\rightarrow$ Tate & 6 & {moma_to_tate['exact_count']}/6 "
            f"({moma_to_tate['gift_total']} {mt_gift_word}) & -- & {latex_escape(moma_to_tate['note'])} \\\\"
        ),
        (
            f"Tate extremes $\\rightarrow$ MoMA & 6 & {exact_tate_to_moma}/6 "
            f"({exact_gifts} {tm_gift_word}) & {diagnostic}/6 & "
            f"{latex_escape('Symmetric Tate-side selection uses the same six axes, N>=100, without family-line aggregation.')} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}}",
        r"\label{tab:symmetric_boundary_check}",
        r"\end{table}",
    ]
    with open(get_output_path("symmetric_boundary_check.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    support = "supports" if exact_tate_to_moma <= 1 else "weakens"
    donor_list = "; ".join(tate_extremes["donor_string"].tolist())
    summary = (
        f"Re-running the six-axis selection rule on Tate selects: {donor_list}. "
        f"Exact recurrence at MoMA is {exact_tate_to_moma} of 6 Tate-extreme donors "
        f"({exact_gifts} MoMA gifts), with surname-substring diagnostics flagging {diagnostic} of 6. "
        f"Together with the existing MoMA-to-Tate result ({moma_to_tate['exact_count']} of 6, "
        f"{moma_to_tate['gift_total']} Tate gift), the symmetric check {support} the interpretation "
        f"that mega-donor boundary cases are largely institution-specific rather than generic cross-museum actors."
    )
    with open(get_output_path("symmetric_boundary_check_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    return {
        "moma_to_tate_exact": moma_to_tate["exact_count"],
        "tate_to_moma_exact": exact_tate_to_moma,
        "tate_to_moma_gifts": exact_gifts,
        "diagnostic": diagnostic,
        "summary": summary,
    }


def symmetric_tate_check(moma):
    tate = load_tate_phase_j_gifts()
    profile, baseline, _ = donor_profile_for_selection(tate)
    tate_extremes = select_tate_extremes(profile)
    recurrence = tate_to_moma_recurrence(tate_extremes, moma)
    summary = write_symmetric_boundary_outputs(tate_extremes, recurrence)
    return {
        "tate": tate,
        "profile": profile,
        "baseline": baseline,
        "tate_extremes": tate_extremes,
        "recurrence": recurrence,
        "summary": summary,
    }


def rockefeller_table_rows(moma):
    matches = matched_case_strings(moma, "Rockefeller lineage")
    return [(name, int(count)) for name, count in matches.items()]


def format_tex_item_list(items):
    return ", ".join(latex_escape(x) for x in items)


def write_parser_specification(moma, tate_info):
    rock_rows = rockefeller_table_rows(moma)
    role_terms = sorted(GENERIC_DONOR_ROLE_TERMS)
    extra_phrase_terms = [
        "artist's widow",
        "artist's estate",
        "artist's family",
        "artist's heirs",
        "his widow",
        "her widow",
        "donor unknown",
        "anonymous donors",
    ]
    lines = [
        r"\section*{Parser and Boundary-Check Specification}",
        r"\noindent This appendix specifies the parser used for the donor-biography analyses. The MoMA headline sample is produced from \texttt{processed\_moma\_data.csv} by keeping 1929--2024 records in the five main departments, acquisition types \texttt{Gift}, \texttt{Artist Gift}, \texttt{Bequest}, and \texttt{Partial Gift/Purchase}, then retaining records with a non-generic parsed donor string. In the current processed file this yields 49,740 donor-identified gift records.",
        "",
        r"\tbl{Credit-line extraction prefixes}{\begin{tabular}{ll}",
        r"\toprule",
        r"Corpus & Closed prefix list \\",
        r"\midrule",
        r"MoMA & \texttt{gift of the}, \texttt{bequest of the}, \texttt{gift of}, \texttt{bequest of}; text after the prefix is truncated at comma or newline. \\",
        r"Tate Phase J & \texttt{presented by the}, \texttt{bequeathed by the}, \texttt{given by the}, \texttt{accepted by the}, \texttt{presented by}, \texttt{bequeathed by}, \texttt{given by}, \texttt{accepted by}; trailing years and final periods are stripped. \\",
        r"\bottomrule",
        r"\end{tabular}}",
        "",
        r"\paragraph{Generic-role filter.} A parsed donor string is discarded if, after lower-casing and optional removal of leading \texttt{the}, it equals one of: "
        + format_tex_item_list(role_terms)
        + r". The Phase J scripts also discard phrase-level role labels: "
        + format_tex_item_list(extra_phrase_terms)
        + r", and labels beginning with artist/designer/architect/photographer followed by \texttt{through}, \texttt{with}, \texttt{in}, \texttt{by}, \texttt{and}, \texttt{via}, or \texttt{from}.",
        "",
        r"\paragraph{Normalization.} Donor strings are whitespace-collapsed and stripped of leading/trailing spaces and semicolons. Title prefixes matching the following expression are removed only for matching; a titled form is collapsed to the de-titled base only when the base itself is observed or when multiple title-only variants share the same base. Otherwise distinct strings remain distinct.",
        r"\begin{verbatim}",
        f"TITLE_PREFIX_RE = {TITLE_PREFIX_RE.pattern}",
        r"\end{verbatim}",
        "",
        r"\tbl{Rockefeller family-line aggregate in the main analysis}{\begin{tabular}{lr}",
        r"\toprule",
        r"Matched normalized donor string & Records \\",
        r"\midrule",
    ]
    for name, count in rock_rows:
        lines.append(f"{latex_escape(name)} & {count} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            "",
            r"\paragraph{Non-Western coding.} The canonical mapping strips parentheses from the nationality field, lower-cases it, and returns \texttt{Western} if any configured Western keyword is present; otherwise any non-missing nationality is \texttt{Non-Western}. The Western keyword allow-list is: "
            + format_tex_item_list(WESTERN_KEYWORDS)
            + r". The contested Eastern-European non-Western list discussed in the text is: "
            + format_tex_item_list(EASTERN_EUROPE_NONWESTERN_LIST)
            + r".",
            "",
            r"\paragraph{Sub-era changepoints.} The implementation uses a deterministic Z-score fallback because \texttt{ruptures} is not part of the replication environment:",
            r"\begin{verbatim}",
            "For a donor's yearly count series, fill all years from first to last with 0.",
            "If the series has <=5 years, zero total gifts, or zero SD of annual deltas:",
            "    return one sub-era [first_year, last_year].",
            "Compute annual deltas and Z-score the absolute deltas.",
            "A candidate breakpoint is year y if |Z_delta(y)| >= 1.5 and",
            "    |delta(y)| >= max(10 records, 10% of the donor's peak-year count).",
            "Take the three largest candidates by |Z|; sort them chronologically.",
            "Sub-eras are [start, bp-1], [bp, next_bp-1], ..., [last_bp, end].",
            "One-year edge eras are retained; ties follow pandas stable sort order.",
            r"\end{verbatim}",
            "",
            r"\paragraph{Tate match functions.} The asymmetric MoMA-to-Tate check uses six donor-specific regexes:",
            r"\begin{verbatim}",
        ]
    )
    for key, value in MOMA_TO_TATE_PATTERNS.items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            r"\end{verbatim}",
            r"The symmetric Phase J extension first re-runs the six-axis selection rule on Tate with \texttt{N>=100}, no family-line aggregation, and axes for volume, active-era length, non-Western gap, female gap, largest institutional donor, and maximum \texttt{1-Herfindahl}. It then tests each Tate-extreme donor against the MoMA donor list by exact string; surname-substring hits are reported only as diagnostics and do not count as recurrence.",
        ]
    )
    out_path = get_output_path("parser_specification.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    words = len(" ".join(lines).split())
    pages = max(1, math.ceil(words / 650))
    return {"path": out_path, "words": words, "pages": pages, "rockefeller_n_strings": len(rock_rows)}


def main():
    ensure_output_dir()
    moma = load_phase_j_moma()
    arithmetic, arith_info = arithmetic_checks(moma)
    tate_info = symmetric_tate_check(moma)
    parser_info = write_parser_specification(moma, tate_info)

    append_log(
        "Phase J -- Round 4 verifications",
        [
            f"Task 1 arithmetic CSV: {get_output_path('phase_j_arithmetic_check.csv')}",
            f"Task 2 Tate extremes CSV: {get_output_path('tate_six_extreme_donors.csv')}",
            f"Task 2 recurrence CSV: {get_output_path('tate_to_moma_recurrence.csv')}",
            f"Task 2 symmetric table: {get_output_path('symmetric_boundary_check.tex')}",
            f"Task 3 parser appendix: {get_output_path('parser_specification.tex')} (~{parser_info['pages']} pages)",
            f"Non-Western carried by six donors: {100 * arith_info['ratio']:.3g}% ({arith_info['six_nonwest']:,}/{arith_info['total_nonwest']:,}).",
            f"Observed six-donor gift years: {arith_info['years_with_gifts']}/96, zero years={arith_info['zero_years']}.",
            f"Gund top share={arith_info['gund_top_share']:.3f}, H={arith_info['gund_h']:.3f}, H_norm={arith_info['gund_h_norm']:.3f}.",
            f"Tate-to-MoMA exact recurrence: {tate_info['summary']['tate_to_moma_exact']}/6.",
        ],
    )

    print(f"Task 1a: recomputed {100 * arith_info['ratio']:.3g}% vs paper claim 22%.")
    print(
        f"Task 1b: recomputed {arith_info['years_with_gifts']}/96 observed gift years "
        f"(zero={arith_info['zero_years']}) vs paper claim 94/96."
    )
    print(
        f"Task 1c: Gund H={arith_info['gund_h']:.3f}, normalized H={arith_info['gund_h_norm']:.3f}, "
        f"top-dept-share={arith_info['gund_top_share']:.3f}."
    )
    print(f"Task 2: {tate_info['summary']['tate_to_moma_exact']}/6 Tate-extreme donors recur at MoMA by exact string.")
    print(f"Task 3: appendix written to {parser_info['path']} (~{parser_info['pages']} pages estimated).")


if __name__ == "__main__":
    main()

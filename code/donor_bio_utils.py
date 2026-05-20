# -*- coding: utf-8 -*-
"""
Shared helpers for the donor-biography experiments (34-37).
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GENERIC_DONOR_ROLE_TERMS, MAIN_DEPARTMENTS, PROJECT_ROOT, get_output_path, get_processed_path
from experiment_utils import safe_pct
from tate_utils import categorize_tate_acquisition, extract_tate_donor_from_creditline
from utils import extract_birth_year, extract_donor_from_gift


GIFT_TYPES = {"Gift", "Artist Gift", "Bequest", "Partial Gift/Purchase"}
TATE_GIFT_TYPES = {"Gift", "Bequest"}
CHUNK_SIZE = 25_000
BASELINE_FEMALE = 16.2
BASELINE_NONWEST = 15.6

CASE_ORDER = [
    "Kleiner",
    "Rockefeller lineage",
    "Jean Pigozzi",
    "Peter J. Cohen",
    "Judith Rothschild Foundation",
    "Agnes Gund",
]

CASE_SLUGS = {
    "Kleiner": "kleiner",
    "Rockefeller lineage": "rockefeller_lineage",
    "Jean Pigozzi": "jean_pigozzi",
    "Peter J. Cohen": "peter_j_cohen",
    "Judith Rothschild Foundation": "judith_rothschild_foundation",
    "Agnes Gund": "agnes_gund",
}

CASE_SHORT = {
    "Kleiner": "Kleiner",
    "Rockefeller lineage": "Rockefeller",
    "Jean Pigozzi": "Pigozzi",
    "Peter J. Cohen": "Cohen",
    "Judith Rothschild Foundation": "Rothschild Fdn.",
    "Agnes Gund": "Gund",
}

TITLE_PREFIX_RE = re.compile(
    r"^(?:(?:mr|mrs|ms|miss|dr|prof)\.?\s+(?:and\s+(?:mr|mrs|ms|miss|dr|prof)\.?\s+)?|"
    r"(?:sir|dame|lady)\s+)",
    re.IGNORECASE,
)


def ensure_output_dir():
    Path(get_output_path("")).mkdir(parents=True, exist_ok=True)


def compact_space(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = text.strip(" ;")
    return text if text else np.nan


def is_generic_donor_name(name):
    if pd.isna(name):
        return True
    low = re.sub(r"\s+", " ", str(name).strip().lower().strip("."))
    low_no_the = low[4:] if low.startswith("the ") else low
    if low in GENERIC_DONOR_ROLE_TERMS or low_no_the in GENERIC_DONOR_ROLE_TERMS:
        return True
    generic_exact = {
        "artist's widow",
        "the artist's widow",
        "artist's estate",
        "the artist's estate",
        "artist's family",
        "the artist's family",
        "artist's heirs",
        "the artist's heirs",
        "his widow",
        "her widow",
        "widow",
        "the donor",
        "donor unknown",
        "anonymous donors",
    }
    if low in generic_exact or low_no_the in generic_exact:
        return True
    if re.fullmatch(r"(?:the )?artist(?:'s)?\s+(?:widow|estate|family|heirs)", low):
        return True
    if re.match(
        r"^(?:the )?(?:artist|artists|architect|architects|designer|designers|"
        r"photographer|photographers)\s+(?:through|with|in|by|and|via|from)\b",
        low,
    ):
        return True
    if "the artist's estate" in low or "the artists' estate" in low:
        return True
    return False


def strip_title_prefix(name):
    text = compact_space(name)
    if pd.isna(text):
        return np.nan
    stripped = TITLE_PREFIX_RE.sub("", text).strip()
    return stripped if stripped else text


def is_specific_base(name):
    if pd.isna(name):
        return False
    text = str(name).strip()
    tokens = [t for t in re.split(r"\s+", text) if t]
    return len(tokens) >= 2 or bool(re.search(r"\b[A-Z]\.", text))


def canonicalize_donors(series):
    cleaned = series.apply(compact_space)
    cleaned = cleaned[~cleaned.apply(is_generic_donor_name)]
    names = cleaned.dropna().astype(str)
    bases = {name: strip_title_prefix(name) for name in names.unique()}
    base_counts = Counter(bases.values())
    observed = set(names.unique())
    mapping = {}
    for name, base in bases.items():
        if base != name and is_specific_base(base) and (base in observed or base_counts[base] > 1):
            mapping[name] = base
        else:
            mapping[name] = name
    return series.apply(compact_space).map(mapping)


def pct_female(df):
    valid = df["Gender_Grouped"].isin(["Female", "Male"])
    return safe_pct((df.loc[valid, "Gender_Grouped"] == "Female").sum(), valid.sum())


def pct_nonwest(df):
    valid = df["GeographicOrigin"].notna()
    return safe_pct((df.loc[valid, "GeographicOrigin"] == "Non-Western").sum(), valid.sum())


def count_female_valid(df):
    valid = df["Gender_Grouped"].isin(["Female", "Male"])
    return int((df.loc[valid, "Gender_Grouped"] == "Female").sum()), int(valid.sum())


def count_nonwest_valid(df):
    valid = df["GeographicOrigin"].notna()
    return int((df.loc[valid, "GeographicOrigin"] == "Non-Western").sum()), int(valid.sum())


def wilson_ci_pct(success, total, z=1.96):
    if total <= 0:
        return np.nan, np.nan
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denom
    return 100 * max(0.0, center - half), 100 * min(1.0, center + half)


def demographic_stats(df):
    f_success, f_total = count_female_valid(df)
    nw_success, nw_total = count_nonwest_valid(df)
    f_lo, f_hi = wilson_ci_pct(f_success, f_total)
    nw_lo, nw_hi = wilson_ci_pct(nw_success, nw_total)
    return {
        "pct_female": pct_female(df),
        "female_ci_lower": f_lo,
        "female_ci_upper": f_hi,
        "female_n_valid": f_total,
        "pct_nonwest": pct_nonwest(df),
        "nonwest_ci_lower": nw_lo,
        "nonwest_ci_upper": nw_hi,
        "nonwest_n_valid": nw_total,
    }


def fmt_float(value, digits=2):
    if pd.isna(value):
        return np.nan
    return round(float(value), digits)


def parse_artist_tokens(row):
    cid = row.get("ConstituentID", np.nan)
    if pd.notna(cid) and str(cid).strip():
        ids = re.findall(r"\d+", str(cid))
        if ids:
            return {f"id:{x}" for x in ids}
    artist = row.get("Artist", np.nan)
    if pd.notna(artist) and str(artist).strip():
        return {f"artist:{compact_space(artist)}"}
    return set()


def donor_artist_set(df):
    artists = set()
    for _, row in df.iterrows():
        artists.update(parse_artist_tokens(row))
    return artists


def load_moma_donor_gifts():
    path = get_processed_path()
    available = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = [
        "Title",
        "Artist",
        "ConstituentID",
        "AcquisitionType",
        "CreditLine",
        "YearAcquired",
        "Department",
        "Gender_Grouped",
        "GeographicOrigin",
        "ArtistBirthYear",
        "BeginDate",
        "Classification",
        "Medium",
        "ObjectID",
    ]
    usecols = [c for c in wanted if c in available]
    chunks = []
    reader = pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, engine="python", on_bad_lines="skip")
    for chunk in reader:
        for col in wanted:
            if col not in chunk.columns:
                chunk[col] = np.nan
        chunk["YearAcquired"] = pd.to_numeric(chunk["YearAcquired"], errors="coerce")
        chunk["ArtistBirthYear"] = pd.to_numeric(chunk["ArtistBirthYear"], errors="coerce")
        missing_birth = chunk["ArtistBirthYear"].isna() & chunk["BeginDate"].notna()
        if missing_birth.any():
            chunk.loc[missing_birth, "ArtistBirthYear"] = chunk.loc[missing_birth, "BeginDate"].apply(
                extract_birth_year
            )
        chunk = chunk[
            chunk["YearAcquired"].between(1929, 2024)
            & chunk["Department"].isin(MAIN_DEPARTMENTS)
            & chunk["AcquisitionType"].isin(GIFT_TYPES)
        ].copy()
        if chunk.empty:
            continue
        chunk["DonorRaw"] = chunk["CreditLine"].apply(extract_donor_from_gift)
        chunk = chunk[chunk["DonorRaw"].notna()].copy()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=wanted + ["DonorRaw", "Donor"])
    df = pd.concat(chunks, ignore_index=True)
    df["Donor"] = canonicalize_donors(df["DonorRaw"])
    df = df[df["Donor"].notna()].copy()
    df["YearAcquired"] = df["YearAcquired"].astype(int)
    return df


def is_rockefeller_donor(name):
    if pd.isna(name):
        return False
    low = str(name).lower()
    low = re.sub(r"\s+", " ", low).strip()
    clear_prefixes = (
        "abby aldrich rockefeller",
        "blanchette hooker rockefeller",
        "family of blanchette hooker rockefeller",
        "nelson a. rockefeller",
        "nelson rockefeller",
        "mrs. john d. rockefeller",
        "mr. and mrs. john d. rockefeller",
        "john d. rockefeller",
        "john d. rockefeller iii",
        "david rockefeller",
        "david and peggy rockefeller",
        "mr. and mrs. david rockefeller",
        "steven c. rockefeller",
    )
    return any(low.startswith(prefix) for prefix in clear_prefixes)


def case_for_donor(name):
    if pd.isna(name):
        return np.nan
    text = str(name).strip()
    if text == "Kleiner":
        return "Kleiner"
    if is_rockefeller_donor(text):
        return "Rockefeller lineage"
    if text == "Jean Pigozzi":
        return "Jean Pigozzi"
    if text == "Peter J. Cohen":
        return "Peter J. Cohen"
    if text == "Judith Rothschild Foundation":
        return "Judith Rothschild Foundation"
    if text == "Agnes Gund":
        return "Agnes Gund"
    return np.nan


def assign_case(df):
    out = df.copy()
    out["CaseDonor"] = out["Donor"].apply(case_for_donor)
    return out


def matched_case_strings(df, case_name):
    case_df = assign_case(df)
    sub = case_df[case_df["CaseDonor"] == case_name]
    if sub.empty:
        return pd.Series(dtype=int)
    return sub["Donor"].value_counts()


def top_breakdown(series, n=5):
    clean = series.dropna().astype(str).map(compact_space)
    clean = clean[clean.notna()]
    if clean.empty:
        return ""
    vc = clean.value_counts().head(n)
    total = len(clean)
    return "|".join(f"{name}:{int(count)} ({100 * count / total:.1f}%)" for name, count in vc.items())


def dept_counts_and_shares(df):
    counts = df["Department"].value_counts()
    total = len(df)
    rows = []
    for dept in MAIN_DEPARTMENTS:
        n = int(counts.get(dept, 0))
        rows.append((dept, n, n / total if total else np.nan))
    return rows


def footprint_type(shares):
    ordered = sorted([s for s in shares if pd.notna(s)], reverse=True)
    if not ordered:
        return "unknown"
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    if top >= 0.9:
        return "single-department"
    if top >= 0.5 and second >= 0.2:
        return "primary-secondary"
    return "spread"


def detect_suberas(year_counts):
    if year_counts.empty:
        return []
    start = int(year_counts.index.min())
    end = int(year_counts.index.max())
    if start == end:
        return [(start, end, [])]
    full = year_counts.reindex(range(start, end + 1), fill_value=0).astype(float)
    if len(full) <= 5 or full.sum() == 0:
        return [(start, end, [])]
    deltas = full.diff().dropna()
    if deltas.std(ddof=0) == 0 or pd.isna(deltas.std(ddof=0)):
        return [(start, end, [])]
    z = (deltas - deltas.mean()) / deltas.std(ddof=0)
    peak = max(1.0, float(full.max()))
    candidates = []
    for year, zval in z.abs().sort_values(ascending=False).items():
        delta = abs(float(deltas.loc[year]))
        if zval >= 1.5 and delta >= max(10.0, 0.10 * peak):
            candidates.append(int(year))
    breakpoints = sorted(candidates[:3])
    if not breakpoints:
        return [(start, end, [])]
    eras = []
    current = start
    for bp in breakpoints:
        if current <= bp - 1:
            eras.append((current, bp - 1, breakpoints))
        current = bp
    if current <= end:
        eras.append((current, end, breakpoints))
    return eras or [(start, end, [])]


def counterfactual_stats(base_df, remove_mask, scope_mask=None):
    if scope_mask is None:
        scope_mask = pd.Series(True, index=base_df.index)
    actual = base_df.loc[scope_mask]
    without = base_df.loc[scope_mask & ~remove_mask]
    actual_demo = demographic_stats(actual)
    without_demo = demographic_stats(without)
    return {
        "with_female_pct": actual_demo["pct_female"],
        "without_female_pct": without_demo["pct_female"],
        "delta_female_pp": without_demo["pct_female"] - actual_demo["pct_female"]
        if pd.notna(without_demo["pct_female"]) and pd.notna(actual_demo["pct_female"])
        else np.nan,
        "with_nonwest_pct": actual_demo["pct_nonwest"],
        "without_nonwest_pct": without_demo["pct_nonwest"],
        "delta_nonwest_pp": without_demo["pct_nonwest"] - actual_demo["pct_nonwest"]
        if pd.notna(without_demo["pct_nonwest"]) and pd.notna(actual_demo["pct_nonwest"])
        else np.nan,
        "with_n": int(len(actual)),
        "without_n": int(len(without)),
    }


def load_tate_records():
    proc_path = os.path.join(PROJECT_ROOT, "data", "tate", "processed_tate_data.csv")
    raw_path = os.path.join(PROJECT_ROOT, "data", "tate", "artwork_data.csv")
    proc = pd.read_csv(proc_path)
    raw_cols = ["id", "artist", "artistId", "medium", "creditLine", "acquisitionYear"]
    raw = pd.read_csv(raw_path, usecols=[c for c in raw_cols if c in pd.read_csv(raw_path, nrows=0).columns])
    if len(proc) == len(raw):
        df = proc.copy()
        for col in raw.columns:
            if col == "creditLine":
                continue
            if col == "artistId" and col in df.columns:
                continue
            df[col] = raw[col].values
    else:
        raw = raw.rename(columns={"acquisitionYear": "YearAcquiredRaw"})
        raw["YearAcquiredRaw"] = pd.to_numeric(raw["YearAcquiredRaw"], errors="coerce")
        proc["YearAcquired"] = pd.to_numeric(proc["YearAcquired"], errors="coerce")
        df = proc.merge(
            raw,
            left_on=["artistId", "creditLine", "YearAcquired"],
            right_on=["artistId", "creditLine", "YearAcquiredRaw"],
            how="left",
        )
    df["YearAcquired"] = pd.to_numeric(df["YearAcquired"], errors="coerce")
    if "AcquisitionType" not in df.columns or df["AcquisitionType"].isna().all():
        df["AcquisitionType"] = df["creditLine"].apply(categorize_tate_acquisition)
    df["DonorRaw"] = df["creditLine"].apply(extract_tate_donor_from_creditline)
    df["Donor"] = canonicalize_donors(df["DonorRaw"])
    return df

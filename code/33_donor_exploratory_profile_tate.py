# -*- coding: utf-8 -*-
"""
Experiment 15: exploratory donor profile for Tate.

This mirrors Experiment 14 where Tate fields permit it, using Tate's gift and
bequest credit lines plus processed demographic fields.
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GENERIC_DONOR_ROLE_TERMS, PROJECT_ROOT, get_output_path
from experiment_utils import append_log, safe_pct
from tate_utils import categorize_tate_acquisition, extract_tate_donor_from_creditline


GIFT_TYPES = {"Gift", "Bequest"}
INSTITUTION_RE = re.compile(
    r"Foundation|Fund|Trust|Estate of|Inc\.|Corporation|Committee|Council|Endowment|Charitable|Collection",
    re.IGNORECASE,
)
TITLE_PREFIX_RE = re.compile(
    r"^(?:(?:mr|mrs|ms|miss|dr|prof)\.?\s+(?:and\s+(?:mr|mrs|ms|miss|dr|prof)\.?\s+)?|"
    r"(?:sir|dame|lady)\s+)",
    re.IGNORECASE,
)
LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "_": r"\_",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
}


def ensure_output_dir():
    Path(get_output_path("")).mkdir(parents=True, exist_ok=True)


def latex_escape(value):
    text = "" if pd.isna(value) else str(value)
    for old, new in LATEX_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


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


def fmt_float(value, digits=2):
    if pd.isna(value):
        return np.nan
    return round(float(value), digits)


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return "---"
    return f"{float(value):.{digits}f}\\%"


def parse_artist_tokens(row):
    artist_id = row.get("artistId", np.nan)
    if pd.notna(artist_id) and str(artist_id).strip():
        found = re.findall(r"\d+", str(artist_id))
        if found:
            return {f"id:{found[0]}"}
    artist = row.get("artist", np.nan)
    if pd.notna(artist) and str(artist).strip():
        return {f"artist:{compact_space(artist)}"}
    return set()


def donor_artist_set(df):
    artists = set()
    for _, row in df.iterrows():
        artists.update(parse_artist_tokens(row))
    return artists


def tate_path(filename):
    return os.path.join(PROJECT_ROOT, "data", "tate", filename)


def read_existing(path, columns=None):
    available = pd.read_csv(path, nrows=0).columns.tolist()
    if columns is None:
        usecols = available
    else:
        usecols = [c for c in columns if c in available]
    return pd.read_csv(path, usecols=usecols)


def load_tate_gift_sample():
    processed_path = tate_path("processed_tate_data.csv")
    raw_path = tate_path("artwork_data.csv")
    proc_cols = [
        "artistId",
        "creditLine",
        "YearAcquired",
        "AcquisitionType",
        "Department",
        "Gender_Grouped",
        "GeographicOrigin",
        "ArtistBirthYear",
        "ArtistAgeAtAcquisition",
    ]
    raw_cols = ["id", "artist", "artistId", "medium", "creditLine", "acquisitionYear"]
    proc = read_existing(processed_path, proc_cols)
    raw = read_existing(raw_path, raw_cols)

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

    for col in proc_cols + raw_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["YearAcquired"] = pd.to_numeric(df["YearAcquired"], errors="coerce")
    if "acquisitionYear" in df.columns:
        raw_year = pd.to_numeric(df["acquisitionYear"], errors="coerce")
        df["YearAcquired"] = df["YearAcquired"].fillna(raw_year)
    if "AcquisitionType" not in df.columns or df["AcquisitionType"].isna().all():
        df["AcquisitionType"] = df["creditLine"].apply(categorize_tate_acquisition)
    df["Department"] = df["Department"].fillna("Unknown").astype(str)
    df["ArtistBirthYear"] = pd.to_numeric(df["ArtistBirthYear"], errors="coerce")

    df = df[df["YearAcquired"].notna() & df["AcquisitionType"].isin(GIFT_TYPES)].copy()
    df["DonorRaw"] = df["creditLine"].apply(extract_tate_donor_from_creditline)
    df = df[df["DonorRaw"].notna()].copy()
    df["Donor"] = canonicalize_donors(df["DonorRaw"])
    df = df[df["Donor"].notna()].copy()
    df["YearAcquired"] = df["YearAcquired"].astype(int)
    return df


def top_value(series):
    clean = series.dropna().astype(str).map(compact_space)
    clean = clean[clean.notna()]
    if clean.empty:
        return np.nan
    return clean.value_counts().index[0]


def build_profile(df, top_k=100):
    museum_female = pct_female(df)
    museum_nonwest = pct_nonwest(df)
    top_donors = df["Donor"].value_counts().head(top_k).index.tolist()
    departments = df["Department"].value_counts().index.tolist()
    rows = []
    for donor in top_donors:
        d = df[df["Donor"] == donor].copy()
        n = int(len(d))
        years = d["YearAcquired"].dropna().astype(int)
        year_counts = years.value_counts().sort_values(ascending=False)
        dept_counts = d["Department"].value_counts()
        top_dept = dept_counts.index[0]
        top_dept_n = int(dept_counts.iloc[0])
        births = pd.to_numeric(d["ArtistBirthYear"], errors="coerce").dropna()
        artists = donor_artist_set(d)
        pct_f = pct_female(d)
        pct_nw = pct_nonwest(d)
        row = {
            "donor": donor,
            "n_gifts": n,
            "active_era_start": int(years.min()) if not years.empty else np.nan,
            "active_era_end": int(years.max()) if not years.empty else np.nan,
            "active_era_years": int(years.max() - years.min()) if not years.empty else np.nan,
            "peak_year": int(year_counts.index[0]) if not year_counts.empty else np.nan,
            "peak_year_n": int(year_counts.iloc[0]) if not year_counts.empty else 0,
            "dept_top": top_dept,
            "dept_top_n": top_dept_n,
            "dept_share_top": round(top_dept_n / n, 3) if n else np.nan,
            "dept_breakdown": "|".join(f"{dept}:{int(dept_counts.get(dept, 0))}" for dept in departments),
            "medium_top": top_value(d["medium"]),
            "pct_female": fmt_float(pct_f),
            "pct_nonwest": fmt_float(pct_nw),
            "pct_female_vs_museum": fmt_float(pct_f - museum_female) if pd.notna(pct_f) else np.nan,
            "pct_nonwest_vs_museum": fmt_float(pct_nw - museum_nonwest) if pd.notna(pct_nw) else np.nan,
            "artist_birth_year_median": fmt_float(births.median(), 1) if not births.empty else np.nan,
            "artist_birth_year_iqr": fmt_float(births.quantile(0.75) - births.quantile(0.25), 1)
            if not births.empty
            else np.nan,
            "n_unique_artists": int(len(artists)),
            "is_institutional": bool(INSTITUTION_RE.search(donor)),
            "bequest_share": round(float((d["AcquisitionType"] == "Bequest").mean()), 3),
        }
        rows.append(row)
    return pd.DataFrame(rows), museum_female, museum_nonwest


def write_top20_table(profile):
    top20 = profile.head(20).copy()
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\tbl{Top donor-identified gift contributors in the Tate sample}",
        r"{\begin{tabular}{p{0.30\textwidth}rp{0.12\textwidth}p{0.21\textwidth}ccr}",
        r"\toprule",
        r"Donor & N & Active era & Top group & \% Female & \% Non-Western & Median birth year \\",
        r"\midrule",
    ]
    for _, row in top20.iterrows():
        era = f"{int(row['active_era_start'])}-{int(row['active_era_end'])}"
        top_dept = f"{row['dept_top']} ({int(row['dept_top_n'])})"
        median_birth = "---" if pd.isna(row["artist_birth_year_median"]) else f"{row['artist_birth_year_median']:.0f}"
        lines.append(
            f"{latex_escape(row['donor'])} & {int(row['n_gifts'])} & {latex_escape(era)} & "
            f"{latex_escape(top_dept)} & {fmt_pct(row['pct_female'])} & "
            f"{fmt_pct(row['pct_nonwest'])} & {median_birth} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\footnotesize{Sample: donor-identified gifts and bequests in the processed Tate collection file. Donor names are parsed from credit lines and conservatively normalized across title-only variants.}",
            r"\label{tab:donor_exploratory_top20_tate}",
            r"\end{table}",
        ]
    )
    with open(get_output_path("table_donor_exploratory_top20_tate.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def structure_summary(profile):
    top20 = profile.head(20)
    top100_n = profile["n_gifts"].sum()
    top20_n = top20["n_gifts"].sum()
    institutional_gifts = top20.loc[top20["is_institutional"], "n_gifts"].sum()
    female_idx = top20["pct_female_vs_museum"].abs().idxmax()
    nonwest_idx = top20["pct_nonwest_vs_museum"].abs().idxmax()
    return {
        "top_donor": top20.iloc[0]["donor"],
        "top_donor_n": int(top20.iloc[0]["n_gifts"]),
        "top_donor_share_top100": 100 * top20.iloc[0]["n_gifts"] / top100_n if top100_n else np.nan,
        "top20_share_top100": 100 * top20_n / top100_n if top100_n else np.nan,
        "institutional_top20_count": int(top20["is_institutional"].sum()),
        "institutional_top20_gift_share": 100 * institutional_gifts / top20_n if top20_n else np.nan,
        "female_outlier": top20.loc[female_idx, "donor"],
        "female_outlier_pp": top20.loc[female_idx, "pct_female_vs_museum"],
        "nonwest_outlier": top20.loc[nonwest_idx, "donor"],
        "nonwest_outlier_pp": top20.loc[nonwest_idx, "pct_nonwest_vs_museum"],
    }


def append_tate_log(profile, df, museum_female, museum_nonwest):
    summary = structure_summary(profile)
    bullets = [
        f"Saved Tate profile CSV: {get_output_path('donor_exploratory_profile_tate.csv')}",
        f"Saved Tate top-20 TEX table: {get_output_path('table_donor_exploratory_top20_tate.tex')}",
        f"Donor-identified Tate sample: {len(df):,} records and {df['Donor'].nunique():,} normalized donors. Baseline is {museum_female:.1f}% female and {museum_nonwest:.1f}% non-Western among valid rows.",
        f"Tate top donor: {summary['top_donor']} ({summary['top_donor_n']:,} gifts; {summary['top_donor_share_top100']:.1f}% of top-100 gifts).",
        f"Tate top-20 concentration: top 20 account for {summary['top20_share_top100']:.1f}% of top-100 gifts.",
        f"Tate institutional presence in top 20: {summary['institutional_top20_count']}/20 donors; {summary['institutional_top20_gift_share']:.1f}% of top-20 gifts.",
        f"Tate strongest top-20 female outlier: {summary['female_outlier']} ({summary['female_outlier_pp']:+.1f} pp from Tate donor baseline).",
        f"Tate strongest top-20 non-Western outlier: {summary['nonwest_outlier']} ({summary['nonwest_outlier_pp']:+.1f} pp from Tate donor baseline).",
    ]

    moma_path = Path(get_output_path("donor_exploratory_profile.csv"))
    if moma_path.exists():
        moma = pd.read_csv(moma_path)
        moma_summary = structure_summary(moma)
        bullets.extend(
            [
                f"MoMA comparison: top donor {moma_summary['top_donor']} ({moma_summary['top_donor_n']:,} gifts; {moma_summary['top_donor_share_top100']:.1f}% of top-100 gifts).",
                f"MoMA comparison: top 20 account for {moma_summary['top20_share_top100']:.1f}% of top-100 gifts.",
                f"MoMA comparison: institutional top-20 count is {moma_summary['institutional_top20_count']}/20, gift share {moma_summary['institutional_top20_gift_share']:.1f}%.",
                f"MoMA comparison: strongest top-20 female outlier is {moma_summary['female_outlier']} ({moma_summary['female_outlier_pp']:+.1f} pp); strongest non-Western outlier is {moma_summary['nonwest_outlier']} ({moma_summary['nonwest_outlier_pp']:+.1f} pp).",
            ]
        )
    append_log("Experiment 15 -- Tate donor exploratory benchmark", bullets)


def main():
    ensure_output_dir()
    df = load_tate_gift_sample()
    if df.empty:
        raise RuntimeError("No donor-identified Tate gift records found.")
    profile, museum_female, museum_nonwest = build_profile(df, top_k=100)
    profile.to_csv(get_output_path("donor_exploratory_profile_tate.csv"), index=False)
    write_top20_table(profile)
    append_tate_log(profile, df, museum_female, museum_nonwest)
    print(
        "Completed Experiment 15 | records=%s donors=%s top=%s (%s gifts)"
        % (f"{len(df):,}", f"{df['Donor'].nunique():,}", profile.iloc[0]["donor"], int(profile.iloc[0]["n_gifts"]))
    )


if __name__ == "__main__":
    main()

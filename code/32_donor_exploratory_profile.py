# -*- coding: utf-8 -*-
"""
Experiment 14: exploratory profile of MoMA mega-donors.

Reads the processed MoMA file, extracts donor-identified gifts in the main
departments, profiles the top 100 donors by gift count, and writes the CSV,
LaTeX table, and markdown log requested for the donor-biography paper.
"""

import itertools
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GENERIC_DONOR_ROLE_TERMS, MAIN_DEPARTMENTS, get_output_path, get_processed_path
from experiment_utils import append_log, safe_pct
from utils import extract_birth_year, extract_donor_from_gift


GIFT_TYPES = {"Gift", "Artist Gift", "Bequest", "Partial Gift/Purchase"}
CHUNK_SIZE = 25_000

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


def load_moma_gift_sample():
    path = get_processed_path()
    available = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = [
        "AcquisitionType",
        "CreditLine",
        "YearAcquired",
        "Department",
        "Gender_Grouped",
        "GeographicOrigin",
        "ArtistBirthYear",
        "BeginDate",
        "Artist",
        "ConstituentID",
        "Classification",
        "Medium",
    ]
    usecols = [c for c in wanted if c in available]
    chunks = []
    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        engine="python",
        on_bad_lines="skip",
    )
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

    rows = []
    donor_artists = {}
    for donor in top_donors:
        d = df[df["Donor"] == donor].copy()
        n = int(len(d))
        years = d["YearAcquired"].dropna().astype(int)
        year_counts = years.value_counts().sort_values(ascending=False)
        dept_counts = d["Department"].value_counts()
        top_dept = dept_counts.index[0]
        top_dept_n = int(dept_counts.iloc[0])
        class_source = d["Classification"].where(d["Classification"].notna(), d["Medium"])
        births = pd.to_numeric(d["ArtistBirthYear"], errors="coerce").dropna()
        artists = donor_artist_set(d)
        donor_artists[donor] = artists
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
            "dept_breakdown": "|".join(f"{dept}:{int(dept_counts.get(dept, 0))}" for dept in MAIN_DEPARTMENTS),
            "medium_top": top_value(class_source),
            "pct_female": fmt_float(pct_female(d)),
            "pct_nonwest": fmt_float(pct_nonwest(d)),
            "artist_birth_year_median": fmt_float(births.median(), 1) if not births.empty else np.nan,
            "artist_birth_year_iqr": fmt_float(births.quantile(0.75) - births.quantile(0.25), 1)
            if not births.empty
            else np.nan,
            "n_unique_artists": int(len(artists)),
            "is_institutional": bool(INSTITUTION_RE.search(donor)),
            "bequest_share": round(float((d["AcquisitionType"] == "Bequest").mean()), 3),
        }
        row["pct_female_vs_museum"] = fmt_float(row["pct_female"] - museum_female) if pd.notna(row["pct_female"]) else np.nan
        row["pct_nonwest_vs_museum"] = (
            fmt_float(row["pct_nonwest"] - museum_nonwest) if pd.notna(row["pct_nonwest"]) else np.nan
        )
        rows.append(row)

    profile = pd.DataFrame(rows)
    return profile, donor_artists, museum_female, museum_nonwest


def write_year_contributions(df, top_donors):
    rows = []
    top_df = df[df["Donor"].isin(top_donors)].copy()
    for (donor, year), group in top_df.groupby(["Donor", "YearAcquired"], sort=True):
        dept_counts = group["Department"].value_counts()
        rows.append(
            {
                "donor": donor,
                "year": int(year),
                "n_gifts": int(len(group)),
                "pct_female_this_year": fmt_float(pct_female(group)),
                "pct_nonwest_this_year": fmt_float(pct_nonwest(group)),
                "top_dept_this_year": dept_counts.index[0] if not dept_counts.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["donor", "year"])
    out.to_csv(get_output_path("donor_year_contributions.csv"), index=False)
    return out


def write_overlap(profile, donor_artists):
    rows = []
    donors = profile["donor"].tolist()
    for donor_a, donor_b in itertools.combinations(donors, 2):
        n_shared = len(donor_artists.get(donor_a, set()) & donor_artists.get(donor_b, set()))
        if n_shared > 0:
            rows.append({"donor_a": donor_a, "donor_b": donor_b, "n_shared_artists": int(n_shared)})
    out = pd.DataFrame(rows, columns=["donor_a", "donor_b", "n_shared_artists"])
    out.to_csv(get_output_path("donor_artist_overlap.csv"), index=False)
    return out


def write_post_donor_drift(df, profile):
    rows = []
    for _, donor_row in profile.iterrows():
        end_year = donor_row["active_era_end"]
        if pd.isna(end_year) or int(end_year) > 2015:
            continue
        end_year = int(end_year)
        top_dept = donor_row["dept_top"]
        dept_df = df[df["Department"] == top_dept]
        pre = dept_df[dept_df["YearAcquired"].between(end_year - 5, end_year - 1)]
        post = dept_df[dept_df["YearAcquired"].between(end_year + 1, end_year + 5)]
        female_pre = pct_female(pre)
        female_post = pct_female(post)
        nw_pre = pct_nonwest(pre)
        nw_post = pct_nonwest(post)
        rows.append(
            {
                "donor": donor_row["donor"],
                "top_dept": top_dept,
                "female_share_pre5": fmt_float(female_pre),
                "female_share_post5": fmt_float(female_post),
                "delta_female_pp": fmt_float(female_post - female_pre)
                if pd.notna(female_post) and pd.notna(female_pre)
                else np.nan,
                "nonwest_share_pre5": fmt_float(nw_pre),
                "nonwest_share_post5": fmt_float(nw_post),
                "delta_nonwest_pp": fmt_float(nw_post - nw_pre)
                if pd.notna(nw_post) and pd.notna(nw_pre)
                else np.nan,
                "n_works_pre5": int(len(pre)),
                "n_works_post5": int(len(post)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("post_donor_drift.csv"), index=False)
    return out


def write_top20_table(profile):
    top20 = profile.head(20).copy()
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\tbl{Top donor-identified gift contributors in the MoMA main-department sample}",
        r"{\begin{tabular}{p{0.30\textwidth}rp{0.12\textwidth}p{0.21\textwidth}ccr}",
        r"\toprule",
        r"Donor & N & Active era & Top dept & \% Female & \% Non-Western & Median birth year \\",
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
            r"\footnotesize{Sample: donor-identified gifts, artist gifts, bequests, and partial gift/purchases in the five main MoMA departments, 1929-2024. Donor names are parsed from credit lines and conservatively normalized across title-only variants.}",
            r"\label{tab:donor_exploratory_top20}",
            r"\end{table}",
        ]
    )
    with open(get_output_path("table_donor_exploratory_top20.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def join_donor_values(rows, value_col, suffix="pp", n=5):
    parts = []
    for _, row in rows.head(n).iterrows():
        value = row[value_col]
        if pd.isna(value):
            continue
        value_text = f"{value:+.1f} {suffix}" if suffix == "pp" else f"{value:.3f}"
        parts.append(f"{row['donor']} ({value_text}, N={int(row['n_gifts'])})")
    return "; ".join(parts) if parts else "n/a"


def choose_candidate_bullets(profile):
    pool = profile[profile["n_gifts"] >= 100].copy()
    if pool.empty:
        pool = profile.copy()
    selected = []
    seen = set()

    def add(row, reason):
        donor = row["donor"]
        if donor in seen:
            return
        seen.add(donor)
        selected.append(
            f"{donor}: {reason} N={int(row['n_gifts'])}, active era "
            f"{int(row['active_era_start'])}-{int(row['active_era_end'])}, top department "
            f"{row['dept_top']} ({row['dept_share_top']:.1%})."
        )

    add(pool.sort_values("n_gifts", ascending=False).iloc[0], "largest donor-identified contribution;")
    add(
        pool.sort_values("pct_female_vs_museum", ascending=False).iloc[0],
        "strong positive female-representation outlier;",
    )
    add(
        pool.sort_values("pct_nonwest_vs_museum", ascending=False).iloc[0],
        "strong positive non-Western-representation outlier;",
    )
    add(pool.sort_values("active_era_years", ascending=False).iloc[0], "long active-era case;")
    institutional = pool[pool["is_institutional"]]
    if not institutional.empty:
        add(institutional.sort_values("n_gifts", ascending=False).iloc[0], "high-volume institutional case;")
    if len(selected) < 5:
        add(pool.sort_values("dept_share_top", ascending=False).iloc[0], "highly department-concentrated case;")
    return selected[:5]


def append_moma_log(profile, df, year_rows, overlap_rows, drift_rows, museum_female, museum_nonwest):
    high_female = profile.sort_values("pct_female_vs_museum", ascending=False)
    low_female = profile.sort_values("pct_female_vs_museum")
    high_nw = profile.sort_values("pct_nonwest_vs_museum", ascending=False)
    low_nw = profile.sort_values("pct_nonwest_vs_museum")
    concentrated = profile.sort_values("dept_share_top", ascending=False)
    spread = profile.sort_values("dept_share_top")
    longest = profile.sort_values("active_era_years", ascending=False)
    top20 = profile.head(20)
    inst_count = int(profile["is_institutional"].sum())
    inst_top20 = int(top20["is_institutional"].sum())

    bullets = [
        f"Saved profile CSV: {get_output_path('donor_exploratory_profile.csv')}",
        f"Saved yearly timeline CSV: {get_output_path('donor_year_contributions.csv')} ({len(year_rows):,} rows)",
        f"Saved donor-artist overlap CSV: {get_output_path('donor_artist_overlap.csv')} ({len(overlap_rows):,} nonzero pairs)",
        f"Saved post-donor drift CSV: {get_output_path('post_donor_drift.csv')} ({len(drift_rows):,} donors with post windows)",
        f"Saved top-20 TEX table: {get_output_path('table_donor_exploratory_top20.tex')}",
        f"Donor-identified MoMA sample: {len(df):,} records and {df['Donor'].nunique():,} normalized donors. Museum-wide donor baseline is {museum_female:.1f}% female and {museum_nonwest:.1f}% non-Western among valid rows.",
        "Largest positive female divergence: " + join_donor_values(high_female, "pct_female_vs_museum"),
        "Largest negative female divergence: " + join_donor_values(low_female, "pct_female_vs_museum"),
        "Largest positive non-Western divergence: " + join_donor_values(high_nw, "pct_nonwest_vs_museum"),
        "Largest negative non-Western divergence: " + join_donor_values(low_nw, "pct_nonwest_vs_museum"),
        "Most single-department concentrated donors: " + join_donor_values(concentrated, "dept_share_top", suffix="share"),
        "Most department-spread donors: " + join_donor_values(spread, "dept_share_top", suffix="share"),
        "Longest active eras: "
        + "; ".join(
            f"{r['donor']} ({int(r['active_era_start'])}-{int(r['active_era_end'])}, {int(r['active_era_years'])} years)"
            for _, r in longest.head(5).iterrows()
        ),
        f"Institutional donors: {inst_count}/100 in the top 100, {inst_top20}/20 in the top 20.",
    ]
    bullets.extend("Biography-candidate outlier: " + item for item in choose_candidate_bullets(profile))
    append_log("Experiment 14 -- MoMA mega-donor exploratory profile", bullets)


def main():
    ensure_output_dir()
    df = load_moma_gift_sample()
    if df.empty:
        raise RuntimeError("No donor-identified MoMA gift records found.")

    profile, donor_artists, museum_female, museum_nonwest = build_profile(df, top_k=100)
    profile.to_csv(get_output_path("donor_exploratory_profile.csv"), index=False)

    top_donors = profile["donor"].tolist()
    year_rows = write_year_contributions(df, top_donors)
    overlap_rows = write_overlap(profile, donor_artists)
    drift_rows = write_post_donor_drift(df, profile)
    write_top20_table(profile)
    append_moma_log(profile, df, year_rows, overlap_rows, drift_rows, museum_female, museum_nonwest)

    print(
        "Completed Experiment 14 | records=%s donors=%s top=%s (%s gifts)"
        % (f"{len(df):,}", f"{df['Donor'].nunique():,}", profile.iloc[0]["donor"], int(profile.iloc[0]["n_gifts"]))
    )


if __name__ == "__main__":
    main()

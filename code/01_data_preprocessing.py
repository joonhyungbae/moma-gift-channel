"""
Build processed MoMA dataset for sn-article: AcquisitionType, Gender_Grouped, GeographicOrigin, YearAcquired, ArtistBirthYear, ArtistAgeAtAcquisition.
데이터는 data/ 폴더에서 읽음 (Artworks_full.csv 등). 결과는 data/processed_moma_data.csv 에 저장.
"""

import os
import sys
import gc
import json
from datetime import datetime
import pandas as pd
import numpy as np

from config import get_artworks_path, get_processed_path, get_output_path, PROCESSED_CSV, DATA_DIR
from utils import (
    categorize_acquisition,
    gender_grouped,
    geographic_origin,
    extract_birth_year,
    extract_named_funds,
)


def load_artworks(path):
    """Load Artworks CSV (mixed types 허용)."""
    return pd.read_csv(path, low_memory=False)


def main():
    # data/ 에서 Artworks 실제 파일 탐지 (Artworks.csv 또는 Artworks_full.csv 등)
    artworks_path = get_artworks_path()
    if not artworks_path:
        # 원본 없으면 기존 전처리 파일에 ArtistBirthYear, ArtistAgeAtAcquisition, NamedFund만 보강
        processed_path = get_processed_path()
        if os.path.isfile(processed_path):
            print(f"Artworks 원본 없음. 전처리 파일 보강: {processed_path}")
            df = pd.read_csv(processed_path, low_memory=False)
            if "BeginDate" in df.columns and "ArtistBirthYear" not in df.columns:
                df["ArtistBirthYear"] = df["BeginDate"].apply(extract_birth_year)
            if "YearAcquired" not in df.columns and "DateAcquired" in df.columns:
                df["YearAcquired"] = pd.to_datetime(df["DateAcquired"], errors="coerce").dt.year
            if "YearAcquired" in df.columns:
                df["YearAcquired"] = pd.to_numeric(df["YearAcquired"], errors="coerce")
            if "ArtistAgeAtAcquisition" not in df.columns:
                df["ArtistAgeAtAcquisition"] = np.nan
                m = df["YearAcquired"].notna() & df["ArtistBirthYear"].notna()
                df.loc[m, "ArtistAgeAtAcquisition"] = df.loc[m, "YearAcquired"] - df.loc[m, "ArtistBirthYear"]
            if "NamedFund" not in df.columns and "CreditLine" in df.columns:
                def first_fund(cl):
                    funds = extract_named_funds(cl)
                    return funds[0] if funds else np.nan
                purchase_mask = df["AcquisitionType"] == "Purchase"
                fund_series = df.loc[purchase_mask, "CreditLine"].apply(first_fund)
                df["NamedFund"] = np.empty(len(df), dtype=object)
                df.loc[purchase_mask, "NamedFund"] = fund_series.values
            out_path = os.path.join(DATA_DIR, PROCESSED_CSV)
            df.to_csv(out_path, index=False)
            print(f"Saved (augmented): {out_path} ({len(df)} rows)")
            del df
            gc.collect()
            return
        print("data/ 에 Artworks_full.csv 또는 Artworks.csv(또는 processed_moma_data.csv) 를 넣어 주세요.", file=sys.stderr)
        sys.exit(1)

    df = load_artworks(artworks_path)
    print(f"Loaded Artworks: {df.shape[0]} rows, {df.shape[1]} columns")

    # YearAcquired
    if "DateAcquired" in df.columns:
        df["YearAcquired"] = pd.to_datetime(df["DateAcquired"], errors="coerce").dt.year
    else:
        df["YearAcquired"] = np.nan

    # AcquisitionType from CreditLine (paper priority order)
    if "CreditLine" in df.columns:
        df["AcquisitionType"] = df["CreditLine"].apply(categorize_acquisition)
    else:
        df["AcquisitionType"] = "Unknown/Other"

    # Gender_Grouped
    if "Gender" in df.columns:
        df["Gender_Grouped"] = df["Gender"].apply(gender_grouped)
    else:
        df["Gender_Grouped"] = "Other/Unknown"

    # GeographicOrigin from Nationality
    if "Nationality" in df.columns:
        df["GeographicOrigin"] = df["Nationality"].apply(geographic_origin)
    else:
        df["GeographicOrigin"] = np.nan

    # Artist birth year and age at acquisition
    if "BeginDate" in df.columns:
        df["ArtistBirthYear"] = df["BeginDate"].apply(extract_birth_year)
    else:
        df["ArtistBirthYear"] = np.nan
    df["ArtistAgeAtAcquisition"] = np.nan
    mask = df["YearAcquired"].notna() & df["ArtistBirthYear"].notna()
    df.loc[mask, "ArtistAgeAtAcquisition"] = (
        df.loc[mask, "YearAcquired"] - df.loc[mask, "ArtistBirthYear"]
    )

    # NamedFund: Purchase 행에만 적용 (object dtype으로 초기화해 문자열 저장, FutureWarning 방지)
    if "CreditLine" in df.columns:
        def first_fund(cl):
            funds = extract_named_funds(cl)
            return funds[0] if funds else np.nan
        purchase_mask = df["AcquisitionType"] == "Purchase"
        fund_series = df.loc[purchase_mask, "CreditLine"].apply(first_fund)
        df["NamedFund"] = np.empty(len(df), dtype=object)
        df.loc[purchase_mask, "NamedFund"] = fund_series.values
    else:
        df["NamedFund"] = None  # object dtype

    # Drop duplicates if any (paper Table 1 N)
    n_before = len(df)
    df = df.drop_duplicates()
    n_total = len(df)
    print(f"Rows after dedup: {n_total} (dropped {n_before - n_total})")

    # Summary (paper Section 5.2: all percentages from this output)
    print("\nAcquisitionType distribution (count and %):")
    at = df["AcquisitionType"].value_counts()
    for k, v in at.items():
        print(f"  {k}: {v:,} ({100*v/n_total:.1f}%)")
    n_gift_full = int(at.get("Gift", 0))
    n_purchase = int(at.get("Purchase", 0))
    n_unknown_other = int(at.get("Unknown/Other", 0))
    n_artist_gift = int(at.get("Artist Gift", 0))
    n_bequest = int(at.get("Bequest", 0))
    n_partial = int(at.get("Partial Gift/Purchase", 0))
    print(f"  Table 1 sample (Purchase vs Gift only): Purchase N={n_purchase:,}, Gift N={n_gift_full:,}")

    print("\nGender_Grouped distribution (%):")
    gg = df["Gender_Grouped"].value_counts()
    for k, v in gg.items():
        print(f"  {k}: {100*v/n_total:.2f}%")
    female_pct = round(100 * gg.get("Female", 0) / n_total, 2) if n_total else None
    male_pct = round(100 * gg.get("Male", 0) / n_total, 2) if n_total else None
    other_unknown_pct = round(100 * (gg.get("Other/Unknown", 0)) / n_total, 2) if n_total else None

    multi_artist_pct = None
    if "Gender" in df.columns and n_total:
        # Paper Section 5.3: multi-artist works = Gender "Group" (MoMA) or contains "group"
        g = df["Gender"].fillna("").astype(str).str.strip()
        multi = (g.str.lower().str.contains("group", na=False) | (g.str.lower() == "group")).sum()
        multi_artist_pct = round(100 * multi / n_total, 2)
        print(f"  Multi-artist (Gender Group or contains 'group'): {multi_artist_pct}%")

    print("\nGeographicOrigin (% of total):")
    geo = df["GeographicOrigin"].value_counts(dropna=False)
    for k, v in geo.items():
        label = "missing" if pd.isna(k) or str(k).strip() == "" else k
        print(f"  {label}: {100*v/n_total:.2f}%")

    print(f"\nYearAcquired: median {df['YearAcquired'].median():.0f}, range {df['YearAcquired'].min():.0f}-{df['YearAcquired'].max():.0f}")

    # Paper Section 5.2: top CreditLine patterns for Unknown/Other and their demographic distribution
    uo = df[df["AcquisitionType"] == "Unknown/Other"]
    if len(uo) > 0 and "CreditLine" in uo.columns:
        def pct_female(s):
            return 100 * (s == "Female").sum() / len(s) if len(s) else np.nan
        def pct_nw(s):
            return 100 * (s == "Non-Western").sum() / len(s) if len(s) else np.nan
        agg = uo.groupby("CreditLine", dropna=False).agg(
            N=("AcquisitionType", "count"),
            pct_Female=("Gender_Grouped", pct_female),
            pct_NonWestern=("GeographicOrigin", pct_nw),
        ).sort_values("N", ascending=False).head(30)
        print("\nTop CreditLine patterns for Unknown/Other (demographic distribution for sensitivity):")
        print(agg.to_string())
        out_uo = get_output_path("unknown_other_creditline_patterns.csv")
        os.makedirs(os.path.dirname(out_uo), exist_ok=True)
        agg.to_csv(out_uo)
        print(f"Saved: {out_uo}")

    # 전처리 결과는 data/ 에 저장. 원본에 ConstituentID 있으면 그대로 유지 (06에서 연도별 artists 수 등에 사용).
    out_path = os.path.join(DATA_DIR, PROCESSED_CSV)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")

    # Paper Section 5.2: "all CreditLines were successfully classified" — every row has one of the 6 AcquisitionType values
    all_classified = df["AcquisitionType"].notna().all() and (df["AcquisitionType"].astype(str).str.strip() != "").all()
    n_types = df["AcquisitionType"].nunique()

    # Run metadata for verification script (paper Section 5.2 vs Table 1; Section 5.3 demographic rates)
    run_metadata = {
        "n_total": int(n_total),
        "n_gift_full": n_gift_full,
        "n_purchase": n_purchase,
        "n_gift_table1_sample": n_gift_full,
        "n_unknown_other": n_unknown_other,
        "n_artist_gift": n_artist_gift,
        "n_bequest": n_bequest,
        "n_partial_gift_purchase": n_partial,
        "all_creditlines_classified": bool(all_classified),
        "n_acquisition_types": int(n_types),
        "run_time": datetime.now().isoformat(),
    }
    if female_pct is not None:
        run_metadata["female_pct"] = female_pct
        run_metadata["male_pct"] = male_pct
        run_metadata["other_unknown_pct"] = other_unknown_pct
    if multi_artist_pct is not None:
        run_metadata["multi_artist_pct"] = multi_artist_pct
    meta_path = get_output_path("run_metadata.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Saved: {meta_path}")

    gc.collect()


if __name__ == "__main__":
    main()

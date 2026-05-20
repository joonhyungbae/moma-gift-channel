#!/usr/bin/env python3
"""
Phase R strong-accept analyses.

This script extends the donor-biography pipeline from six selected cases to
the full normalized MoMA donor population. It writes population-level
decorrelation tests, a funnel figure, and out-of-sample catalogue validation
artifacts to output/.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAIN_DEPARTMENTS, WESTERN_KEYWORDS, get_output_path, get_processed_path
from donor_bio_utils import (
    CASE_ORDER,
    CASE_SHORT,
    GIFT_TYPES,
    canonicalize_donors,
    case_for_donor,
    compact_space,
    detect_suberas,
    ensure_output_dir,
    wilson_ci_pct,
)
from experiment_utils import append_log
from utils import extract_birth_year, extract_donor_from_gift


BASELINE_FEMALE = 16.21
BASELINE_NONWEST = 15.59
MIN_YEAR = 1929
MAX_YEAR = 2024
RANDOM_STATE = 42

OUTPUT_DIR = Path(get_output_path(""))
PROCESSED_PATH = Path(get_processed_path())

GRAY = ["#111111", "#3b3b3b", "#666666", "#8a8a8a", "#b0b0b0", "#d0d0d0"]

CASE_LABEL_DONORS = {
    "Kleiner": "Kleiner",
    "Rockefeller lineage": "Abby Aldrich Rockefeller",
    "Jean Pigozzi": "Jean Pigozzi",
    "Peter J. Cohen": "Peter J. Cohen",
    "Judith Rothschild Foundation": "Judith Rothschild Foundation",
    "Agnes Gund": "Agnes Gund",
}

EASTERN_EUROPE_ALT_WESTERN = {
    "albanian",
    "belarusian",
    "bosnian",
    "bulgarian",
    "croatian",
    "czech",
    "czechoslovak",
    "estonian",
    "georgian",
    "hungarian",
    "latvian",
    "lithuanian",
    "macedonian",
    "moldovan",
    "montenegrin",
    "polish",
    "romanian",
    "russian",
    "serbian",
    "slovak",
    "slovenian",
    "soviet",
    "ukrainian",
    "yugoslav",
}

INSTITUTION_RE = re.compile(
    r"\b("
    r"foundation|fund|trust|estate of|inc\.?|corporation|committee|council|"
    r"endowment|charitable|collection|company|ltd\.?|limited|studio|"
    r"gallery|galleries|society|press|museum|association"
    r")\b",
    re.IGNORECASE,
)

ANONYMOUS_ARTIST_RE = re.compile(
    r"\b(unidentified|unknown|anonymous|various artists?|artist unknown|"
    r"unrecorded|not assigned|maker unknown)\b",
    re.IGNORECASE,
)

TYPE_NAMES = [
    "Volume contributor",
    "Founding lineage",
    "Single-artist patron",
    "Anonymous-archive donor",
    "Institutional mass-gift",
    "Cross-departmental diversifying",
]


def safe_pct(numer: float, denom: float) -> float:
    if denom is None or denom == 0 or pd.isna(denom):
        return np.nan
    return 100.0 * float(numer) / float(denom)


def normalize_artist_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = compact_space(str(value))
    return text if text else ""


def alt_geo_origin(nationality: object) -> object:
    """Alternative coding that treats Eastern-European nationalities as Western."""
    if pd.isna(nationality):
        return np.nan
    text = str(nationality).lower()
    text = re.sub(r"[()\[\],;/]", " ", text)
    text = compact_space(text)
    if pd.isna(text) or not text or text in {"nationality unknown", "unknown"}:
        return np.nan
    if any(keyword.lower() in text for keyword in WESTERN_KEYWORDS):
        return "Western"
    if any(keyword in text for keyword in EASTERN_EUROPE_ALT_WESTERN):
        return "Western"
    return "Non-Western"


def load_canonical_moma_donors() -> pd.DataFrame:
    """Load the canonical donor-identified gift sample with nationality retained."""
    usecols = [
        "ObjectID",
        "Title",
        "Artist",
        "ConstituentID",
        "Nationality",
        "Gender_Grouped",
        "GeographicOrigin",
        "ArtistBirthYear",
        "BeginDate",
        "YearAcquired",
        "AcquisitionType",
        "Department",
        "CreditLine",
        "Classification",
        "Medium",
    ]
    frames = []
    for chunk in pd.read_csv(
        PROCESSED_PATH,
        usecols=lambda col: col in usecols,
        chunksize=25_000,
        engine="python",
        on_bad_lines="skip",
    ):
        frame = chunk.copy()
        frame["YearAcquired"] = pd.to_numeric(frame["YearAcquired"], errors="coerce")
        frame = frame[
            frame["YearAcquired"].between(MIN_YEAR, MAX_YEAR)
            & frame["Department"].isin(MAIN_DEPARTMENTS)
            & frame["AcquisitionType"].isin(GIFT_TYPES)
        ].copy()
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        raise RuntimeError("No donor-eligible MoMA records loaded.")

    df = pd.concat(frames, ignore_index=True)
    df["YearAcquired"] = df["YearAcquired"].astype(int)

    if "ArtistBirthYear" not in df.columns:
        df["ArtistBirthYear"] = np.nan
    missing_birth = df["ArtistBirthYear"].isna()
    if "BeginDate" in df.columns and missing_birth.any():
        df.loc[missing_birth, "ArtistBirthYear"] = df.loc[missing_birth, "BeginDate"].apply(
            extract_birth_year
        )
    df["ArtistBirthYear"] = pd.to_numeric(df["ArtistBirthYear"], errors="coerce")

    df["DonorRaw"] = df["CreditLine"].apply(extract_donor_from_gift)
    df = df[df["DonorRaw"].notna()].copy()
    df["Donor"] = canonicalize_donors(df["DonorRaw"])
    df = df[df["Donor"].notna()].copy()
    df["Donor"] = df["Donor"].map(compact_space)
    df["CaseDonor"] = df["Donor"].apply(case_for_donor)
    df["ArtistClean"] = df["Artist"].apply(normalize_artist_name)
    df["GeoAlt"] = df["Nationality"].apply(alt_geo_origin)
    return df.reset_index(drop=True)


def gender_counts(frame: pd.DataFrame) -> tuple[int, int]:
    valid = frame["Gender_Grouped"].isin(["Female", "Male"])
    denom = int(valid.sum())
    numer = int((frame.loc[valid, "Gender_Grouped"] == "Female").sum())
    return numer, denom


def nonwest_counts(frame: pd.DataFrame, column: str = "GeographicOrigin") -> tuple[int, int]:
    valid = frame[column].notna()
    denom = int(valid.sum())
    numer = int((frame.loc[valid, column] == "Non-Western").sum())
    return numer, denom


def has_subera_regime_change(frame: pd.DataFrame) -> bool:
    year_counts = frame["YearAcquired"].value_counts().sort_index()
    suberas = detect_suberas(year_counts)
    if len(suberas) < 2:
        return False

    frame = frame.copy()
    for start, end, _breakpoints in suberas[1:]:
        current = frame[frame["YearAcquired"].between(start, end)]
        prior = frame[frame["YearAcquired"] < start]
        if current.empty or prior.empty:
            continue

        current_f, current_f_n = gender_counts(current)
        prior_f, prior_f_n = gender_counts(prior)
        if current_f_n > 0 and prior_f_n > 0:
            low, high = wilson_ci_pct(current_f, current_f_n)
            prior_pct = safe_pct(prior_f, prior_f_n)
            if not pd.isna(prior_pct) and (prior_pct < low or prior_pct > high):
                return True

        current_nw, current_nw_n = nonwest_counts(current)
        prior_nw, prior_nw_n = nonwest_counts(prior)
        if current_nw_n > 0 and prior_nw_n > 0:
            low, high = wilson_ci_pct(current_nw, current_nw_n)
            prior_pct = safe_pct(prior_nw, prior_nw_n)
            if not pd.isna(prior_pct) and (prior_pct < low or prior_pct > high):
                return True

    return False


def build_donor_population(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    alt_valid = df["GeoAlt"].notna()
    alt_baseline = safe_pct((df.loc[alt_valid, "GeoAlt"] == "Non-Western").sum(), alt_valid.sum())

    rows = []
    for donor, group in df.groupby("Donor", sort=False):
        n_gifts = int(len(group))
        years = group["YearAcquired"].dropna().astype(int)
        era_start = int(years.min()) if not years.empty else np.nan
        era_end = int(years.max()) if not years.empty else np.nan
        active_era_years = int(era_end - era_start) if not pd.isna(era_start) else np.nan

        year_counts = group["YearAcquired"].value_counts()
        peak_year = int(year_counts.idxmax()) if not year_counts.empty else np.nan
        peak_year_n = int(year_counts.max()) if not year_counts.empty else 0
        peak_year_share = safe_pct(peak_year_n, n_gifts)

        dept_counts = group["Department"].value_counts()
        top_dept = str(dept_counts.idxmax()) if not dept_counts.empty else ""
        top_dept_n = int(dept_counts.max()) if not dept_counts.empty else 0
        top_dept_share = safe_pct(top_dept_n, n_gifts)
        dept_shares = {
            f"share_{dept.replace(' ', '_').replace('&', 'and')}": safe_pct(
                int(dept_counts.get(dept, 0)), n_gifts
            )
            for dept in MAIN_DEPARTMENTS
        }
        dept_herfindahl = float(
            sum((int(dept_counts.get(dept, 0)) / n_gifts) ** 2 for dept in MAIN_DEPARTMENTS)
        )

        artist_counts = group["ArtistClean"].replace("", np.nan).dropna().value_counts()
        top_artist = str(artist_counts.idxmax()) if not artist_counts.empty else ""
        top_artist_n = int(artist_counts.max()) if not artist_counts.empty else 0
        top_artist_share = safe_pct(top_artist_n, n_gifts)

        female_n, valid_female = gender_counts(group)
        nonwest_n, valid_nonwest = nonwest_counts(group, "GeographicOrigin")
        nonwest_alt_n, valid_nonwest_alt = nonwest_counts(group, "GeoAlt")
        pct_f = safe_pct(female_n, valid_female)
        pct_nw = safe_pct(nonwest_n, valid_nonwest)
        pct_nw_alt = safe_pct(nonwest_alt_n, valid_nonwest_alt)

        female_ci_low, female_ci_high = (
            wilson_ci_pct(female_n, valid_female) if valid_female > 0 else (np.nan, np.nan)
        )
        nonwest_ci_low, nonwest_ci_high = (
            wilson_ci_pct(nonwest_n, valid_nonwest) if valid_nonwest > 0 else (np.nan, np.nan)
        )

        case_values = [case for case in group["CaseDonor"].dropna().unique()]
        case_donor = case_values[0] if case_values else ""

        row = {
            "donor": donor,
            "n_gifts": n_gifts,
            "active_era_start": era_start,
            "active_era_end": era_end,
            "active_era_years": active_era_years,
            "peak_year": peak_year,
            "peak_year_n": peak_year_n,
            "peak_year_share": peak_year_share,
            "top_dept": top_dept,
            "top_dept_n": top_dept_n,
            "top_dept_share": top_dept_share,
            "dept_herfindahl": dept_herfindahl,
            "top_artist": top_artist,
            "top_artist_n": top_artist_n,
            "top_artist_share": top_artist_share,
            "pct_female": pct_f,
            "female_gap": abs(pct_f - BASELINE_FEMALE) if not pd.isna(pct_f) else np.nan,
            "pct_nonwest": pct_nw,
            "nonwest_gap": abs(pct_nw - BASELINE_NONWEST) if not pd.isna(pct_nw) else np.nan,
            "pct_nonwest_alt": pct_nw_alt,
            "nonwest_gap_alt": (
                abs(pct_nw_alt - alt_baseline)
                if not pd.isna(pct_nw_alt) and not pd.isna(alt_baseline)
                else np.nan
            ),
            "valid_n_female": valid_female,
            "valid_n_nonwest": valid_nonwest,
            "valid_n_nonwest_alt": valid_nonwest_alt,
            "female_n": female_n,
            "nonwest_n": nonwest_n,
            "nonwest_alt_n": nonwest_alt_n,
            "female_ci_low": female_ci_low,
            "female_ci_high": female_ci_high,
            "nonwest_ci_low": nonwest_ci_low,
            "nonwest_ci_high": nonwest_ci_high,
            "is_institutional": bool(INSTITUTION_RE.search(donor)),
            "is_selected_case": bool(case_donor),
            "case_donor": case_donor,
            "subera_regime_change": has_subera_regime_change(group) if n_gifts >= 100 else False,
        }
        row.update(dept_shares)
        rows.append(row)

    donors = pd.DataFrame(rows).sort_values(["n_gifts", "donor"], ascending=[False, True])
    return donors.reset_index(drop=True), alt_baseline


def weighted_gap_model(frame: pd.DataFrame, gap_col: str, pct_col: str, n_col: str) -> dict:
    model_df = frame[["n_gifts", gap_col, pct_col, n_col]].dropna().copy()
    model_df = model_df[(model_df["n_gifts"] > 0) & (model_df[n_col] > 0)]
    if len(model_df) < 3:
        return {
            "ols_beta": np.nan,
            "ols_ci_low": np.nan,
            "ols_ci_high": np.nan,
            "r2": np.nan,
        }

    p = (model_df[pct_col] / 100.0).clip(0.001, 0.999)
    variance = (100.0**2) * p * (1.0 - p) / model_df[n_col].astype(float)
    weights = 1.0 / variance.replace(0, np.nan)
    model_df = model_df[weights.notna()].copy()
    weights = weights.loc[model_df.index]
    if len(model_df) < 3:
        return {
            "ols_beta": np.nan,
            "ols_ci_low": np.nan,
            "ols_ci_high": np.nan,
            "r2": np.nan,
        }

    x = sm.add_constant(np.log10(model_df["n_gifts"].astype(float)))
    model = sm.WLS(model_df[gap_col].astype(float), x, weights=weights.astype(float)).fit()
    beta = float(model.params.iloc[1])
    ci_low, ci_high = [float(v) for v in model.conf_int().iloc[1]]
    return {
        "ols_beta": beta,
        "ols_ci_low": ci_low,
        "ols_ci_high": ci_high,
        "r2": float(model.rsquared),
    }


def run_decorrelation(donors: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("female", "female_gap", "pct_female", "valid_n_female"),
        ("nonwest", "nonwest_gap", "pct_nonwest", "valid_n_nonwest"),
        ("nonwest_alt_eastern_europe_western", "nonwest_gap_alt", "pct_nonwest_alt", "valid_n_nonwest_alt"),
    ]
    rows = []
    for threshold in [50, 100, 200]:
        for dimension, gap_col, pct_col, n_col in specs:
            subset = donors[(donors["n_gifts"] >= threshold) & donors[gap_col].notna()].copy()
            if len(subset) >= 3:
                rho, p_value = spearmanr(subset["n_gifts"], subset[gap_col])
                ols = weighted_gap_model(subset, gap_col, pct_col, n_col)
            else:
                rho, p_value = np.nan, np.nan
                ols = {
                    "ols_beta": np.nan,
                    "ols_ci_low": np.nan,
                    "ols_ci_high": np.nan,
                    "r2": np.nan,
                }
            rows.append(
                {
                    "dimension": dimension,
                    "sample": f"N>={threshold}",
                    "n_donors": int(len(subset)),
                    "spearman_rho": rho,
                    "spearman_p": p_value,
                    **ols,
                }
            )
    return pd.DataFrame(rows)


def fit_line_for_plot(frame: pd.DataFrame, gap_col: str, pct_col: str, n_col: str):
    plot_df = frame[(frame["n_gifts"] >= 100) & frame[gap_col].notna()].copy()
    plot_df = plot_df[plot_df[n_col] > 0]
    if len(plot_df) < 3:
        return None
    p = (plot_df[pct_col] / 100.0).clip(0.001, 0.999)
    variance = (100.0**2) * p * (1.0 - p) / plot_df[n_col].astype(float)
    weights = 1.0 / variance.replace(0, np.nan)
    plot_df = plot_df[weights.notna()].copy()
    weights = weights.loc[plot_df.index]
    x = np.log10(plot_df["n_gifts"].astype(float))
    design = sm.add_constant(x)
    model = sm.WLS(plot_df[gap_col].astype(float), design, weights=weights.astype(float)).fit()
    grid_x = np.linspace(x.min(), x.max(), 100)
    pred_design = sm.add_constant(grid_x)
    pred = model.get_prediction(pred_design).summary_frame(alpha=0.05)
    return grid_x, pred


def write_volume_gap_figure(donors: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharex=True)
    panels = [
        ("female_gap", "pct_female", "valid_n_female", "Female gap (percentage points)"),
        ("nonwest_gap", "pct_nonwest", "valid_n_nonwest", "Non-Western gap (percentage points)"),
    ]

    for ax, (gap_col, pct_col, n_col, ylabel) in zip(axes, panels):
        all_valid = donors[donors[gap_col].notna()].copy()
        x_all = np.log10(all_valid["n_gifts"].astype(float))
        small = all_valid["n_gifts"] < 100
        large = ~small

        ax.scatter(
            x_all[small],
            all_valid.loc[small, gap_col],
            s=14,
            c=GRAY[4],
            edgecolors="none",
            label="N < 100",
            zorder=2,
        )
        ax.scatter(
            x_all[large],
            all_valid.loc[large, gap_col],
            s=22,
            facecolors="white",
            edgecolors=GRAY[0],
            linewidths=0.7,
            label="N >= 100",
            zorder=2,
        )

        line = fit_line_for_plot(donors, gap_col, pct_col, n_col)
        if line is not None:
            grid_x, pred = line
            ax.fill_between(
                grid_x,
                pred["mean_ci_lower"].astype(float).to_numpy(),
                pred["mean_ci_upper"].astype(float).to_numpy(),
                color="#d9d9d9",
                alpha=1.0,
                linewidth=0,
                zorder=1,
            )
            ax.plot(grid_x, pred["mean"], color=GRAY[0], linewidth=1.5, label="Weighted OLS", zorder=3)

        for case, donor in CASE_LABEL_DONORS.items():
            row = all_valid[all_valid["donor"] == donor]
            if row.empty:
                continue
            item = row.iloc[0]
            ax.annotate(
                CASE_SHORT.get(case, case),
                xy=(math.log10(float(item["n_gifts"])), float(item[gap_col])),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7.5,
                color=GRAY[0],
            )

        ax.set_ylabel(ylabel)
        ax.set_xlabel("log10(gift count)")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUTPUT_DIR / "fig_volume_vs_gap.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_volume_vs_gap.eps", bbox_inches="tight")
    plt.close(fig)


def baseline_inside_ci(low: float, high: float, baseline: float) -> bool:
    if pd.isna(low) or pd.isna(high):
        return False
    return low <= baseline <= high


def apply_type_rules(row: pd.Series) -> list[str]:
    matched = []
    if (
        row["top_dept_share"] >= 90.0
        and baseline_inside_ci(row["female_ci_low"], row["female_ci_high"], BASELINE_FEMALE)
        and baseline_inside_ci(row["nonwest_ci_low"], row["nonwest_ci_high"], BASELINE_NONWEST)
    ):
        matched.append("Volume contributor")

    if (
        row["active_era_years"] >= 60
        and row["pct_female"] < BASELINE_FEMALE
        and row["pct_nonwest"] < BASELINE_NONWEST
    ):
        matched.append("Founding lineage")

    if row["top_artist_share"] > 90.0:
        matched.append("Single-artist patron")

    if (
        bool(ANONYMOUS_ARTIST_RE.search(str(row["top_artist"])))
        and row["top_dept_share"] >= 90.0
    ):
        matched.append("Anonymous-archive donor")

    if bool(row["is_institutional"]) and row["peak_year_share"] > 90.0:
        matched.append("Institutional mass-gift")

    if row["top_dept_share"] <= 60.0 and bool(row["subera_regime_change"]):
        matched.append("Cross-departmental diversifying")

    return matched


def run_catalogue_validation(donors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eligible = donors[(donors["n_gifts"] >= 100) & (~donors["is_selected_case"])].copy()
    validation_rows = []
    for _, row in eligible.iterrows():
        matched = apply_type_rules(row)
        n_matches = len(matched)
        if n_matches == 0:
            classification = "unclassifiable"
        elif n_matches == 1:
            classification = "clean"
        else:
            classification = "ambiguous"

        validation_rows.append(
            {
                "donor": row["donor"],
                "n_gifts": int(row["n_gifts"]),
                "matched_types": "; ".join(matched),
                "n_matches": n_matches,
                "classification": classification,
                "top_dept": row["top_dept"],
                "top_dept_share": row["top_dept_share"],
                "top_artist": row["top_artist"],
                "top_artist_share": row["top_artist_share"],
                "peak_year_share": row["peak_year_share"],
                "active_era_years": row["active_era_years"],
                "is_institutional": row["is_institutional"],
                "female_gap": row["female_gap"],
                "nonwest_gap": row["nonwest_gap"],
                "subera_regime_change": row["subera_regime_change"],
            }
        )

    validation = pd.DataFrame(validation_rows).sort_values(["n_gifts", "donor"], ascending=[False, True])

    summary_rows = []
    class_counts = validation["classification"].value_counts()
    for key in ["clean", "ambiguous", "unclassifiable"]:
        summary_rows.append(
            {
                "section": "classification_count",
                "metric": key,
                "value": int(class_counts.get(key, 0)),
                "notes": "",
            }
        )

    for type_name in TYPE_NAMES:
        count = int(validation["matched_types"].str.contains(re.escape(type_name), na=False).sum())
        summary_rows.append(
            {
                "section": "type_count",
                "metric": type_name,
                "value": count,
                "notes": "",
            }
        )

    feature_cols = [
        "top_dept_share",
        "top_artist_share",
        "peak_year_share",
        "active_era_years",
        "is_institutional",
        "female_gap",
        "nonwest_gap",
        "subera_regime_change",
    ]
    feature_df = validation[feature_cols].copy()
    for col in ["is_institutional", "subera_regime_change"]:
        feature_df[col] = feature_df[col].astype(int)
    for col in feature_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        if feature_df[col].isna().any():
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    cluster_info = {
        "kmeans_optimal_k": np.nan,
        "kmeans_optimal_silhouette": np.nan,
        "hierarchical_optimal_k": np.nan,
        "hierarchical_optimal_silhouette": np.nan,
        "near_six": False,
        "centroid_alignment": "not evaluated",
    }

    if len(feature_df) >= 4:
        x_scaled = StandardScaler().fit_transform(feature_df)
        kmeans_scores = {}
        hierarchical_scores = {}
        max_k = min(10, len(feature_df) - 1)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
            km_labels = km.fit_predict(x_scaled)
            km_score = float(silhouette_score(x_scaled, km_labels))
            kmeans_scores[k] = km_score
            summary_rows.append(
                {
                    "section": "kmeans_silhouette",
                    "metric": str(k),
                    "value": km_score,
                    "notes": "",
                }
            )

            agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
            agg_labels = agg.fit_predict(x_scaled)
            agg_score = float(silhouette_score(x_scaled, agg_labels))
            hierarchical_scores[k] = agg_score
            summary_rows.append(
                {
                    "section": "hierarchical_silhouette",
                    "metric": str(k),
                    "value": agg_score,
                    "notes": "",
                }
            )

        optimal_k = max(kmeans_scores, key=kmeans_scores.get)
        optimal_h = max(hierarchical_scores, key=hierarchical_scores.get)
        near_six = abs(optimal_k - 6) <= 1
        centroid_alignment = "partial" if near_six else "no"
        if near_six:
            centroid_alignment = "partial: silhouette optimum is near six; inspect centroids before treating labels as partitions"
        else:
            centroid_alignment = "no: silhouette optimum is not near six"

        cluster_info = {
            "kmeans_optimal_k": int(optimal_k),
            "kmeans_optimal_silhouette": float(kmeans_scores[optimal_k]),
            "hierarchical_optimal_k": int(optimal_h),
            "hierarchical_optimal_silhouette": float(hierarchical_scores[optimal_h]),
            "near_six": bool(near_six),
            "centroid_alignment": centroid_alignment,
        }

        summary_rows.append(
            {
                "section": "clustering_summary",
                "metric": "kmeans_optimal_k",
                "value": int(optimal_k),
                "notes": f"silhouette={kmeans_scores[optimal_k]:.3f}",
            }
        )
        summary_rows.append(
            {
                "section": "clustering_summary",
                "metric": "hierarchical_optimal_k",
                "value": int(optimal_h),
                "notes": f"silhouette={hierarchical_scores[optimal_h]:.3f}",
            }
        )
        summary_rows.append(
            {
                "section": "clustering_summary",
                "metric": "centroid_alignment",
                "value": "",
                "notes": centroid_alignment,
            }
        )

    summary = pd.DataFrame(summary_rows)
    return validation.reset_index(drop=True), summary, cluster_info


def decorrelation_verdict(rho: float, p_value: float, r2: float | None = None) -> str:
    if pd.isna(rho):
        return "no estimate"
    if abs(rho) < 0.2 and (pd.isna(p_value) or p_value >= 0.05):
        return "yes"
    return "no"


def remove_existing_log_section(title: str) -> None:
    log_path = OUTPUT_DIR / "codex_experiments_log.md"
    if not log_path.exists():
        return
    text = log_path.read_text(encoding="utf-8")
    pattern = rf"## {re.escape(title)}\n\n.*?(?=\n## |\Z)"
    cleaned = re.sub(pattern, "", text, flags=re.S).rstrip() + "\n\n"
    log_path.write_text(cleaned, encoding="utf-8")


def append_phase_log(
    decorrelation: pd.DataFrame,
    validation: pd.DataFrame,
    cluster_info: dict,
    alt_baseline: float,
) -> None:
    primary = decorrelation[decorrelation["sample"] == "N>=100"].set_index("dimension")
    female = primary.loc["female"]
    nonwest = primary.loc["nonwest"]
    nonwest_alt = primary.loc["nonwest_alt_eastern_europe_western"]

    class_counts = validation["classification"].value_counts()
    clean = int(class_counts.get("clean", 0))
    ambiguous = int(class_counts.get("ambiguous", 0))
    unclassifiable = int(class_counts.get("unclassifiable", 0))

    bullets = [
        "Population-level decorrelation uses all normalized donor strings, with N>=100 as the primary sample to avoid the small-N funnel artifact.",
        f"Female gap, N>=100: Spearman rho={female['spearman_rho']:.3f}, p={female['spearman_p']:.3g}; weighted-OLS beta={female['ols_beta']:.3f} [{female['ols_ci_low']:.3f}, {female['ols_ci_high']:.3f}], R2={female['r2']:.3f}.",
        f"Non-Western gap, N>=100: Spearman rho={nonwest['spearman_rho']:.3f}, p={nonwest['spearman_p']:.3g}; weighted-OLS beta={nonwest['ols_beta']:.3f} [{nonwest['ols_ci_low']:.3f}, {nonwest['ols_ci_high']:.3f}], R2={nonwest['r2']:.3f}.",
        f"Alternative Eastern-European-as-Western baseline={alt_baseline:.2f}%; N>=100 non-Western sensitivity rho={nonwest_alt['spearman_rho']:.3f}, p={nonwest_alt['spearman_p']:.3g}.",
        "Small-N note: the funnel shape among tiny donors is a sampling artifact because low gift counts can only realize coarse demographic shares; it is not the evidentiary sample for the paper's headline.",
        f"Catalogue OOS validation: clean={clean}, ambiguous={ambiguous}, unclassifiable={unclassifiable}; k-means optimal k={cluster_info['kmeans_optimal_k']} ({cluster_info['centroid_alignment']}).",
    ]
    title = "Experiment 41: strong-accept population analyses"
    remove_existing_log_section(title)
    append_log(title, bullets)


def write_outputs_and_print_summary(
    donors: pd.DataFrame,
    decorrelation: pd.DataFrame,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    cluster_info: dict,
    alt_baseline: float,
) -> None:
    donor_cols = [
        "donor",
        "n_gifts",
        "pct_female",
        "female_gap",
        "pct_nonwest",
        "nonwest_gap",
        "pct_nonwest_alt",
        "nonwest_gap_alt",
        "valid_n_female",
        "valid_n_nonwest",
        "valid_n_nonwest_alt",
        "active_era_start",
        "active_era_end",
        "active_era_years",
        "top_dept",
        "top_dept_share",
        "top_artist",
        "top_artist_share",
        "peak_year_share",
        "is_institutional",
        "subera_regime_change",
        "is_selected_case",
        "case_donor",
    ]
    donors[donor_cols].to_csv(OUTPUT_DIR / "donor_population_gaps.csv", index=False)
    decorrelation.to_csv(OUTPUT_DIR / "population_decorrelation.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "catalogue_oos_validation.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "catalogue_oos_summary.csv", index=False)
    write_volume_gap_figure(donors)
    append_phase_log(decorrelation, validation, cluster_info, alt_baseline)

    primary = decorrelation[decorrelation["sample"] == "N>=100"].set_index("dimension")
    female = primary.loc["female"]
    nonwest = primary.loc["nonwest"]
    female_verdict = decorrelation_verdict(female["spearman_rho"], female["spearman_p"])
    nonwest_verdict = decorrelation_verdict(nonwest["spearman_rho"], nonwest["spearman_p"])
    female_ci_spans = female["ols_ci_low"] <= 0 <= female["ols_ci_high"]
    nonwest_ci_spans = nonwest["ols_ci_low"] <= 0 <= nonwest["ols_ci_high"]

    class_counts = validation["classification"].value_counts()
    clean = int(class_counts.get("clean", 0))
    unclassifiable = int(class_counts.get("unclassifiable", 0))
    total = int(len(validation))
    rank_support = female_verdict == "yes" and nonwest_verdict == "yes"
    support = (
        "Rank evidence supports the headline beyond the six cases; qualify that weighted female OLS remains negative."
        if rank_support and not female_ci_spans
        else "Population evidence supports the headline beyond the six cases."
        if rank_support
        else "Population evidence is mixed; update the headline language with the reported associations."
    )

    print(
        f"Female: Spearman rho={female['spearman_rho']:.3f}, p={female['spearman_p']:.3g} (N>=100); decorrelated={female_verdict}."
    )
    print(
        f"Non-Western: Spearman rho={nonwest['spearman_rho']:.3f}, p={nonwest['spearman_p']:.3g} (N>=100); decorrelated={nonwest_verdict}."
    )
    print(
        "OLS beta CIs: "
        f"female {female['ols_beta']:.3f} [{female['ols_ci_low']:.3f}, {female['ols_ci_high']:.3f}] "
        f"({'spans 0' if female_ci_spans else 'does not span 0'}); "
        f"non-Western {nonwest['ols_beta']:.3f} [{nonwest['ols_ci_low']:.3f}, {nonwest['ols_ci_high']:.3f}] "
        f"({'spans 0' if nonwest_ci_spans else 'does not span 0'})."
    )
    print(f"OOS catalogue: {clean} of {total} donors clean single-type; {unclassifiable} unclassifiable.")
    print(
        f"Clustering: k-means silhouette-optimal k={cluster_info['kmeans_optimal_k']} "
        f"({'near 6' if cluster_info['near_six'] else 'not near 6'})."
    )
    print(support)


def main() -> None:
    ensure_output_dir()
    df = load_canonical_moma_donors()
    donors, alt_baseline = build_donor_population(df)

    decorrelation = run_decorrelation(donors)
    validation, catalogue_summary, cluster_info = run_catalogue_validation(donors)
    write_outputs_and_print_summary(
        donors,
        decorrelation,
        validation,
        catalogue_summary,
        cluster_info,
        alt_baseline,
    )


if __name__ == "__main__":
    main()

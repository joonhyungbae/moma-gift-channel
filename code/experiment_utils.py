# -*- coding: utf-8 -*-
"""
Shared helpers for MMC revision experiments (20_*.py to 26_*.py).
"""

import os
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MAIN_DEPARTMENTS, get_processed_path, get_output_path
from utils import (
    extract_birth_year,
    extract_donor_from_gift,
    extract_named_funds,
    reclassify_unknown_other_enhanced,
)

CHUNK_SIZE = 25_000
TOP_MEDIUM = 12
TOP_CLASS = 10

BASE_COLS = [
    "AcquisitionType",
    "CreditLine",
    "Gender_Grouped",
    "GeographicOrigin",
    "YearAcquired",
    "ArtistBirthYear",
    "BeginDate",
    "Department",
    "Medium",
    "Classification",
    "ConstituentID",
    "NamedFund",
]

AMBIGUOUS_PATTERNS = [
    "fund",
    "collection",
    "anonymously",
    "given anonymously",
    "acquired through",
    "promised gift",
    "partial",
    "fractional",
    "transfer",
]


def ensure_output_dir():
    Path(get_output_path("")).mkdir(parents=True, exist_ok=True)


def parse_first_constituent(cid):
    if pd.isna(cid):
        return np.nan
    s = str(cid).strip()
    if not s:
        return np.nan
    return s.split(",")[0].strip()


def cluster_codes(series):
    return series.fillna("__MISSING__").astype(str).astype("category").cat.codes.to_numpy()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_base_data(extra_cols=None, require_birthyear=True):
    path = get_processed_path()
    avail = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = list(dict.fromkeys(BASE_COLS + (extra_cols or [])))
    usecols = [c for c in wanted if c in avail]
    chunks = []
    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        engine="python",
        on_bad_lines="skip",
    )
    for chunk in reader:
        chunk["YearAcquired"] = pd.to_numeric(chunk["YearAcquired"], errors="coerce")
        chunk["ArtistBirthYear"] = pd.to_numeric(chunk["ArtistBirthYear"], errors="coerce")
        if "BeginDate" in chunk.columns:
            miss = chunk["ArtistBirthYear"].isna()
            if miss.any():
                chunk.loc[miss, "ArtistBirthYear"] = chunk.loc[miss, "BeginDate"].apply(extract_birth_year)
        chunk = chunk[
            chunk["YearAcquired"].notna()
            & (chunk["YearAcquired"] >= 1930)
            & (chunk["YearAcquired"] <= 2024)
            & chunk["Department"].isin(MAIN_DEPARTMENTS)
        ].copy()
        if require_birthyear:
            chunk = chunk[chunk["ArtistBirthYear"].notna()].copy()
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(chunks, ignore_index=True)
    return df


def add_common_fields(df):
    df = df.copy()
    df["AcqType_enhanced"] = df["AcquisitionType"].copy()
    uo = df["AcquisitionType"] == "Unknown/Other"
    if uo.any():
        df.loc[uo, "AcqType_enhanced"] = df.loc[uo, "CreditLine"].apply(reclassify_unknown_other_enhanced)

    df["Purchase_enhanced"] = np.nan
    purchase_like = df["AcqType_enhanced"].isin(["Purchase", "Fund Purchase"])
    gift_like = df["AcqType_enhanced"].isin(["Gift", "Anonymous Gift", "Collection Gift"])
    df.loc[purchase_like, "Purchase_enhanced"] = 1
    df.loc[gift_like, "Purchase_enhanced"] = 0

    df["Decade"] = (df["YearAcquired"] // 10 * 10).astype(int)
    df["DeptDecade"] = df["Department"].astype(str) + "_" + df["Decade"].astype(str)
    df["BirthYear_c"] = df["ArtistBirthYear"] - df["ArtistBirthYear"].median()
    if "ConstituentID" in df.columns:
        df["ArtistID"] = df["ConstituentID"].apply(parse_first_constituent)
    else:
        df["ArtistID"] = np.nan

    if "Medium" in df.columns and df["Medium"].notna().any():
        top_med = df["Medium"].fillna("Missing").value_counts().head(TOP_MEDIUM).index.tolist()
        df["Medium_grp"] = (
            df["Medium"].fillna("Missing").where(df["Medium"].fillna("Missing").isin(top_med), "Other").astype(str)
        )
    else:
        df["Medium_grp"] = "Other"

    if "Classification" in df.columns and df["Classification"].notna().any():
        top_cl = df["Classification"].fillna("Missing").value_counts().head(TOP_CLASS).index.tolist()
        df["Classification_grp"] = (
            df["Classification"]
            .fillna("Missing")
            .where(df["Classification"].fillna("Missing").isin(top_cl), "Other")
            .astype(str)
        )
    else:
        df["Classification_grp"] = "Other"

    if "NamedFund" not in df.columns:
        df["NamedFund"] = np.nan

    def _primary_fund(row):
        if pd.notna(row.get("NamedFund", np.nan)) and str(row["NamedFund"]).strip():
            return str(row["NamedFund"]).strip()
        funds = extract_named_funds(row.get("CreditLine"))
        if funds:
            return funds[0]
        cl = str(row.get("CreditLine", "")).strip()
        return cl[:120] if cl else np.nan

    fund_mask = df["AcqType_enhanced"] == "Fund Purchase"
    if fund_mask.any():
        df.loc[fund_mask, "PrimaryFund"] = df.loc[fund_mask].apply(_primary_fund, axis=1)
    else:
        df["PrimaryFund"] = np.nan

    return df


def get_enhanced_main_sample():
    df = load_base_data()
    df = add_common_fields(df)
    df = df[df["Purchase_enhanced"].notna()].copy()
    df["Purchase"] = df["Purchase_enhanced"].astype(int)
    return df


def keep_varying(df_sub, outcome_col, group_col="DeptDecade"):
    keep = df_sub.groupby(group_col)[outcome_col].transform(
        lambda x: (x.nunique() >= 2) & (x.sum() >= 1) & ((1 - x).sum() >= 1)
    )
    return df_sub.loc[keep].copy()


def prepare_outcome_sample(df, outcome):
    d = df.copy()
    if outcome == "IsFemale":
        d = d[d["Gender_Grouped"].isin(["Female", "Male"])].copy()
        d["IsFemale"] = (d["Gender_Grouped"] == "Female").astype(int)
        d = d.dropna(subset=["ArtistBirthYear", "DeptDecade"])
        d = keep_varying(d, "IsFemale")
    elif outcome == "IsNonWest":
        d = d[d["GeographicOrigin"].notna()].copy()
        d["IsNonWest"] = (d["GeographicOrigin"] == "Non-Western").astype(int)
        d = d.dropna(subset=["ArtistBirthYear", "DeptDecade"])
        d = keep_varying(d, "IsNonWest")
    else:
        raise ValueError(f"Unsupported outcome: {outcome}")
    return d


def fit_glm_formula(df, formula, outcome_label, cluster_type="none"):
    y, X = dmatrices(formula, data=df, return_type="dataframe")
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200)
    cov = np.asarray(model.cov_params())
    if cluster_type == "artist":
        cov = cov_cluster(model, cluster_codes(df.loc[X.index, "ArtistID"]))
    elif cluster_type == "deptdecade":
        cov = cov_cluster(model, cluster_codes(df.loc[X.index, "DeptDecade"]))
    elif cluster_type == "twoway":
        cov, _, _ = cov_cluster_2groups(
            model,
            cluster_codes(df.loc[X.index, "ArtistID"]),
            cluster_codes(df.loc[X.index, "DeptDecade"]),
        )
    elif cluster_type == "none":
        pass
    else:
        raise ValueError(f"Unknown cluster_type: {cluster_type}")
    return model, X, cov


def extract_coef(model, X, cov, term):
    idx = X.columns.get_loc(term)
    beta = float(model.params.iloc[idx])
    se = float(np.sqrt(max(cov[idx, idx], 0.0)))
    ci_lo, ci_hi = beta - 1.96 * se, beta + 1.96 * se
    return {
        "beta": beta,
        "SE": se,
        "OR": float(np.exp(beta)),
        "CI_lower": float(np.exp(ci_lo)),
        "CI_upper": float(np.exp(ci_hi)),
        "p": float(model.pvalues.iloc[idx]),
    }


def fit_main_effect(df, outcome, extra_terms=None, cluster_types=("none",)):
    d = prepare_outcome_sample(df, outcome)
    rhs = ["Purchase", "BirthYear_c", "C(DeptDecade)", "C(Medium_grp)", "C(Classification_grp)"]
    if extra_terms:
        rhs = list(extra_terms) + [x for x in rhs if x not in extra_terms]
    formula = f"{outcome} ~ " + " + ".join(rhs)
    out = {"N": len(d), "Outcome": outcome, "PseudoR2": np.nan}
    for ctype in cluster_types:
        model, X, cov = fit_glm_formula(d, formula, outcome, cluster_type=ctype)
        stats = extract_coef(model, X, cov, "Purchase")
        prefix = "Wald" if ctype == "none" else ("ArtistCluster" if ctype == "artist" else "TwoWay")
        out[f"{prefix}_OR"] = stats["OR"]
        out[f"{prefix}_CI_lower"] = stats["CI_lower"]
        out[f"{prefix}_CI_upper"] = stats["CI_upper"]
        out[f"{prefix}_SE"] = stats["SE"]
        out[f"{prefix}_p"] = stats["p"]
        if ctype == "none":
            out["PseudoR2"] = model.pseudo_rsquared(kind="mcf") if hasattr(model, "pseudo_rsquared") else np.nan
    return out


def make_clarity_flag(df):
    df = df.copy()
    cl = df["CreditLine"].fillna("").astype(str).str.lower()
    ambiguous = pd.Series(False, index=df.index)
    for pat in AMBIGUOUS_PATTERNS:
        ambiguous = ambiguous | cl.str.contains(pat, regex=False, na=False)
    clear_types = df["AcquisitionType"].isin(["Purchase", "Gift", "Bequest", "Artist Gift"])
    df["RecordClarity"] = np.where(clear_types & ~ambiguous, "Clear", "Ambiguous")
    return df


def append_log(section_title, bullets, paper_paragraph=None):
    ensure_output_dir()
    log_path = Path(get_output_path("codex_experiments_log.md"))
    if not log_path.exists():
        log_path.write_text("# Codex Experiments Log\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"## {section_title}\n\n")
        if isinstance(bullets, str):
            f.write(bullets.strip() + "\n\n")
        else:
            for line in bullets:
                f.write(f"- {line}\n")
            f.write("\n")
        if paper_paragraph:
            f.write("**Paper-ready paragraph**\n\n")
            f.write(paper_paragraph.strip() + "\n\n")


def safe_pct(num, den):
    if den in (0, None) or pd.isna(den):
        return np.nan
    return 100.0 * num / den


def get_top_donors(gifts_df, top_k=25):
    g = gifts_df.copy()
    g["Donor"] = g["CreditLine"].apply(extract_donor_from_gift)
    g = g[g["Donor"].notna()].copy()
    vc = g["Donor"].value_counts().head(top_k)
    return vc, g


def booktabs_table(lines, caption, label, colspec, notes=None):
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
    ]
    tex.extend(lines)
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    if notes:
        tex.append(f"\\footnotesize{{{notes}}}")
    tex.append("\\end{table}")
    return "\n".join(tex)


def try_read_csv(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return None

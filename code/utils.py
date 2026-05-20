# Shared utilities for sn-article analysis

import re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from config import WESTERN_KEYWORDS


def categorize_acquisition(creditline):
    """
    Engineer AcquisitionType from CreditLine (paper Section 5.2).
    Priority: (1) Artist Gift, (2) Partial Gift/Purchase, (3) Purchase, (4) Bequest, (5) Gift, (6) Unknown/Other.
    Note: Purchase before Bequest so "Purchased with funds from the bequest of X" -> Purchase.
    """
    if pd.isna(creditline) or str(creditline).strip() == "":
        return "Unknown/Other"
    cl = str(creditline).lower()
    if "gift of the artist" in cl or "artist's gift" in cl or "artist gift" in cl:
        return "Artist Gift"
    if ("partial" in cl and ("gift" in cl or "purchase" in cl or "bequest" in cl)) or "gift and purchase" in cl or "purchase and gift" in cl:
        return "Partial Gift/Purchase"
    if "fractional" in cl or "promised" in cl:
        return "Unknown/Other"
    if "purchase" in cl:
        return "Purchase"
    if "bequest" in cl:
        return "Bequest"
    if "gift" in cl:
        return "Gift"
    return "Unknown/Other"


def reclassify_unknown_other_enhanced(creditline):
    """
    Enhanced CreditLine pattern matching for records currently classified as Unknown/Other.
    Applied as a second pass; does not change already-classified types.
    Order: Exchange/Transfer → Anonymous Gift → Collection Gift → Fund Purchase → Unknown/Other.
    - "Fund" (without "Gift" or "Purchase") → "Fund Purchase" (subcategory of Purchase)
    - "Exchange" or "Transfer" → "Institutional Transfer"
    - "anonymously" → "Anonymous Gift"
    - "Collection" (e.g. "The Louis E. Stern Collection") → "Collection Gift"
    """
    if pd.isna(creditline) or str(creditline).strip() == "":
        return "Unknown/Other"
    cl = str(creditline).strip().lower()
    if "exchange" in cl or "transfer" in cl:
        return "Institutional Transfer"
    if "anonymously" in cl:
        return "Anonymous Gift"
    if "collection" in cl:
        return "Collection Gift"
    # Fund without explicit Gift or Purchase → Fund Purchase (counts as Purchase in regressions)
    if "fund" in cl and "gift" not in cl and "purchase" not in cl:
        return "Fund Purchase"
    return "Unknown/Other"


def gender_grouped(gender):
    """Map Gender to Female / Male / Other/Unknown. data: '(male)', '(female)' 등 괄호 형식 지원."""
    if pd.isna(gender):
        return "Other/Unknown"
    g = str(gender).strip()
    if g.startswith("(") and g.endswith(")"):
        g = g[1:-1]
    g = g.lower()
    if "female" in g:
        return "Female"
    if "male" in g:
        return "Male"
    return "Other/Unknown"


def geographic_origin(nationality):
    """Western vs Non-Western from Nationality. data: '(American)', '(French)' 등 괄호 형식 지원."""
    if pd.isna(nationality) or str(nationality).strip() in ("", "()", "nan"):
        return np.nan
    s = str(nationality).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    nat = s.lower()
    for kw in WESTERN_KEYWORDS:
        if kw in nat:
            return "Western"
    return "Non-Western"


# Paper robustness (iv): 3 categories Western / Non-Western excl. East Asia / East Asia
EAST_ASIA_KEYWORDS = ["japanese", "chinese", "korean", "taiwanese"]


def geographic_origin_3(nationality):
    """Returns Western, NonWestern_exclEastAsia, or EastAsia. For robustness (iv)."""
    if pd.isna(nationality) or str(nationality).strip() in ("", "()", "nan"):
        return np.nan
    s = str(nationality).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    nat = s.lower()
    for kw in WESTERN_KEYWORDS:
        if kw in nat:
            return "Western"
    for kw in EAST_ASIA_KEYWORDS:
        if kw in nat:
            return "EastAsia"
    return "NonWestern_exclEastAsia"


def extract_birth_year(begin_date):
    """Parse BeginDate (e.g. '(1841)' or '1841') to integer year."""
    if pd.isna(begin_date):
        return np.nan
    s = str(begin_date).strip()
    m = re.search(r"\(?(\d{4})\)?", s)
    if m:
        y = int(m.group(1))
        if 1000 <= y <= 2030:
            return y
    return np.nan


def extract_death_year(end_date):
    """Parse EndDate (e.g. '(1957)' or '1957') to integer year. Returns nan for missing or '(0)' (living)."""
    if pd.isna(end_date):
        return np.nan
    s = str(end_date).strip()
    m = re.search(r"\(?(\d{4})\)?", s)
    if m:
        y = int(m.group(1))
        if 1000 <= y <= 2030 and y != 0:
            return y
    return np.nan


def ensure_analysis_columns(df):
    """
    Add ArtistBirthYear, ArtistAgeAtAcquisition to df in-place if missing (from BeginDate, YearAcquired).
    Also ensure YearAcquired is numeric. Returns df.
    """
    if "YearAcquired" in df.columns:
        df["YearAcquired"] = pd.to_numeric(df["YearAcquired"], errors="coerce")
    if "ArtistBirthYear" not in df.columns and "BeginDate" in df.columns:
        df["ArtistBirthYear"] = df["BeginDate"].apply(extract_birth_year)
    if "ArtistAgeAtAcquisition" not in df.columns:
        df["ArtistAgeAtAcquisition"] = np.nan
        if "YearAcquired" in df.columns and "ArtistBirthYear" in df.columns:
            m = df["YearAcquired"].notna() & df["ArtistBirthYear"].notna()
            df.loc[m, "ArtistAgeAtAcquisition"] = df.loc[m, "YearAcquired"] - df.loc[m, "ArtistBirthYear"]
    return df


def gini_coefficient(x):
    """Gini for a 1D array (e.g. counts by year or by donor). 0=equality, 1=max inequality."""
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        return np.nan
    sorted_x = np.sort(x)
    index = np.arange(1, n + 1, dtype=float)
    return (2 * np.sum(index * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n


def coefficient_of_variation(x):
    """CV = std / mean (sample std)."""
    x = np.asarray(x, dtype=float)
    m, s = np.mean(x), np.std(x, ddof=1)
    return np.nan if m == 0 else s / m


def cramers_v(contingency):
    """Cramér's V for a contingency table (effect size for chi-squared)."""
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1
    if min_dim <= 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def chi2_and_cramers(obs):
    """Return chi2, p, dof, Cramér's V for contingency table."""
    chi2, p, dof, expected = chi2_contingency(obs)
    v = cramers_v(obs)
    return chi2, p, dof, v


def extract_named_funds(creditline):
    """
    Extract named purchase fund(s) from CreditLine (paper: patterns like '[X] Purchase Fund', 'Purchased with funds from [X]').
    Returns list of fund name strings.
    """
    if pd.isna(creditline):
        return []
    line = str(creditline)
    funds = []
    # Pattern: "X Purchase Fund" or "X Purchase Funds"
    for m in re.finditer(r"([A-Z][A-Za-z0-9\s\.&'\-]+?)\s+Purchase\s+Funds?(?:\s|,|$|\.)", line):
        name = m.group(1).strip()
        name = re.sub(r"^(?:through|from|by|the|and)\s+", "", name, flags=re.IGNORECASE).strip()
        if name and len(name) > 2:
            funds.append(name + " Purchase Fund")
    if not funds:
        for m in re.finditer(r"Purchased?\s+(?:in part\s+)?with funds\s+(?:provided by|from)\s+(?:the\s+)?([A-Z][A-Za-z\s\.&'\-]+?)(?:\s+Fund)?", line, re.IGNORECASE):
            name = m.group(1).strip()
            if name and len(name) > 2:
                funds.append(name + " Fund")
    return list(dict.fromkeys(funds))  # unique, order preserved


def extract_donor_from_gift(creditline):
    """Extract donor name from Gift CreditLine (e.g. 'Gift of [Donor]').

    Returns NaN for credit lines that name a role rather than an individual
    (e.g., 'Gift of the architect', 'Gift of the designer'). Such strings
    aggregate many distinct people under one label and would inflate donor-
    concentration metrics if treated as a single donor.
    """
    if pd.isna(creditline):
        return np.nan
    try:
        from config import GENERIC_DONOR_ROLE_TERMS
    except ImportError:
        GENERIC_DONOR_ROLE_TERMS = {
            "architect", "architects", "designer", "designers", "manufacturer",
            "manufacturers", "artist", "artists", "photographer", "photographers",
            "publisher", "publishers", "estate", "family", "foundation",
            "company", "firm", "studio", "office", "above", "donor", "donors",
            "anonymous donor", "anonymous", "various donors",
        }
    cl = str(creditline).strip()
    cl_lower = cl.lower()
    for phrase in ("gift of the ", "bequest of the ", "gift of ", "bequest of "):
        idx = cl_lower.find(phrase)
        if idx >= 0:
            rest = cl[idx + len(phrase):].strip()
            for sep in (",", "\n"):
                if sep in rest:
                    rest = rest.split(sep)[0].strip()
            if not rest:
                return np.nan
            if rest.strip().lower() in GENERIC_DONOR_ROLE_TERMS:
                return np.nan
            return rest
    return np.nan


def shannon_entropy(proportions):
    """Shannon entropy (bits) for a probability vector. proportions should sum to 1."""
    p = np.asarray(proportions, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def hhi(shares):
    """Herfindahl-Hirschman Index for share vector (sum of squared shares)."""
    s = np.asarray(shares, dtype=float)
    return np.sum(s ** 2)

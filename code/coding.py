# -*- coding: utf-8 -*-
"""
Coding rules for the revised analysis (R1 revision).

Everything that turns a raw MoMA field into an analytic category lives here so
that the rules reported in the paper's supplement and the rules the pipeline
executes are the same text:

  * constituent tokens   positional parsing of the parenthesised ConstituentID /
                         Gender / Nationality lists attached to each record
  * gender               per-constituent and per-record coding
  * region               explicit nationality -> region lookup (no substring
                         matching), baseline Western / non-Western binary and
                         the alternative schemes used in the sensitivity table
  * credit line          donor-name extraction with suffix retention,
                         dedication-clause stripping, and co-donor / fund-led /
                         by-exchange flags
"""

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constituent tokens
# --------------------------------------------------------------------------


def split_paren_tokens(value):
    """Split '(a) (b (c)) ()' into ['a', 'b (c)', ''] (top-level groups only)."""
    if not isinstance(value, str):
        return []
    tokens, depth, buf = [], 0, []
    for ch in value:
        if ch == "(":
            if depth > 0:
                buf.append(ch)
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                tokens.append("".join(buf).strip())
                buf = []
            elif depth > 0:
                buf.append(ch)
            else:
                depth = 0
        elif depth > 0:
            buf.append(ch)
    return tokens


def split_ids(value):
    if isinstance(value, str):
        return re.findall(r"\d+", value)
    if pd.notna(value):
        return [str(int(value))]
    return []


# --------------------------------------------------------------------------
# Gender
# --------------------------------------------------------------------------

FEMALE_TOKENS = {"female", "female (transwoman)", "transgender woman"}
MALE_TOKENS = {"male", "male (trans? ftm?)"}
# Recorded by MoMA but outside the binary used in the paper's shares.
NONBINARY_TOKENS = {"non-binary", "gender non-conforming"}


def gender_token(token):
    """Per-constituent gender: 'F', 'M', 'NB' (recorded, non-binary) or None."""
    t = (token or "").strip().lower()
    if t in FEMALE_TOKENS:
        return "F"
    if t in MALE_TOKENS:
        return "M"
    if t in NONBINARY_TOKENS:
        return "NB"
    return None


def gender_record(gender_field):
    """Record-level gender from the record's Gender field.

    Female / Male : every gender-coded constituent is a woman / a man.
    Mixed         : at least one woman and at least one man.
    Other/Unknown : no constituent carries a binary gender code (unidentified
                    makers, firms, groups, non-binary only).
    Constituents without a code are ignored when at least one other
    constituent is coded.
    """
    codes = {gender_token(t) for t in split_paren_tokens(gender_field)}
    has_f, has_m = "F" in codes, "M" in codes
    if has_f and has_m:
        return "Mixed"
    if has_f:
        return "Female"
    if has_m:
        return "Male"
    return "Other/Unknown"


# --------------------------------------------------------------------------
# Region
# --------------------------------------------------------------------------

REGIONS = [
    "North America",
    "Western Europe",
    "Australia and New Zealand",
    "Eastern Europe and former USSR",
    "Latin America and Caribbean",
    "East Asia",
    "South and Southeast Asia",
    "Middle East and North Africa",
    "Sub-Saharan Africa",
    "Indigenous peoples of North America and Australia",
]

_REGION_MEMBERS = {
    "North America": [
        "American", "Canadian",
    ],
    # Nationality terms that name an Indigenous nation are kept apart from the
    # settler states; under the baseline rule they are non-Western.
    "Indigenous peoples of North America and Australia": [
        "Native American", "Canadian Inuit", "Member of Wood Mountain Lakota First Nations", "Oneida",
        "Spirit Lake Dakota/Cheyenne River Lakota", "Anmatyerr [Australian]",
    ],
    "Western Europe": [
        "German", "British", "French", "Italian", "Swiss", "Dutch", "Austrian",
        "Spanish", "Swedish", "Danish", "Belgian", "Finnish", "Norwegian",
        "Irish", "Scottish", "Greek", "Icelandic", "Portuguese", "Luxembourger",
        "Welsh", "English", "Catalan", "Cypriot",
    ],
    "Australia and New Zealand": [
        "Australian", "New Zealander",
    ],
    "Eastern Europe and former USSR": [
        "Russian", "Polish", "Czech", "Hungarian", "Croatian", "Romanian",
        "Slovenian", "Serbian", "Ukrainian", "Slovak", "Bosnian", "Georgian",
        "Bulgarian", "Czechoslovakian", "Macedonian", "Latvian", "Lithuanian",
        "Albanian", "Estonian", "Yugoslavian", "Belarusian", "Azerbaijani",
        "Kyrgyz", "Uzbekistani", "Moldovan", "Montenegrin", "Armenian",
    ],
    "Latin America and Caribbean": [
        "Brazilian", "Mexican", "Argentine", "Chilean", "Venezuelan",
        "Colombian", "Cuban", "Peruvian", "Uruguayan", "Haitian",
        "Puerto Rican", "Guatemalan", "Ecuadorian", "Bolivian", "Panamanian",
        "Nicaraguan", "Costa Rican", "Paraguayan", "Bahamian", "Salvadoran",
        "Trinidad and Tobagonian", "Jamaican", "Caribbean",
    ],
    "East Asia": ["Japanese", "Chinese", "Korean", "South Korean", "Taiwanese"],
    "South and Southeast Asia": [
        "Indian", "Thai", "Vietnamese", "Pakistani", "Bangladeshi", "Filipino",
        "Malaysian", "Singaporean", "Cambodian", "Indonesian", "Sri Lankan",
        "Nepali", "Ivatan", "Afghan",
    ],
    "Middle East and North Africa": [
        "Israeli", "Turkish", "Iranian", "Egyptian", "Lebanese", "Moroccan",
        "Palestinian", "Algerian", "Sudanese", "Tunisian", "Emirati", "Syrian",
        "Iraqi", "Kuwaiti", "Sahrawi", "Coptic", "Persian",
    ],
    "Sub-Saharan Africa": [
        "South African", "Nigerian", "Zimbabwean", "Congolese", "Senegalese",
        "Kenyan", "Malian", "Ghanaian", "Tanzanian", "Ethiopian", "Namibian",
        "Cameroonian", "Mozambican", "Ivorian", "Ugandan", "Burkinabé",
        "Beninese", "Sierra Leonean",
    ],
}

NATIONALITY_TO_REGION = {
    nat.lower(): region for region, members in _REGION_MEMBERS.items() for nat in members
}
UNKNOWN_NATIONALITY = {"", "nationality unknown", "unknown", "nan"}

# Baseline rule of the paper: North America, Western / Northern / Southern
# Europe, Australia and New Zealand are Western; every other region is not.
BASELINE_WESTERN_REGIONS = {"North America", "Western Europe", "Australia and New Zealand"}

# Alternative schemes. Each moves one contested block to the Western side.
SCHEMES = {
    "baseline": {"regions": set(), "nationalities": set()},
    "east_europe_western": {"regions": {"Eastern Europe and former USSR"}, "nationalities": set()},
    "latin_america_western": {"regions": {"Latin America and Caribbean"}, "nationalities": set()},
    "japan_western": {"regions": set(), "nationalities": {"japanese"}},
    "israel_turkey_western": {"regions": set(), "nationalities": {"israeli", "turkish"}},
    "narrowest": {
        "regions": {"Eastern Europe and former USSR", "Latin America and Caribbean"},
        "nationalities": {"japanese", "israeli", "turkish"},
    },
}

SCHEME_LABELS = {
    "baseline": "Baseline",
    "east_europe_western": "Eastern Europe and former USSR as Western",
    "latin_america_western": "Latin America and Caribbean as Western",
    "japan_western": "Japan as Western",
    "israel_turkey_western": "Israel and Turkey as Western",
    "narrowest": "All four recodings together",
}


def region_token(token):
    """Per-constituent region, or None when nationality is blank / unknown."""
    t = (token or "").strip().lower()
    if t in UNKNOWN_NATIONALITY:
        return None
    return NATIONALITY_TO_REGION.get(t, "UNMAPPED")


def origin_token(token, scheme="baseline"):
    """Per-constituent 'Western' / 'Non-Western' / None under a scheme."""
    region = region_token(token)
    if region is None or region == "UNMAPPED":
        return None
    rule = SCHEMES[scheme]
    if (
        region in BASELINE_WESTERN_REGIONS
        or region in rule["regions"]
        or (token or "").strip().lower() in rule["nationalities"]
    ):
        return "Western"
    return "Non-Western"


def _collapse(values):
    values = {v for v in values if v is not None}
    if not values:
        return np.nan
    if len(values) == 1:
        return next(iter(values))
    return "Mixed"


def origin_record(nationality_field, scheme="baseline"):
    """Record-level origin: Western / Non-Western / Mixed / NaN (no coded constituent)."""
    return _collapse(origin_token(t, scheme) for t in split_paren_tokens(nationality_field))


def region_record(nationality_field):
    """Record-level region: a region name, 'Mixed', or NaN."""
    regions = [region_token(t) for t in split_paren_tokens(nationality_field)]
    return _collapse(r for r in regions if r != "UNMAPPED")


def unmapped_nationalities(series):
    """Nationality tokens the lookup does not cover (should be empty)."""
    seen = {}
    for field in series.dropna().unique():
        for t in split_paren_tokens(field):
            if region_token(t) == "UNMAPPED":
                seen[t] = seen.get(t, 0) + 1
    return seen


# --------------------------------------------------------------------------
# Constituent-level (artist-level) table
# --------------------------------------------------------------------------

# Constituents that name no identifiable maker. 'Unidentified' covers anonymous
# makers; 'Various Artists' and similar group labels are placeholders but not
# anonymous (the makers are known and listed elsewhere).
UNIDENTIFIED_ARTIST_RE = re.compile(r"\b(?:unidentified|unknown|anonymous|unrecorded|not assigned)\b", re.IGNORECASE)
# Institutional donors: foundations, trusts, companies, newspapers, railways,
# associations, galleries and publishers. Museum-internal bodies (committees,
# councils, funds) and private collections named 'Collection' are not
# institutional donors.
INSTITUTIONAL_DONOR_RE = re.compile(
    r"\b(?:foundation|trust|inc|incorporated|corp|corporation|company|co|ltd|llc|gmbh|communications|"
    r"railroads?|railways?|association|society|gallery|galleries|press|shimbun|bank|editions|publishers?)\b",
    re.IGNORECASE,
)
PLACEHOLDER_ARTIST_RE = re.compile(
    r"\b(unidentified|unknown|anonymous|various|unrecorded|not assigned)\b", re.IGNORECASE
)


def explode_constituents(df, id_col="ConstituentID", gender_col="Gender", nat_col="Nationality"):
    """One row per (record, constituent) with per-constituent gender and region.

    Requires the three positional fields; they are aligned one-to-one in the
    MoMA export (verified on the full snapshot).
    """
    rows = []
    for rec_idx, ids, genders, nats in zip(
        df.index, df[id_col].map(split_ids), df[gender_col].map(split_paren_tokens), df[nat_col].map(split_paren_tokens)
    ):
        for pos, cid in enumerate(ids):
            g = genders[pos] if pos < len(genders) else ""
            n = nats[pos] if pos < len(nats) else ""
            rows.append((rec_idx, cid, gender_token(g), n.strip(), region_token(n), origin_token(n)))
    out = pd.DataFrame(rows, columns=["rec", "cid", "gender", "nationality", "region", "origin"])
    out.loc[out["region"] == "UNMAPPED", "region"] = None
    return out


# --------------------------------------------------------------------------
# Credit line
# --------------------------------------------------------------------------

GIFT_PHRASE_RE = re.compile(r"\b(?:gift|bequest)\s+of\s+", re.IGNORECASE)

# Clauses that follow the donor name and say something about the transfer,
# not about who the donor is.
TRAILING_CLAUSE_RE = re.compile(
    r"(?:\s*\(by exchange\)"
    r"|,?\s+by exchange\b"
    r"|,?\s+in\s+(?:loving\s+|fond\s+|grateful\s+)?(?:honor|honour|memory|celebration|appreciation|recognition|tribute|gratitude)\b"
    r"|,?\s+to honor\b"
    r"|,?\s+and\s+(?:partial\s+)?purchase\b"
    r"|,?\s+on the occasion\b"
    r"|,?\s*\(?through\b"
    r"|\s+and\s+(?:of\b|acquired\b|gift\b)"
    r"|,?\s+and\s+the\s+Committee\s+on\b"
    r"|\.\s+Conservation\b"
    r"|\s+\(both\b"
    r"|;"
    r").*$",
    re.IGNORECASE | re.DOTALL,
)
# Name-first credit lines: '<Name> Gift' or '<Name> Bequest' with nothing after
# it but an exchange note. Lists of funds and donors (commas, semicolons) are
# not matched.
NAME_FIRST_RE = re.compile(
    r"^(?:Acquired through the\s+|[^,;:]*?\bfunds from the\s+)?(?:The\s+)?(?P<name>[^,;:]+?)\s+(?:Gift|Bequest)"
    r"(?:\s*\((?:both\s+)?by exchange\)|,?\s+by exchange)?\.?$",
    re.IGNORECASE,
)
NAME_FIRST_GENERIC = {"anonymous", "an anonymous", "partial", "promised", "fractional", "fractional and promised"}
# Spelling variants of one named collection.
DONOR_ALIASES = {
    "Judith Rothschild Contemporary Drawings Collection": "Judith Rothschild Foundation Contemporary Drawings Collection",
}
BY_EXCHANGE_RE = re.compile(r"\bby exchange\b", re.IGNORECASE)

# ", Jr." / ", Inc." style suffixes that belong to the name before the comma.
SUFFIX_RE = re.compile(
    r"^\s*(Jr|Sr|II|III|IV|2nd|3rd|4th|Inc|Incorporated|Ltd|LLC|Co|Corp|S\.\s?A|N\.\s?V|GmbH|AG)\b\.?",
    re.IGNORECASE,
)
# "Kleiner, Bell & Co." style firm names with an internal comma.
FIRM_TAIL_RE = re.compile(
    r"^\s*([A-Z][\w'.\-]*(?:\s+[A-Z][\w'.\-]*)*),?\s*(?:&|and)\s+(Co|Company|Sons|Associates|Partners)\b\.?",
)
SUFFIX_END_RE = re.compile(r"(?:Jr|Sr|Inc|Ltd|Co|Corp|S\.A|N\.V)\.$")
# Firm names made of surnames ('Skidmore, Owings and Merrill') are one donor.
SURNAME_FIRM_RE = re.compile(r"^[A-Z][\w'’\-]+(?:, [A-Z][\w'’\-]+)+,? (?:and|&) [A-Z][\w'’\-]+\.?$")
FUND_PREFIX_RE = re.compile(r"\bfunds?\b|purchase|committee|acquired|council", re.IGNORECASE)
CO_DONOR_RE = re.compile(r"\band\b|&|\b(?:Mr|Mrs|Ms|Dr)\b\.?|,", re.IGNORECASE)

_SUFFIX_CANON = {
    "jr": "Jr.", "sr": "Sr.", "ii": "II", "iii": "III", "iv": "IV", "2nd": "2nd",
    "3rd": "3rd", "4th": "4th", "inc": "Inc.", "incorporated": "Inc.", "ltd": "Ltd.",
    "llc": "LLC", "co": "Co.", "corp": "Corp.", "s.a": "S.A.", "n.v": "N.V.",
    "gmbh": "GmbH", "ag": "AG",
}


def parse_credit_line(creditline, generic_terms=()):
    """Parse a gift / bequest credit line.

    Returns a dict with
      donor        first-named donor, with name suffixes kept, dedication and
                   exchange clauses removed; NaN when the line names no donor
                   or names a role ('the architect').
      co_donor     further donors are named after the first.
      fund_led     the line opens with a museum fund, committee or purchase
                   formula before the gift phrase.
      by_exchange  the work was acquired by exchange of an earlier gift.
    """
    empty = {"donor": np.nan, "co_donor": False, "fund_led": False, "by_exchange": False}
    if pd.isna(creditline):
        return empty
    cl = str(creditline).strip()
    m = GIFT_PHRASE_RE.search(cl)
    if not m:
        # Second credit-line form: the donor's name comes first ('Curt Valentin
        # Bequest', 'The Gilbert and Lila Silverman Fluxus Collection Gift',
        # 'Acquired through the Lillie P. Bliss Bequest (by exchange)').
        t = NAME_FIRST_RE.match(cl)
        if not t:
            return empty
        donor = DONOR_ALIASES.get(t.group("name").strip(), t.group("name").strip())
        if donor.lower() in generic_terms or donor.lower() in NAME_FIRST_GENERIC:
            return empty
        return {"donor": donor, "co_donor": False, "fund_led": False, "by_exchange": bool(BY_EXCHANGE_RE.search(cl))}
    prefix = cl[: m.start()]
    rest = cl[m.end():].split("\n")[0].strip()
    rest = re.sub(r"^the\s+", "", rest, flags=re.IGNORECASE)
    by_exchange = bool(BY_EXCHANGE_RE.search(rest))
    seg = TRAILING_CLAUSE_RE.sub("", rest).strip()

    head, sep, tail = seg.partition(",")
    donor = head.strip()
    remaining = tail if sep else ""
    if SURNAME_FIRM_RE.match(seg) and not FIRM_TAIL_RE.match(remaining):
        donor, remaining = seg, ""
    while remaining.strip():
        firm = FIRM_TAIL_RE.match(remaining)
        if firm:
            donor = f"{donor}, {firm.group(1)} & {_SUFFIX_CANON.get(firm.group(2).lower(), firm.group(2))}"
            remaining = remaining[firm.end():].lstrip(" ,")
            continue
        suffix = SUFFIX_RE.match(remaining)
        if suffix:
            donor = f"{donor}, {_SUFFIX_CANON[suffix.group(1).lower().replace(' ', '')]}"
            remaining = remaining[suffix.end():].lstrip(" ,")
            continue
        break

    donor = donor.strip(" ;")
    if not SUFFIX_END_RE.search(donor):
        donor = donor.rstrip(". ")
    if not donor or donor.lower() in generic_terms:
        return {**empty, "by_exchange": by_exchange}
    # A second donor joined by 'and' inside the first comma-free segment
    # ('Agnes Gund and Daniel Shapiro') is one joint donor unit, as in the
    # submitted analysis. Text after the first donor marks co-donors.
    co_donor = bool(remaining.strip()) and bool(CO_DONOR_RE.search(remaining)) or bool(
        remaining.strip() and re.match(r"^[A-Z]", remaining.strip()) and len(remaining.split()) >= 2
        and not re.fullmatch(r"(New York|Los Angeles|San Francisco|Buenos Aires|São Paulo|Mexico City)", remaining.strip())
    )
    return {
        "donor": donor,
        "co_donor": bool(co_donor),
        "fund_led": bool(FUND_PREFIX_RE.search(prefix)),
        "by_exchange": by_exchange,
    }


def donor_key(name):
    """Grouping key that ignores punctuation, case, 'and'/'&' and a leading 'the'."""
    if pd.isna(name):
        return np.nan
    key = str(name).lower()
    key = re.sub(r"\s+and\s+", " & ", key)
    key = re.sub(r"[.,]", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    key = re.sub(r"^the ", "", key)
    return key or np.nan

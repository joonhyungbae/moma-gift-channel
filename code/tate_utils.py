# Tate-specific utilities: acquisition from creditLine, nationality from placeOfBirth.
# Western/Non-Western uses same criteria as MoMA (config.WESTERN_KEYWORDS).

import re
import pandas as pd
import numpy as np

from config import WESTERN_KEYWORDS

# Tate creditLine: "Purchased...", "Presented by...", "Bequeathed by...", "Transferred from..."
def categorize_tate_acquisition(creditline):
    """Classify Tate creditLine into Purchase, Gift, Bequest, Other."""
    if pd.isna(creditline) or str(creditline).strip() == "":
        return "Other"
    cl = str(creditline).lower()
    if "purchased" in cl:
        return "Purchase"
    if "presented by" in cl or "presented to" in cl:
        return "Gift"
    if "bequeathed" in cl:
        return "Bequest"
    if "transferred from" in cl or "transferred to" in cl:
        return "Other"
    # Fallback: "given by" etc.
    if "given by" in cl or "gift of" in cl:
        return "Gift"
    return "Other"


# Tate artist_data: placeOfBirth is like "London, United Kingdom", "Polska", "Beijing, Zhonghua"
# Map country (last part of place) to nationality string for WESTERN_KEYWORDS match
COUNTRY_TO_NATIONALITY = {
    "united kingdom": "british", "uk": "british", "england": "british", "scotland": "scottish",
    "wales": "welsh", "ireland": "irish",
    "united states": "american", "usa": "american", "us": "american",
    "canada": "canadian", "france": "french", "germany": "german", "deutschland": "german",
    "italy": "italian", "italia": "italian", "spain": "spanish", "netherlands": "dutch",
    "belgium": "belgian", "switzerland": "swiss", "schweiz": "swiss",
    "austria": "austrian", "sweden": "swedish", "norway": "norwegian", "denmark": "danish",
    "finland": "finnish", "suomi": "finnish", "portugal": "portuguese", "greece": "greek",
    "australia": "australian", "new zealand": "new zealand", "iceland": "icelandic",
    "poland": "polish", "polska": "polish", "czech republic": "czech", "czechia": "czech",
    "hungary": "hungarian", "romania": "romanian", "russia": "russian",
    "ukraine": "ukrainian", "belarus": "belarusian", "israel": "israeli", "yisra'el": "israeli",
    "japan": "japanese", "china": "chinese", "zhonghua": "chinese", "korea": "korean",
    "taiwan": "taiwanese", "india": "indian", "brazil": "brazilian", "mexico": "mexican",
    "argentina": "argentine", "egypt": "egyptian", "iran": "iranian", "iraq": "iraqi",
    "al-'iraq": "iraqi", "turkey": "turkish", "türkiye": "turkish", "pakistan": "pakistani",
    "nigeria": "nigerian", "south africa": "south african", "indonesia": "indonesian",
    "thailand": "thai", "vietnam": "vietnamese", "philippines": "filipino",
    "estonia": "estonian", "latvia": "latvian", "lithuania": "lithuanian",
    "slovenia": "slovenian", "slovakia": "slovak", "croatia": "croatian",
    "serbia": "serbian", "bosnia": "bosnian", "bulgaria": "bulgarian",
}


def place_of_birth_to_nationality(place):
    """Extract country from Tate placeOfBirth and return nationality string for geographic_origin."""
    if pd.isna(place) or str(place).strip() == "":
        return np.nan
    s = str(place).strip().lower()
    # Take part after last comma (country), or whole string
    if "," in s:
        country = s.split(",")[-1].strip()
    else:
        country = s
    return COUNTRY_TO_NATIONALITY.get(country, country)


def geographic_origin_tate(place_of_birth):
    """Western vs Non-Western from Tate placeOfBirth; same criteria as MoMA."""
    nat = place_of_birth_to_nationality(place_of_birth)
    if pd.isna(nat):
        return np.nan
    nat_lower = str(nat).lower()
    for kw in WESTERN_KEYWORDS:
        if kw in nat_lower:
            return "Western"
    return "Non-Western"


def gender_grouped_tate(gender):
    """Map Tate gender (Female/Male) to Female / Male / Other-Unknown."""
    if pd.isna(gender):
        return "Other/Unknown"
    g = str(gender).strip().lower()
    if "female" in g:
        return "Female"
    if "male" in g:
        return "Male"
    return "Other/Unknown"


def extract_tate_donor_from_creditline(creditline):
    """Extract donor from Tate creditLine: 'Presented by X' or 'Bequeathed by X'.

    Returns NaN for role/generic labels ('the artist', 'the architect', 'the estate',
    etc.) since these aggregate many distinct people under one string and inflate
    concentration metrics.
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
    for phrase in ("presented by the ", "bequeathed by the ", "given by the ",
                   "presented by ", "bequeathed by ", "given by "):
        idx = cl_lower.find(phrase)
        if idx >= 0:
            rest = cl[idx + len(phrase):].strip()
            for sep in (",", "\n"):
                if sep in rest:
                    rest = rest.split(sep)[0].strip()
                    break
            rest = re.sub(r"\s*\d{4}\s*$", "", rest).strip()
            if not rest:
                return np.nan
            if rest.strip().lower() in GENERIC_DONOR_ROLE_TERMS:
                return np.nan
            return rest
    return np.nan

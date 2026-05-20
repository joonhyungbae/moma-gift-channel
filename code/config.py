# Configuration for sn-article analysis (MoMA acquisitions and diversity)
# 데이터셋은 모두 data/ 폴더에 있음 (data/ 기준)

import os

# Project root (parent of code/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 데이터셋 폴더 (고정)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Input files in data/
ARTWORKS_CSV = "Artworks.csv"
ARTISTS_CSV = "Artists.csv"
# Artworks.csv가 Git LFS 포인터일 때 시도할 실제 데이터 파일 (data/ 내)
ARTWORKS_FALLBACK = [
    "Artworks_full.csv",
    "Artworks_original.csv",
    "Artworks_main.csv",
]
PROCESSED_CSV = "processed_moma_data.csv"

# 분석 스크립트(02~07)에서만 로드할 컬럼. Title, Artist, URL 등 대용량 문자열 제외해 메모리 절감.
USECOLS_ANALYSIS = [
    "AcquisitionType", "Gender_Grouped", "GeographicOrigin", "YearAcquired",
    "ArtistBirthYear", "ArtistAgeAtAcquisition", "BeginDate", "Department",
    "CreditLine", "Nationality", "NamedFund",
]


def get_data_path(filename):
    """파일 경로 해석: DATA_DIR(data/) 우선, 그다음 OUTPUT_DIR, code/, PROJECT_ROOT."""
    for base in [DATA_DIR, OUTPUT_DIR, os.path.join(PROJECT_ROOT, "code"), PROJECT_ROOT]:
        p = os.path.join(base, filename)
        if os.path.isfile(p):
            return p
    return os.path.join(DATA_DIR, filename)


def get_artworks_path():
    """
    Artworks 데이터 파일 경로. data/Artworks.csv가 있으면 사용,
    없거나 LFS 포인터(크기 < 1000바이트)면 data/ 내 Artworks_full 등 fallback 사용.
    """
    p = os.path.join(DATA_DIR, ARTWORKS_CSV)
    if os.path.isfile(p) and os.path.getsize(p) >= 1000:
        return p
    for name in ARTWORKS_FALLBACK:
        path = os.path.join(DATA_DIR, name)
        if os.path.isfile(path):
            return path
    return None


def get_processed_path():
    """전처리된 분석용 데이터 경로 (data/ 또는 output/)."""
    return get_data_path(PROCESSED_CSV)

def get_output_path(filename):
    """결과 파일은 output/에 저장."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)

# AcquisitionType categories (priority order per paper).
# Enhanced reclassification (12_unknown_other_enhanced_mice) adds from Unknown/Other:
# Fund Purchase (subcategory of Purchase), Institutional Transfer, Anonymous Gift, Collection Gift.
ACQUISITION_TYPES = [
    "Artist Gift",
    "Partial Gift/Purchase",
    "Bequest",
    "Purchase",
    "Gift",
    "Unknown/Other",
]

# Main departments for analysis (paper tables)
MAIN_DEPARTMENTS = [
    "Architecture & Design",
    "Drawings & Prints",
    "Media and Performance",
    "Painting & Sculpture",
    "Photography",
]

# Western nationality keywords for GeographicOrigin.
# Paper definition: North America + Western/Northern/Southern Europe + Australia/NZ.
# Eastern Europe (Russia, Poland, Czechia, Hungary, Romania, Slovakia, Slovenia, Croatia,
# Bosnia, Serbia, Bulgaria, Ukraine, Belarus, Baltic states) and Israel are treated as
# NON-Western, consistent with the appendix regional disaggregation that lists "Eastern
# Europe" as a non-Western region.
WESTERN_KEYWORDS = [
    "american", "united states", "canadian", "british", "english", "scottish",
    "irish", "welsh", "french", "german", "italian", "spanish", "dutch",
    "belgian", "swiss", "austrian", "swedish", "norwegian", "danish", "finnish",
    "portuguese", "greek", "australian", "new zealand", "iceland", "luxembour",
]

# Non-Western role/profession terms that the donor parser should not return as donor
# names. The MoMA credit-line corpus contains formulations like "Gift of the architect"
# or "Gift of the designer" that name a role rather than an individual; treating those
# strings as donors inflates donor-concentration metrics.
GENERIC_DONOR_ROLE_TERMS = {
    "architect", "architects", "designer", "designers", "manufacturer", "manufacturers",
    "artist", "artists", "photographer", "photographers", "publisher", "publishers",
    "estate", "family", "foundation", "company", "firm", "studio", "office",
    "above", "donor", "donors", "anonymous donor", "anonymous", "various donors",
}

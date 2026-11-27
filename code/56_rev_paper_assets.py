# -*- coding: utf-8 -*-
"""
Paper assets for the revised manuscript.

Every data-derived number in the manuscript is a LaTeX macro defined in
rev_numbers.tex, and every table body is generated here, so the text cannot
drift from the pipeline. Run after scripts 50-55.

Outputs (output/, copied to moma_paper/output/ by the caller):
  rev_numbers.tex                 \\newcommand definitions
  rev_numbers.csv                 the same values, readable
  rev_tab_*.tex                   main-text tables
  rev_supp_*.tex                  supplement tables
  fig_volume_vs_gap.{pdf,eps}     Figure 1
  fig_record_vs_artist.{pdf,eps}  Figure 2
"""

import os
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coding import GIFT_PHRASE_RE, REGIONS, origin_token
from config import MAIN_DEPARTMENTS, PROJECT_ROOT, get_processed_path
from rev_utils import artist_names, constituents, load_gifts, out, pct, record_shares

PAPER_OUT = os.path.join(PROJECT_ROOT, "moma_paper", "output")
MACROS = {}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
GRAY = ["#111111", "#3b3b3b", "#666666", "#8a8a8a", "#b0b0b0", "#d0d0d0"]


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def fint(v):
    return f"{int(round(float(v))):,}".replace(",", "{,}")


def fdec(v, d=1):
    return f"{float(v):.{d}f}"


def fsigned(v, d=2):
    v = float(v)
    if round(v, d) == 0:
        v = 0.0
    sign = "+" if v >= 0 else "-"
    return f"\\ensuremath{{{sign}{abs(v):.{d}f}}}"


def put(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in MACROS, f"duplicate macro {name}"
    MACROS[name] = str(value)


def tex_escape(s):
    s = str(s)
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def write_table(name, caption, header, rows, align, label, note=None, size=r"\small", placement="ht", resize=False):
    body = [r"\begin{table}[" + placement + "]", r"\centering", r"\tbl{" + caption + "}"]
    inner = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        inner.append(r if isinstance(r, str) else " & ".join(str(c) for c in r) + r" \\")
    inner += [r"\bottomrule", r"\end{tabular}"]
    block = "\n".join(inner)
    if resize:
        size = r"\footnotesize\setlength{\tabcolsep}{4pt}"
    body.append("{" + size + block + "}")
    if note:
        body.append(r"\begin{tabnote}" + note + r"\end{tabnote}")
    body.append(r"\label{" + label + "}")
    body.append(r"\end{table}")
    with open(out(name), "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")


def cell(v, d=1, dash_if=None):
    if v is None or (isinstance(v, float) and np.isnan(v)) or (dash_if is not None and dash_if):
        return "--"
    return fdec(v, d)


def csigned(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    v = float(v)
    if round(v, d) == 0:
        v = 0.0
    return f"${'+' if v >= 0 else '-'}{abs(v):.{d}f}$"


# --------------------------------------------------------------------------
CASES = [("K", "Kleiner, Bell & Co."), ("R", "Rockefeller lineage"), ("P", "Jean Pigozzi"),
         ("C", "Peter J. Cohen"), ("J", "Judith Rothschild Foundation"), ("G", "Agnes Gund")]
CASE_TEX = {"Kleiner, Bell & Co.": r"Kleiner, Bell \& Co.", "Rockefeller lineage": "Rockefeller lineage",
            "Jean Pigozzi": "Jean Pigozzi", "Peter J. Cohen": "Peter J.~Cohen",
            "Judith Rothschild Foundation": "Judith Rothschild Fdn.", "Agnes Gund": "Agnes Gund"}


FIG1_OFFSETS = {
    "is_F": {"Kleiner, Bell & Co.": (8, 22), "Rockefeller lineage": (-60, -30), "Jack Shear": (8, -18),
             "Judith Rothschild Foundation": (10, 14), "Peter J. Cohen": (8, 6), "Jean Pigozzi": (-8, -22),
             "Agnes Gund": (8, 8), "Patricia Phelps de Cisneros": (-8, 14), "Helen Kornblum": (-8, 6)},
    "is_NW": {"Kleiner, Bell & Co.": (8, 24), "Rockefeller lineage": (-10, 30), "Jack Shear": (8, -16),
              "Judith Rothschild Foundation": (8, -14), "Peter J. Cohen": (8, -24), "Jean Pigozzi": (8, 6),
              "Agnes Gund": (-8, 16), "Patricia Phelps de Cisneros": (-34, 16), "Helen Kornblum": (-8, -18)},
}
FIG2_OFFSETS = {
    "contrib_f": {"Jack Shear": (4, 8), "Helen Kornblum": (6, 0), "Judith Rothschild Foundation": (6, -12),
                  "Peter J. Cohen": (4, -14), "Agnes Gund": (6, 8), "Jean Pigozzi": (-6, 10), "Rockefeller lineage": (-6, -14)},
    "contrib_nw": {"Jack Shear": (-4, 10), "Patricia Phelps de Cisneros": (6, 4), "Judith Rothschild Foundation": (-10, 12),
                   "Jean Pigozzi": (6, -12), "Kleiner, Bell & Co.": (-4, -16), "Rockefeller lineage": (-30, 12)},
}


def main():
    gifts = load_gifts()
    names = artist_names()
    cons = constituents(gifts, names)
    cons["unit"] = gifts["Unit"].reindex(cons["rec"]).to_numpy()
    museum = pd.read_csv(out("rev_museum.csv")).iloc[0]
    units = pd.read_csv(out("rev_units.csv"), index_col=0)
    six = pd.read_csv(out("rev_six_cases.csv"), index_col=0)
    cf = pd.read_csv(out("rev_joint_counterfactual.csv")).set_index("removed")
    loo = pd.read_csv(out("rev_artist_unit_contributions.csv"), index_col=0)
    first = pd.read_csv(out("rev_artist_first_source.csv"), index_col=0)
    conc = pd.read_csv(out("rev_artist_concentration.csv")).set_index("measure")
    classes = pd.read_csv(out("rev_artist_size_classes.csv")).set_index("class")
    bound = pd.read_csv(out("rev_rothschild_origin_bound.csv")).iloc[0]
    assoc = pd.read_csv(out("rev_volume_gap_association.csv")).set_index("measure")
    gden = pd.read_csv(out("rev_gender_denominators.csv"), index_col=0)
    gkind = pd.read_csv(out("rev_gender_uncoded_kinds.csv"), index_col=0)
    galt = pd.read_csv(out("rev_gender_alternative_rules.csv"))
    gunc = pd.read_csv(out("rev_gender_uncoded_by_donor.csv"), index_col=0)
    gval = pd.read_csv(out("rev_gender_validation.csv")).set_index("metric")["value"]
    schemes = pd.read_csv(out("rev_region_schemes.csv")).set_index("scheme")
    regrec = pd.read_csv(out("rev_region_composition_records.csv"), index_col=0)
    regart = pd.read_csv(out("rev_region_composition_artists.csv"), index_col=0)
    rock = pd.read_csv(out("rev_rockefeller.csv")).set_index("part")
    rock_strings = pd.read_csv(out("rev_rockefeller_strings.csv"), index_col=0)
    typ = pd.read_csv(out("rev_typology_classification.csv"), index_col=0)
    typsum = pd.read_csv(out("rev_typology_summary.csv")).set_index("metric")["value"]
    typsens = pd.read_csv(out("rev_typology_threshold_sensitivity.csv"))
    clus = pd.read_csv(out("rev_typology_clustering.csv"))
    tate = pd.read_csv(out("rev_tate_vs_moma_patterns.csv"), index_col=0)
    tate_units = pd.read_csv(out("rev_tate_units.csv"), index_col=0)
    tate_rules = pd.read_csv(out("rev_tate_rules.csv"), index_col=0)
    sube = pd.read_csv(out("rev_suberas.csv"))
    sel = pd.read_csv(out("rev_selection_audit.csv"))
    decades = pd.read_csv(out("rev_new_artists_by_decade.csv"))

    # ---------------------------------------------------------------- sample
    full = pd.read_csv(get_processed_path(), usecols=["Department", "YearAcquired", "CreditLine"], low_memory=False)
    main_all = full[full["Department"].isin(MAIN_DEPARTMENTS)
                    & pd.to_numeric(full["YearAcquired"], errors="coerce").between(1929, 2024)]
    put("NAll", fint(len(main_all)))
    frac = main_all["CreditLine"].fillna("").str.contains(r"fractional|promised", case=False)
    put("NFracPromised", fint(frac.sum()))
    n = len(gifts)
    put("NGift", fint(n))
    put("NUnits", fint(museum["n_units"]))
    put("NStrings", fint(gifts["Donor"].nunique()))
    for key, col in (("CoDonor", "n_co_donor"), ("FundLed", "n_fund_led"), ("ByExchange", "n_by_exchange")):
        put("N" + key, fint(museum[col]))
        put("Pct" + key, fdec(pct(museum[col], n)))
    sizes = gifts["Unit"].value_counts()
    k1 = int(round(0.01 * len(sizes)))
    put("TopUnitShare", fdec(pct(sizes.iloc[0], n)))
    put("TopOnePctN", fint(k1))
    put("TopOnePctShare", fdec(pct(sizes.head(k1).sum(), n)))
    put("TopTwentyShare", fdec(pct(sizes.head(20).sum(), n)))
    put("NUnitsOneRecord", fint((sizes == 1).sum()))
    put("PctUnitsOneRecord", fdec(pct((sizes == 1).sum(), len(sizes))))
    big = sizes[sizes >= 100]
    put("NUnitsBig", fint(len(big)))
    put("ShareBigUnits", fdec(pct(big.sum(), n)))
    # Record- against artist-level contribution across the large units (absolute values; Section 5.2, abstract)
    _bc = units.loc[units.index.isin(big.index), ["contrib_f", "contrib_nw"]].join(loo[["art_contrib_f", "art_contrib_nw"]])
    for _k in ("f", "nw"):
        _r, _a = _bc["contrib_" + _k].abs(), _bc["art_contrib_" + _k].fillna(0).abs()
        K = _k.upper() if _k == "f" else "NW"
        put("MedRecCon" + K, fdec(_r.median(), 3)); put("MedArtCon" + K, fdec(_a.median(), 3))
        put("NArtSmaller" + K, fint((_a < _r).sum()))
        assert _a.median() < 0.5 * _r.median()  # 'less than half as much'
        assert (_a < _r).sum() > len(_bc) / 2

    # ---------------------------------------------------------------- gender
    g0 = gden.loc["All donor-identified gifts"]
    put("NFemRec", fint(g0["female"])); put("NMaleRec", fint(g0["male"]))
    put("NMixedRec", fint(g0["mixed"])); put("PctMixedRec", fdec(g0["pct_mixed"]))
    put("NUncodedRec", fint(g0["uncoded"])); put("PctUncodedRec", fdec(g0["pct_uncoded"]))
    put("NGValid", fint(museum["g_valid"]))
    put("BaseF", fdec(museum["pct_f"], 2)); put("BaseFOne", fdec(museum["pct_f"], 1))
    put("BoundFLo", fdec(g0["pct_f_lower"])); put("BoundFHi", fdec(g0["pct_f_upper"]))
    put("FracF", fdec(galt.loc[galt["rule"].str.startswith("Fractional"), "pct_f"].iloc[0]))
    put("SubmittedF", fdec(galt.loc[galt["rule"].str.startswith("Submitted"), "pct_f"].iloc[0], 2))
    put("UncodedADPct", fdec(gden.loc["Department: Architecture & Design", "pct_uncoded"]))
    put("UncodedPhotoPct", fdec(gden.loc["Department: Photography", "pct_uncoded"]))
    put("UncodedDPPct", fdec(gden.loc["Department: Drawings & Prints", "pct_uncoded"]))
    put("UncodedUnidentPct", fdec(gkind.loc["Unidentified maker", "pct"], 0))
    put("UncodedNamedPct", fdec(gkind.loc["Named maker, gender not recorded", "pct"], 0))
    put("UncodedFirmPct", fdec(gkind.loc["Firm, studio or collective", "pct"], 0))
    put("CohenShareUncoded", fdec(gunc.loc["Peter J. Cohen", "pct_of_all_uncoded"], 0))
    put("WdLinked", fint(gval["linked to a Wikidata item"]))
    put("WdBoth", fint(gval["both sources give a binary code"]))
    put("WdAgree", fint(gval["agreement (n)"]))
    put("WdAgreePct", fdec(gval["agreement (%)"], 2))
    put("WdDisagree", fint(gval["both sources give a binary code"] - gval["agreement (n)"]))
    nb = (cons["gender"] == "NB").sum()
    put("NNonBinaryCons", fint(cons.loc[cons["gender"] == "NB", "cid"].nunique()))

    # ---------------------------------------------------------------- origin
    put("NNWRec", fint(museum["nw"])); put("NOValid", fint(museum["o_valid"]))
    put("BaseNW", fdec(museum["pct_nw"], 2)); put("BaseNWOne", fdec(museum["pct_nw"], 1))
    put("NMixedOrigin", fint(museum["n_mixed_origin"])); put("NUncodedOrigin", fint(museum["n_uncoded_origin"]))
    put("PctUncodedOrigin", fdec(pct(museum["n_uncoded_origin"], n)))
    put("NArtists", fint(museum["n_artists"]))
    put("ArtBaseF", fdec(museum["art_pct_f"], 2)); put("ArtBaseFOne", fdec(museum["art_pct_f"], 1))
    put("NArtF", fint(museum["art_f"])); put("NArtGValid", fint(museum["art_g_valid"]))
    put("ArtBaseNW", fdec(museum["art_pct_nw"], 2)); put("ArtBaseNWOne", fdec(museum["art_pct_nw"], 1))
    put("NArtNW", fint(museum["art_nw"])); put("NArtOValid", fint(museum["art_o_valid"]))
    put("NArtNoOrigin", fint(museum["n_artists"] - museum["art_o_valid"]))

    # ------------------------------------------------------------- six cases
    for p, u in CASES:
        r = six.loc[u]
        a = loo.loc[u] if u in loo.index else None
        put(p + "N", fint(r["n"])); put(p + "NArt", fint(r["n_artists"]))
        put(p + "ShareRec", fdec(r["share_of_records"]))
        put(p + "PctF", fdec(r["pct_f"])); put(p + "PctNW", fdec(r["pct_nw"]))
        put(p + "GValid", fint(r["g_valid"])); put(p + "OValid", fint(r["o_valid"]))
        put(p + "NF", fint(r["f"])); put(p + "NNW", fint(r["nw"]))
        put(p + "ArtF", fdec(r["art_pct_f"])); put(p + "ArtNW", fdec(r["art_pct_nw"]))
        put(p + "GapF", fsigned(r["gap_f"], 1)); put(p + "GapNW", fsigned(r["gap_nw"], 1))
        put(p + "ExpF", fdec(r["exp_f"])); put(p + "ExpNW", fdec(r["exp_nw"]))
        put(p + "MGapF", fsigned(r["gap_f_matched"], 1)); put(p + "MGapNW", fsigned(r["gap_nw_matched"], 1))
        put(p + "ConF", fsigned(r["contrib_f"])); put(p + "ConNW", fsigned(r["contrib_nw"]))
        put(p + "ArtConF", fsigned(a["art_contrib_f"] if a is not None else 0.0))
        put(p + "ArtConNW", fsigned(a["art_contrib_nw"] if a is not None else 0.0))
        put(p + "Intro", fint(first["new_artists"].get(u, 0))); put(p + "Sole", fint(r["sole_n"]))
        put(p + "PeakYear", str(int(r["peak_year"]))); put(p + "PeakShare", fdec(r["peak_year_share"]))
        put(p + "TopArtistShare", fdec(r["top_artist_share"]))
        put(p + "TopDeptShare", fdec(r["top_dept_share"]))
        put(p + "EraStart", str(int(r["era_start"]))); put(p + "EraEnd", str(int(r["era_end"])))
        put(p + "FLo", fdec(r["f_lo"])); put(p + "FHi", fdec(r["f_hi"]))
        put(p + "NWLo", fdec(r["nw_lo"])); put(p + "NWHi", fdec(r["nw_hi"]))
        put(p + "CoDonor", fint(r["n_co_donor"])); put(p + "FundLed", fint(r["n_fund_led"]))
        put(p + "ByExchange", fint(r["n_by_exchange"]))

    # case-specific details
    k = gifts[gifts["Unit"] == "Kleiner, Bell & Co."]
    _kr = six.loc["Kleiner, Bell & Co."]
    if True:  # Section 4.1: non-Western expectation above the resampling range, female inside it
        assert _kr["exp_nw"] > _kr["nw_hi"] and _kr["f_lo"] <= _kr["exp_f"] <= _kr["f_hi"]
    put("KAccessionYears", fint(k["YearAcquired"].nunique()))
    ref = gifts[(gifts["Department"] == "Drawings & Prints") & gifts["YearAcquired"].between(1967, 1974)
                & (gifts["Unit"] != "Kleiner, Bell & Co.")]
    rs = record_shares(ref)
    put("KRefF", fdec(rs["pct_f"])); put("KRefNW", fdec(rs["pct_nw"])); put("KRefN", fint(rs["n"]))
    p_ = gifts[gifts["Unit"] == "Jean Pigozzi"]
    p19 = p_[p_["YearAcquired"] == 2019]
    put("PNPeak", fint(len(p19)))
    _p19c = cons[cons["rec"].isin(p19.index) & ~cons["placeholder"]]
    put("PPeakNArt", fint(_p19c["cid"].nunique()))
    put("PAlphabet", fint(p19["Title"].fillna("").str.contains("Alphabet B", case=False).sum()))
    put("PAlphabetCards", fint(p19["Title"].fillna("").str.contains("from Alphabet B", case=False).sum()))
    put("PBouabre", fint(p_["Artist"].fillna("").str.contains("Bouabr").sum()))
    pa = cons[(cons["unit"] == "Jean Pigozzi") & ~cons["placeholder"]].drop_duplicates("cid")
    put("PArtNWn", fint((pa["origin"] == "Non-Western").sum()))
    c_ = gifts[gifts["Unit"] == "Peter J. Cohen"]
    put("CUnident", fint(c_["Artist"].fillna("").str.fullmatch("Unidentified photographer").sum()))
    put("CPctUncoded", fdec(gden.loc["Case: Peter J. Cohen", "pct_uncoded"], 0))
    cf_year = c_[c_["Gender_Grouped"] == "Female"]["YearAcquired"]
    put("CFemYear", str(int(cf_year.mode().iloc[0]))); put("CFemYearN", fint((cf_year == cf_year.mode().iloc[0]).sum()))
    put("CParker", fint(c_["Artist"].fillna("").str.contains("Blanche Parker").sum()))
    j_ = gifts[gifts["Unit"] == "Judith Rothschild Foundation"]
    raw = j_["DonorRaw"].map(lambda v: re.sub(r"^the\s+", "", str(v), flags=re.I))
    put("JStrings", fint(raw.nunique()))
    put("JMainN", fint((raw == "Judith Rothschild Foundation").sum()))
    put("JKerdimun", fint(raw.str.contains("Kerdimun").sum()))
    put("JEarlyDated", fint((j_["YearAcquired"] < 1990).sum()))
    put("JMixed", fint((j_["Gender_Grouped"] == "Mixed").sum()))
    put("JUncodedGender", fint((j_["Gender_Grouped"] == "Other/Unknown").sum()))
    put("JPlaceholder", fint(six.loc["Judith Rothschild Foundation", "placeholder_records"]))
    put("JEastEurope", fint(regrec.loc["Judith Rothschild Foundation", "Eastern Europe and former USSR"]))
    put("JArtNoOrigin", fint(bound["artists_without_nationality"]))
    put("JArtConNWBound", fsigned(bound["contribution_if_non_western_pp"]))
    put("JArtConNWCoded", fsigned(bound["contribution_coded_only_pp"]))
    put("JArtConNWLow", fsigned(bound["contribution_if_western_pp"]))
    put("JAltPctNW", fdec(schemes.loc["Eastern Europe and former USSR as Western", "Judith Rothschild Foundation | pct_nw"]))
    put("JAltConNW", fsigned(schemes.loc["Eastern Europe and former USSR as Western",
                                         "Judith Rothschild Foundation | contribution_pp"]))
    g_ = gifts[gifts["Unit"] == "Agnes Gund"]
    g21 = g_[g_["YearAcquired"] == 2021]
    put("GNPeak", fint(len(g21))); put("GPeakArtists", fint(g21["Cluster"].nunique()))
    _gwo = record_shares(g_[g_["YearAcquired"] != 2021])
    put("GWoPeakF", fdec(_gwo["pct_f"])); put("GPeakFemRec", fint(int(g21["is_F"].sum())))
    put("GPeakFundLed", fint(g21["fund_led"].sum()))
    solo = record_shares(g_[~g_["co_donor"] & ~g_["fund_led"]])
    put("GSoloN", fint(solo["n"])); put("GSoloF", fdec(solo["pct_f"]))
    put("GArtGapF", fsigned(six.loc["Agnes Gund", "art_pct_f"] - museum["art_pct_f"], 1))
    rk = rock
    put("RStrings", fint(len(rock_strings)))
    put("RAbbyN", fint(rk.loc["Abby Aldrich Rockefeller, all", "n"]))
    put("RAbbyShare", fdec(pct(rk.loc["Abby Aldrich Rockefeller, all", "n"], rk.loc["Rockefeller lineage (all strings)", "n"]), 0))
    put("RAbbyExchange", fint(rk.loc["Abby Aldrich Rockefeller, by exchange", "n"]))
    put("RAbbyGiftN", fint(rk.loc["Abby Aldrich Rockefeller, gifts", "n"]))
    put("RAbbyGiftEnd", str(int(rk.loc["Abby Aldrich Rockefeller, gifts", "era_end"])))
    ab = gifts[(gifts["Donor"] == "Abby Aldrich Rockefeller") & ~gifts["by_exchange"]]
    put("RAbbyAfterDeath", fint((ab["YearAcquired"] > 1948).sum()))
    yc = gifts.groupby(["Unit", "YearAcquired"]).size().sort_values(ascending=False)
    put("BiggestBlockN", fint(yc.iloc[0])); put("BiggestBlockYear", str(int(yc.index[0][1])))
    put("BiggestBlockUnitIsShear", "1" if yc.index[0][0] == "Jack Shear" else "0")
    put("KPeakN", fint(yc.loc[("Kleiner, Bell & Co.", int(six.loc["Kleiner, Bell & Co.", "peak_year"]))]))
    order = sizes.index.tolist()
    put("JRank", fint(order.index("Judith Rothschild Foundation") + 1))
    put("ROthersN", fint(rk.loc["Other family members", "n"]))
    put("RAbbyConF", fsigned(rk.loc["Abby Aldrich Rockefeller, all", "contrib_f"]))
    put("RAbbyConNW", fsigned(rk.loc["Abby Aldrich Rockefeller, all", "contrib_nw"]))

    # ------------------------------------------------- joint and concentration
    a6 = cf.loc["All six"]
    put("SixN", fint(a6["n_removed"])); put("SixShareRec", fdec(pct(a6["n_removed"], n)))
    put("SixConF", fsigned(-a6["rec_delta_f"])); put("SixConNW", fsigned(-a6["rec_delta_nw"]))
    # The same joint figure with Abby Aldrich Rockefeller's records in place of the family lineage (referee 2, comment 6).
    _five = [u_ for _, u_ in CASES if u_ != "Rockefeller lineage"]
    _mask = gifts["Unit"].isin(_five) | gifts["Donor"].eq("Abby Aldrich Rockefeller")
    _all, _rest = record_shares(gifts), record_shares(gifts[~_mask])
    put("SixAbbyConF", fsigned(_all["pct_f"] - _rest["pct_f"])); put("SixAbbyConNW", fsigned(_all["pct_nw"] - _rest["pct_nw"]))
    put("SixArtConF", fsigned(-a6["art_delta_f"])); put("SixArtConNW", fsigned(-a6["art_delta_nw"]))
    put("SixArtConNWOne", fdec(-a6["art_delta_nw"], 1))
    _all_s = record_shares(gifts); _wo_j = record_shares(gifts[gifts["Unit"] != "Judith Rothschild Foundation"])
    put("JWoNWOne", fdec(_wo_j["pct_nw"], 1))
    assert -a6["rec_delta_nw"] > 0 and -a6["art_delta_nw"] > 0
    put("SixConNWAbs", fdec(-a6["rec_delta_nw"], 2)); put("SixArtConNWAbs", fdec(-a6["art_delta_nw"], 2))
    _ratio = a6["rec_delta_nw"] / a6["art_delta_nw"]
    _words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    assert 2 <= _ratio < 10
    put("SixRecArtTimes", _words[int(_ratio)])   # 'more than <word> times'

    put("SixArtLost", fint(a6["artists_lost"])); put("SixArtLostPct", fdec(pct(a6["artists_lost"], museum["n_artists"])))
    put("SixShareNWRec", fdec(schemes.loc["Baseline", "six_share_of_nw_records"]))
    rnd = cf[cf.index.str.startswith("Random")].iloc[0]
    put("TailArtLost", fint(round(rnd["artists_lost"])))
    put("TailArtConF", fsigned(-rnd["art_delta_f"])); put("TailArtConNW", fsigned(-rnd["art_delta_nw"]))
    assert rnd["artists_lost"] > a6["artists_lost"]  # Section 5.2: the random records hold more artists
    for key, m in (("NWRec", "non-Western records"), ("FRec", "female records"),
                   ("NWArt", "non-Western artists"), ("FArt", "female artists")):
        r = conc.loc[m]
        put("Half" + key, fint(r["units_for_half"])); put("Half" + key + "Shuf", fint(r["shuffled_mean"]))
        put("Half" + key + "Lo", fint(r["shuffled_lo"])); put("Half" + key + "Hi", fint(r["shuffled_hi"]))
        put("TopTen" + key, fdec(r["top10_share"])); put("TopTen" + key + "Shuf", fdec(r["top10_shuffled_mean"]))
        put("Sources" + key, fint(r["units_supplying_any"]))
    put("HalfAllRec", fint(conc.loc["all records", "units_for_half"]))
    put("HalfAllArt", fint(conc.loc["all artists", "units_for_half"]))
    t10 = classes.loc["ten largest units"]
    put("TenRecShare", fdec(t10["pct_records"])); put("TenNWArtShare", fdec(t10["pct_new_nw_artists"]))
    put("TenFArtShare", fdec(t10["pct_new_f_artists"]))
    put("TenNWArtN", fint(t10["new_nw_artists"])); put("TenFArtN", fint(t10["new_f_artists"]))
    sm = classes.loc["1-9 records"]
    put("SmallUnitsN", fint(sm["units"])); put("SmallRecShare", fdec(sm["pct_records"]))
    put("SmallNWArtShare", fdec(sm["pct_new_nw_artists"])); put("SmallFArtShare", fdec(sm["pct_new_f_artists"]))
    bg = classes.loc["100+ records"]
    put("BigNWArtShare", fdec(bg["pct_new_nw_artists"], 0)); put("BigFArtShare", fdec(bg["pct_new_f_artists"], 0))

    # other named units
    for p, u in (("Shear", "Jack Shear"), ("Cis", "Patricia Phelps de Cisneros"), ("Korn", "Helen Kornblum")):
        r = units.loc[u]
        put(p + "N", fint(r["n"])); put(p + "NArt", fint(r["n_artists"]))
        put(p + "ConF", fsigned(r["contrib_f"])); put(p + "ConNW", fsigned(r["contrib_nw"]))
        put(p + "ArtConF", fsigned(loo.loc[u, "art_contrib_f"])); put(p + "ArtConNW", fsigned(loo.loc[u, "art_contrib_nw"]))
        put(p + "TopArtistShare", fdec(r["top_artist_share"])); put(p + "PeakShare", fdec(r["peak_year_share"]))
        put(p + "PeakYear", str(int(r["peak_year"])))
        put(p + "EraStart", str(int(r["era_start"]))); put(p + "EraEnd", str(int(r["era_end"])))
        put(p + "NewNW", fint(first.loc[u, "new_nw"])); put(p + "NewF", fint(first.loc[u, "new_f"]))
        put(p + "ArtF", fdec(r["art_pct_f"])); put(p + "ArtNW", fdec(r["art_pct_nw"]))
    nk = "Nihon Keizai Shimbun"
    put("NihonN", fint(sizes[nk])); put("NihonNewNW", fint(first.loc[nk, "new_nw"]))
    put("NihonYear", str(int(gifts.loc[gifts["Unit"] == nk, "YearAcquired"].mode().iloc[0])))

    # single-artist blocks among units with 100+ records
    bigu = units[units["n"] >= 100]
    blocks = bigu[bigu["top_artist_share"] > 90]
    meta = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "Artists_full.csv"), dtype={"ConstituentID": str}).set_index("ConstituentID")
    top_ids = blocks["top_artist"].astype(int).astype(str)
    art = cons.drop_duplicates("cid").set_index("cid")
    tg, to = top_ids.map(art["gender"]), top_ids.map(art["origin"])
    put("NBlocks", fint(len(blocks))); put("PctBlocks", fdec(pct(len(blocks), len(bigu)), 0))
    put("NBlocksMaleWestern", fint(((tg == "M") & (to == "Western")).sum()))
    put("NBlocksFemale", fint((tg == "F").sum())); put("NBlocksNW", fint((to == "Non-Western").sum()))
    put("NBlocksBoth", fint(((tg == "F") & (to == "Non-Western")).sum())); put("NBlocksNonPerson", fint(tg.isna().sum()))
    assert ((tg == "M") & (to == "Western")).sum() + (to == "Non-Western").sum() + (tg == "F").sum() \
        - ((tg == "F") & (to == "Non-Western")).sum() + tg.isna().sum() == len(blocks), "single-artist breakdown in Section 5.1 must add up"
    ev = bigu[bigu["peak_year_share"] > 90]
    put("NEvents", fint(len(ev))); put("PctEvents", fdec(pct(len(ev), len(bigu)), 0))
    put("EventsRecShare", fdec(pct(ev["n"].sum(), n)))
    neg = bigu["contrib_nw"].sort_values()
    put("NNegNW", fint((bigu["contrib_nw"] < -0.05).sum())); put("NPosNW", fint((bigu["contrib_nw"] > 0.05).sum()))

    # associations
    for key, m in (("GapF", "records: |gap| female"), ("GapNW", "records: |gap| non-Western"),
                   ("MGapF", "records: |matched gap| female"), ("MGapNW", "records: |matched gap| non-Western"),
                   ("ArtGapF", "artists: |gap| female"), ("ArtGapNW", "artists: |gap| non-Western"),
                   ("ConF", "records: |contribution| female"), ("ConNW", "records: |contribution| non-Western")):
        r = assoc.loc[m]
        put("Rho" + key, fsigned(r["spearman_rho"])); put("Pval" + key, fdec(r["perm_p"], 2) if r["perm_p"] >= 0.005 else "<0.01")
        put("RhoN" + key, fint(r["n_units"]))

    # typology
    put("TypOthers", fint(typsum["non-anchor units"]))
    put("TypClean", fint(typsum["non-anchor clean"])); put("TypAmbig", fint(typsum["non-anchor ambiguous"]))
    put("TypNone", fint(typsum["non-anchor unclassified"]))
    put("TypAny", fint(int(float(typsum["non-anchor clean"])) + int(float(typsum["non-anchor ambiguous"]))))
    assert 0.35 < (int(float(typsum["non-anchor clean"])) + int(float(typsum["non-anchor ambiguous"]))) / int(float(typsum["non-anchor units"])) < 0.45, "text says about two in five"
    assert int(float(typsum["non-anchor clean"])) + int(float(typsum["non-anchor ambiguous"])) + int(float(typsum["non-anchor unclassified"])) == int(float(typsum["non-anchor units"]))
    for key, ty in (("Vol", "Volume gift"), ("Found", "Founding-lineage gift"), ("Single", "Single-artist gift"),
                    ("Anon", "Anonymous-archive gift"), ("Inst", "Institutional block gift"),
                    ("Cross", "Cross-departmental gift")):
        put("Typ" + key, fint(typsum[f"type count (non-anchor, any match): {ty}"]))
    put("TypCleanMin", fint(typsens["clean"].min())); put("TypCleanMax", fint(typsens["clean"].max()))
    put("TypNoneMin", fint(typsens["unclassified"].min())); put("TypNoneMax", fint(typsens["unclassified"].max()))
    put("SilMin", fdec(min(clus["kmeans_silhouette"].min(), clus["ward_silhouette"].min()), 2))
    put("SilMax", fdec(max(clus["kmeans_silhouette"].max(), clus["ward_silhouette"].max()), 2))
    put("SilShufMin", fdec(clus["shuffled_p95"].min(), 2)); put("SilShufMax", fdec(clus["shuffled_p95"].max(), 2))
    put("SilSix", fdec(clus.loc[clus["k"] == 6, "kmeans_silhouette"].iloc[0], 2))
    put("SilSixShuf", fdec(clus.loc[clus["k"] == 6, "shuffled_p95"].iloc[0], 2))
    put("TypARI", fdec(clus["ari_rules_vs_k6_clean_units"].iloc[0], 2))

    # regional schemes
    for key, s in (("Base", "Baseline"), ("EE", "Eastern Europe and former USSR as Western"),
                   ("LA", "Latin America and Caribbean as Western"), ("JP", "Japan as Western"),
                   ("IT", "Israel and Turkey as Western"), ("All", "All four recodings together")):
        put("Sch" + key + "Museum", fdec(schemes.loc[s, "museum_pct_nw"])); put("Sch" + key + "Six", fdec(schemes.loc[s, "six_contribution_pp"], 2))

    # Tate
    T = lambda row, col: tate.loc[row, col]
    for col, p in (("MoMA", "M"), ("Tate", "T")):
        put(p + "TateRecords", fint(float(T("gift_records", col)))); put(p + "TateUnits", fint(float(T("donor_units", col))))
        put(p + "TateBig", fint(float(T("units_100plus", col))))
        put(p + "TateTopOne", fdec(float(T("top_1pct_share_of_records", col))))
        put(p + "TateEvent", fdec(float(T("units_peak_year_over_90", col)), 0))
        put(p + "TateSingle", fdec(float(T("units_top_artist_over_90", col)), 0))
        put(p + "TateRhoF", fsigned(float(T("vol_gap_rho_f", col)))); put(p + "TateRhoNW", fsigned(float(T("vol_gap_rho_nw", col))))
        put(p + "TatePF", fdec(float(T("vol_gap_p_f", col)), 2)); put(p + "TatePNW", fdec(float(T("vol_gap_p_nw", col)), 2))
        put(p + "TateRecF", fdec(float(T("record_pct_f", col)))); put(p + "TateRecNW", fdec(float(T("record_pct_nw", col))))
        put(p + "TateArtF", fdec(float(T("artist_pct_f", col)))); put(p + "TateArtNW", fdec(float(T("artist_pct_nw", col))))
    fer = tate_units.loc["Alicia and León Ferrari"]
    put("FerrariN", fint(fer["n"])); put("FerrariCon", fsigned(fer["contrib_nw"], 1))
    put("TateBigFifty", fint(float(T("units_100plus", "Tate (units with 50+ records)"))))

    put("NWEastEurope", fint(regrec.loc["All donor-identified gifts", "Eastern Europe and former USSR"]))
    put("NWEastEuropePct", fdec(pct(regrec.loc["All donor-identified gifts", "Eastern Europe and former USSR"], museum["nw"]), 0))
    ee = "Eastern Europe and former USSR as Western"
    for p, u in CASES:
        put(p + "ConNWEE", fsigned(schemes.loc[ee, f"{u} | contribution_pp"]))
    # clustering: range of k at which the observed silhouette exceeds the shuffled 95th percentile
    above = clus[clus["kmeans_silhouette"] > clus["shuffled_p95"]]["k"]
    put("SilAboveFrom", fint(above.min()) if len(above) else "--"); put("SilAboveTo", fint(above.max()) if len(above) else "--")
    put("NClusterFeatures", "eight")
    put("TypAnchorsStableMin", fint(typsens["anchors_own_pattern"].min()))
    # Tate rules
    for p, lab in (("M", "MoMA"), ("T", "Tate")):
        r = tate_rules.loc[lab]
        put(p + "RulesUnits", fint(r["units"])); put(p + "RulesOne", fint(r["one pattern"])); put(p + "RulesNone", fint(r["none"]))
        put(p + "RulesSingle", fint(r["Single-artist gift"])); put(p + "RulesVol", fint(r["Volume gift"]))
        put(p + "RulesInst", fint(r["Institutional block gift"])); put(p + "RulesFound", fint(r["Founding-lineage gift"]))
        put(p + "RulesCross", fint(r["Cross-departmental gift"])); put(p + "RulesAnon", fint(r["Anonymous-archive gift"]))
    put("ShearConNWEE", fsigned(pd.read_csv(out("rev_region_scheme_top_contributors.csv")).query(
        "scheme == 'Eastern Europe and former USSR as Western' and unit == 'Jack Shear'")["contribution_pp"].iloc[0]))
    for p_, u_ in (("Parkett", "Parkett"), ("CisFirst", "Patricia Phelps de Cisneros"), ("KornFirst", "Helen Kornblum")):
        put(p_ + "NewArt", fint(first.loc[u_, "new_artists"])); put(p_ + "NewF", fint(first.loc[u_, "new_f"]))
        put(p_ + "NewNW", fint(first.loc[u_, "new_nw"]))
    # Joint artist-level contribution of the six if the Rothschild artists without a nationality count as non-Western.
    six_units = {u for _, u in CASES}
    cc = cons[~cons["placeholder"]]
    A_ = cc.groupby("cid").agg(origin=("origin", "first"), units=("unit", lambda x: frozenset(x)))
    in_j = A_["units"].map(lambda x: "Judith Rothschild Foundation" in x)
    o2 = A_["origin"].where(~(in_j & A_["origin"].isna()), "Non-Western")
    keep = ~A_["units"].map(lambda x: x <= six_units)
    sh = lambda o: 100 * (o == "Non-Western").sum() / o.notna().sum()
    put("SixArtConNWBound", fsigned(sh(o2) - sh(o2[keep])))
    # Artist-level contributions when Eastern Europe and the former USSR count as Western.
    _nat = cc.groupby("cid")["nationality"].first()
    o_ee = _nat.map(lambda t: origin_token(t, "east_europe_western")).reindex(A_.index)
    keep_j = ~A_["units"].map(lambda x: x <= {"Judith Rothschild Foundation"})
    put("SixArtConNWEE", fsigned(sh(o_ee) - sh(o_ee[keep])))
    put("JArtConNWEE", fsigned(sh(o_ee) - sh(o_ee[keep_j])))
    # Nationality filled from Wikidata (01_data_preprocessing.py, 59_rev_wikidata_origin.py)
    _wd = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "wikidata_origin_cache.csv"), dtype={"cid": str})
    _filled = _wd[(_wd["status"] == "matched") & _wd["nationality_term"].notna()]
    put("WdOrigUncoded", fint(len(_wd))); put("WdOrigFilled", fint(len(_filled)))
    put("WdOrigFilledPct", fdec(pct(len(_filled), len(_wd)), 0))
    _wj = _wd["rothschild"].astype(bool)
    put("JWdUncoded", fint(_wj.sum())); put("JWdFilled", fint(_filled["rothschild"].astype(bool).sum()))
    put("WdOrigLinked", fint(_wd["link_source"].notna().sum()))
    put("WdOrigBirthRule", fint((_filled["rule"] == "place of birth").sum()))
    put("WdOrigFilledRec", fint((gifts["NationalityFilled"].fillna(0) > 0).sum()))
    # Worked example for the accessions questions: Drawings & Prints gifts accessioned in 2019.
    dp = gifts[(gifts["Department"] == "Drawings & Prints") & (gifts["YearAcquired"] == 2019)]
    s_all, s_wo = record_shares(dp), record_shares(dp[dp["Unit"] != "Jean Pigozzi"])
    put("DPNineteenN", fint(s_all["n"])); put("DPNineteenNW", fdec(s_all["pct_nw"]))
    put("DPNineteenNWwoP", fdec(s_wo["pct_nw"])); put("DPNineteenPShare", fdec(pct((dp["Unit"] == "Jean Pigozzi").sum(), len(dp))))
    # ---- random gifts of the same structure (script 57): a supplementary check, four designs.
    nun = pd.read_csv(out("rev_null_units.csv"))
    nsum = pd.read_csv(out("rev_null_summary.csv"))
    NULL_DESIGNS = ["rec_matched", "rec_pooled", "art_matched", "art_pooled"]
    def nsel(museum, share):
        return nsum[(nsum["museum"] == museum) & (nsum["share"] == share)].set_index("design").loc[NULL_DESIGNS]
    for key, share in (("F", "female"), ("NW", "non-Western")):
        t_ = nsel("MoMA", share)
        flagged = t_["bh_up"] + t_["bh_down"]
        put("NullFlagMin" + key, fint(flagged.min())); put("NullFlagMax" + key, fint(flagged.max()))
        put("NullCannotDown" + key, fint(t_["cannot_flag_down"].min()))
    put("NullUnits", fint(nsel("MoMA", "non-Western")["units"].iloc[0]))
    put("NullDraws", "100{,}000")
    put("NullAltUnits", fint(nsel("MoMA", "non-Western (Eastern Europe as Western)")["units"].iloc[0]))
    put("NullUnderTen", fint(nsel("MoMA", "non-Western")["units_under_10_makers"].iloc[0]))
    put("NullMedianEff", fdec(nsel("MoMA", "non-Western")["median_effective_makers"].iloc[0], 0))
    def nrow(unit, share):
        return nun[(nun["museum"] == "MoMA") & (nun["share"] == share) & (nun["unit"] == unit)].iloc[0]
    for p_, u_ in CASES:
        for key, share in (("F", "female"), ("NW", "non-Western")):
            # The matched record draw is centred on the matched baseline of Table 1 and Table S11.
            assert abs(nrow(u_, share)["rec_matched_expected"] - six.loc[u_, "exp_f" if key == "F" else "exp_nw"]) < 0.05, (u_, key)
    shear_ = nrow("Jack Shear", "non-Western")
    put("ShearNullMakers", fint(shear_["makers"])); put("ShearExpNW", fdec(shear_["rec_matched_expected"]))
    put("ShearPctNW", fdec(shear_["obs_share"]))
    assert all(shear_[f"{d_}_q"] >= 0.05 for d_ in NULL_DESIGNS), "supplement cites Jack Shear as a unit no design flags"
    cdc = "Judith Rothschild Foundation Contemporary Drawings Collection"
    # ---- first-entry attribution (script 55)
    ten = pd.read_csv(out("rev_ten_largest_decade_matched.csv"), index_col=0)
    put("TenFArtObs", fint(ten.loc["f", "observed"])); put("TenFArtExp", fint(ten.loc["f", "expected"]))
    put("TenNWArtObs", fint(ten.loc["nw", "observed"])); put("TenNWArtExp", fint(ten.loc["nw", "expected"]))
    put("TenNWArtCmp", "more" if ten.loc["nw", "observed"] > ten.loc["nw", "expected"] else "fewer")
    fe = pd.read_csv(out("rev_first_entry_overview.csv")).iloc[0]
    put("NArtCollection", fint(fe["artists_in_collection"])); put("NFirstEntryGift", fint(fe["first_entry_by_identified_gift"]))
    put("PctFirstEntryGift", fdec(fe["pct"], 0))
    dd = decades.set_index("decade")
    put("DecFStart", fdec(dd.loc[1930, "pct_f"], 0)); put("DecFEnd", fdec(dd.loc[2020, "pct_f"], 0))
    put("DecFMid", fdec(dd.loc[1990, "pct_f"], 0)); put("DecFEndN", fint(dd.loc[2020, "new_f"]))
    assert all(round(dd.loc[k_, "pct_f"]) == 25 for k_ in (1990, 2000, 2010)), "text says 25% in each decade from the 1990s to the 2010s"
    put("DecUnitsFStart", fint(dd.loc[1930, "units_with_new_f"])); put("DecUnitsFTens", fint(dd.loc[2010, "units_with_new_f"]))
    # ---- related units
    for p_, u_ in (("CDC", cdc), ("Silv", "Gilbert and Lila Silverman Fluxus Collection")):
        r_ = units.loc[u_]
        put(p_ + "N", fint(r_["n"])); put(p_ + "NArt", fint(r_["n_artists"])); put(p_ + "Year", str(int(r_["peak_year"])))
        put(p_ + "PctF", fdec(r_["pct_f"])); put(p_ + "PctNW", fdec(r_["pct_nw"]))
        put(p_ + "ConF", fsigned(r_["contrib_f"])); put(p_ + "ConNW", fsigned(r_["contrib_nw"]))
        put(p_ + "NewF", fint(first.loc[u_, "new_f"])); put(p_ + "NewArt", fint(first.loc[u_, "new_artists"]))
    both_j = gifts[gifts["Unit"].isin([cdc, "Judith Rothschild Foundation"])]
    bj = record_shares(both_j)
    put("JBothN", fint(bj["n"])); put("JBothPctNW", fdec(bj["pct_nw"])); put("JBothPctF", fdec(bj["pct_f"]))
    assert sizes.index[0] == "Gilbert and Lila Silverman Fluxus Collection" and sizes.index[1] == "Kleiner, Bell & Co."
    multi = [u_ for u_ in sizes.index[:5] if units.loc[u_, "peak_year_share"] <= 90]
    assert multi[0] == "Kleiner, Bell & Co.", "text calls Kleiner, Bell the largest unit given over several years"
    assert first["new_f"].idxmax() == cdc, "text names the Contemporary Drawings Collection as the leading source of new women"
    assert set(first.sort_values("new_nw", ascending=False).head(3).index) == {
        "Nihon Keizai Shimbun", "Patricia Phelps de Cisneros", "Judith Rothschild Foundation"}
    assert first["new_artists"].idxmax() == "Judith Rothschild Foundation"
    down_order = units["contrib_nw"].sort_values().head(4).index.tolist()
    assert down_order == ["Jack Shear", cdc, "Kleiner, Bell & Co.", "Rockefeller lineage"], down_order
    up_order = units["contrib_nw"].sort_values(ascending=False).head(3).index.tolist()
    assert up_order == ["Judith Rothschild Foundation", "Jean Pigozzi", "Patricia Phelps de Cisneros"], up_order
    others = [schemes.loc[k, "six_contribution_pp"] for k in ("Latin America and Caribbean as Western", "Japan as Western",
                                                              "Israel and Turkey as Western")]
    put("SchOtherMin", fdec(min(others), 2)); put("SchOtherMax", fdec(max(others), 2))
    put("SampleSharePct", fdec(pct(n, len(main_all)), 0))
    vol_others = typ[(~typ["is_case"]) & typ["types"].fillna("").str.contains("Volume gift")].index.tolist()
    assert set(vol_others) == {"Lawrence Benenson", "Curt Valentin"}, vol_others
    stern = main_all["CreditLine"].fillna("").str.fullmatch(r"The Louis E\. Stern Collection").sum()
    anon = main_all["CreditLine"].fillna("").str.fullmatch(r"(?i)anonymous gift|given anonymously").sum()
    put("NStern", fint(stern)); put("NAnon", fint(anon))
    name_first = ~gifts["CreditLine"].fillna("").map(lambda c: bool(GIFT_PHRASE_RE.search(c)))
    put("NNameFirst", fint(name_first.sum()))
    put("NOtherParserGain", fint(n - int(name_first.sum()) - 49740))  # 49,740 is the submitted sample size
    assert n - int(name_first.sum()) - 49740 > 0
    put("SixConNWOne", fdec(schemes.loc["Baseline", "six_contribution_pp"], 1))
    assert schemes.loc["Eastern Europe and former USSR as Western", "six_contribution_pp"] < 0.5 * schemes.loc["Baseline", "six_contribution_pp"], "text says most of the rise disappears"
    put("SchEESixSigned", fsigned(schemes.loc["Eastern Europe and former USSR as Western", "six_contribution_pp"]))
    put("SchEESixOne", fdec(schemes.loc["Eastern Europe and former USSR as Western", "six_contribution_pp"], 1))
    put("NIndig", fint(regrec.loc["All donor-identified gifts", "Indigenous peoples of North America and Australia"]))
    put("JFirstNewNW", fint(first.loc["Judith Rothschild Foundation", "new_nw"]))
    _lead = first["new_nw"].sort_values(ascending=False).index[:3].tolist()  # order of the Section 5.2 sentence
    assert _lead == ["Judith Rothschild Foundation", nk, "Patricia Phelps de Cisneros"], _lead
    put("JFirstNewF", fint(first.loc["Judith Rothschild Foundation", "new_f"]))
    put("SmallArtShare", fdec(classes.loc["1-9 records", "pct_new_artists"]))
    put("TenArtShare", fdec(classes.loc["ten largest units", "pct_new_artists"]))
    put("BigArtShare", fdec(classes.loc["100+ records", "pct_new_artists"], 0))

    # Where each (purposively chosen) case stands on its axis among units with 100+ records.
    from coding import INSTITUTIONAL_DONOR_RE
    pop = units[units["n"] >= 100]
    def rank_of(series, unit, ascending=False):
        r_ = series.dropna().rank(ascending=ascending, method="min")
        return int(r_[unit])
    def ordinal(k):
        return ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"][k - 1]
    inst = pop[[bool(INSTITUTIONAL_DONOR_RE.search(u_)) for u_ in pop.index]]
    put("KRankVolume", ordinal(rank_of(pop["n"], "Kleiner, Bell & Co.")))
    put("RRankEra", ordinal(rank_of(pop["era_years"], "Rockefeller lineage")))
    put("PRankNW", ordinal(rank_of(pop["pct_nw"].where(pop["o_valid"] >= 30), "Jean Pigozzi")))
    put("CRankF", ordinal(rank_of(pop["pct_f"].where(pop["g_valid"] >= 30), "Peter J. Cohen")))
    put("JRankInstPeak", ordinal(rank_of(inst["peak_year_share"], "Judith Rothschild Foundation")))
    put("JRankInstSize", ordinal(rank_of(inst.loc[inst["peak_year_share"] > 90, "n"], "Judith Rothschild Foundation")))
    put("GRankSpread", ordinal(rank_of(pop["top_dept_share"], "Agnes Gund", ascending=True)))
    assert rank_of(pop["era_years"], "Rockefeller lineage") == 1
    big_inst = inst.loc[inst["peak_year_share"] > 90, "n"].sort_values(ascending=False)
    assert big_inst.index[:2].tolist() == [cdc, "Judith Rothschild Foundation"], big_inst.head(3)
    assert rank_of(pop["top_dept_share"], "Agnes Gund", ascending=True) == 2, "text says one other large donor matches Gund's spread"

    # ---------------------------------------------------------------- write macros
    with open(out("rev_numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% Generated by code/56_rev_paper_assets.py. Do not edit by hand.\n")
        for k_, v in MACROS.items():
            f.write(f"\\newcommand{{\\{k_}}}{{{v}\\xspace}}\n")
    pd.DataFrame({"macro": list(MACROS), "value": list(MACROS.values())}).to_csv(out("rev_numbers.csv"), index=False)

    # ================================================================ tables
    # Table: six cases at three levels (two panels).
    def case_rows(dim):
        rows = []
        for p, u in CASES:
            r = six.loc[u]
            a_ = loo.loc[u] if u in loo.index else None
            valid = r["g_valid"] if dim == "f" else r["o_valid"]
            mark = "$^{a}$" if valid < 0.5 * r["n"] else ""
            row = [CASE_TEX[u]]
            if dim == "f":
                row += [fint(r["n"]), fint(r["n_artists"]), fint(first["new_artists"].get(u, 0))]
            row += [cell(r[f"pct_{dim}"]) + mark, cell(r[f"art_pct_{dim}"]), csigned(r[f"gap_{dim}_matched"], 1),
                    csigned(r[f"contrib_{dim}"]), csigned(a_[f"art_contrib_{dim}"] if a_ is not None else 0)]
            if dim == "nw":
                row.append(csigned(schemes.loc["Eastern Europe and former USSR as Western", f"{u} | contribution_pp"]))
            rows.append(" & ".join(row) + r" \\")
        tot = ["Gift channel"]
        if dim == "f":
            tot += [fint(n), fint(museum["n_artists"]), ""]
        tot += [cell(museum[f"pct_{dim}"]), cell(museum[f"art_pct_{dim}"]), "", "", ""] + ([""] if dim == "nw" else [])
        rows += [r"\midrule", " & ".join(tot) + r" \\"]
        return "\n".join(rows)
    head5 = r"\% rec. & \% art. & Matched gap & Contrib.\ rec. & Contrib.\ art."
    with open(out("rev_tab_cases.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n"
                "\\tbl{The six cases counted as records, as distinct artists, and as artists new to the collection.}\n"
                "{\\begin{minipage}{\\textwidth}\\centering\n"
                "{\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
                "\\multicolumn{9}{@{}l}{\\emph{(a) Counts and female share}} \\\\\n"
                "Donor & Records & Artists & New & " + head5 + " \\\\\n\\midrule\n" + case_rows("f") + "\n\\bottomrule\n"
                "\\end{tabular}}\n\n\\vspace{6pt}\n"
                "{\\footnotesize\\setlength{\\tabcolsep}{3.2pt}\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
                "\\multicolumn{7}{@{}l}{\\emph{(b) Non-Western share}} \\\\\n"
                "Donor & " + head5 + " & Alt.\\ rule$^{b}$ \\\\\n\\midrule\n" + case_rows("nw") + "\n\\bottomrule\n\\end{tabular}}\n\\end{minipage}}\n"
                "\\begin{tabnote}`New' counts artists new to the collection through the donor, whose earliest MoMA accession in any channel is a gift from the donor. "
                "Shares (\\%) are of coded records and of coded artists. "
                "`Matched gap' is the record share minus the share of other donors' gifts in the same departments and decades. "
                "`Contrib.' is the gift-channel share minus the share without the donor's gifts. `rec.' counts records and `art.' artists. Gaps and contributions are in percentage points and are computed from unrounded shares. "
                "$^{a}$Fewer than half of the donor's records carry the code. "
                "$^{b}$Contribution to the non-Western share of records when Eastern Europe and the former USSR count as Western (percentage points).\\end{tabnote}\n"
                "\\label{tab:cases}\n\\end{table}\n")

    # Table: MoMA and Tate patterns.
    def trow(label, key, d=1, signed=False, integer=False):
        vals = []
        for col in ("MoMA", "Tate"):
            v = float(tate.loc[key, col])
            vals.append(fint(v) if integer else (csigned(v) if signed else fdec(v, d)))
        return [label] + vals
    trows = [
        trow("Gift records naming a donor", "gift_records", integer=True),
        trow("Donor units", "donor_units", integer=True),
        trow("Units with at least 100 records", "units_100plus", integer=True),
        trow(r"Share of records held by the top 1\% of units (\%)", "top_1pct_share_of_records"),
        r"\midrule",
        r"\multicolumn{3}{l}{\emph{Among units with at least 100 records}} \\",
        trow(r"Units with more than 90\% of records in one year (\%)", "units_peak_year_over_90", 0),
        trow(r"Units with more than 90\% of records by one artist (\%)", "units_top_artist_over_90", 0),
    ]
    trows.append(r"\midrule")
    trows.append(r"\multicolumn{3}{l}{\emph{Pattern rules of Table~\ref{tab:donor_type_taxonomy}, units with at least 100 records (at MoMA other than the six anchors)}} \\")
    trows.append(["Units", fint(typsum["non-anchor units"]), fint(tate_rules.loc["Tate", "units"])])
    trows.append(["Matching exactly one pattern", fint(typsum["non-anchor clean"]), fint(tate_rules.loc["Tate", "one pattern"])])
    trows.append(["Matching more than one pattern", fint(typsum["non-anchor ambiguous"]), fint(tate_rules.loc["Tate", "several"])])
    trows.append(["Matching the single-artist rule", fint(typsum["type count (non-anchor, any match): Single-artist gift"]),
                  fint(tate_rules.loc["Tate", "Single-artist gift"])])
    trows.append(["Matching no pattern", fint(typsum["non-anchor unclassified"]), fint(tate_rules.loc["Tate", "none"])])
    write_table("rev_tab_tate.tex", "Gift patterns at MoMA and at the Tate, computed with comparable definitions.",
                ["", "MoMA", "Tate"], trows, "lrr", "tab:tate",
                note="Both museums' credit lines are parsed with the same rules for donor names and dedication clauses. "
                     "Tate origin is coded from place of birth and MoMA origin from nationality, and the Tate's departments are coarser. "
                     "The public Tate file ends in 2013. Tate credit lines that name an acquisition channel (for example the Turner Bequest) are excluded.")

    # ------------------------------------------------------------ supplement
    # S: every donor unit with 100 or more records, MoMA and Tate (one table; replaces the earlier top-20,
    # pattern and Tate unit tables).
    PAT_CODE = {"Volume gift": "V", "Founding-lineage gift": "L", "Single-artist gift": "S", "Anonymous-archive gift": "A",
                "Institutional block gift": "I", "Cross-departmental gift": "C"}
    def codes_of(t):
        return "".join(PAT_CODE[t_.strip()] for t_ in t.split(";")) if isinstance(t, str) and t.strip() else "--"
    def unit_row(name, n, art, era, peak, dept, artist, pf, gv, pnw, ov, cf, cnw, pat, star=False):
        return [tex_escape(name) + (r"$^{\ast}$" if star else ""), fint(n), fint(art), fint(era), fdec(peak, 0), fdec(dept, 0), fdec(artist, 0),
                cell(pf, dash_if=gv < 30), cell(pnw, dash_if=ov < 30), csigned(cf), csigned(cnw), pat]
    rows = [r"\multicolumn{12}{@{}l}{\emph{(a) MoMA}} \\"]
    for u, r in typ.sort_values("n", ascending=False).iterrows():
        uu = units.loc[u]
        rows.append(unit_row(u, r["n"], r["n_artists"], r["era_years"], r["peak_year_share"], r["top_dept_share"], r["top_artist_share"],
                             uu["pct_f"], uu["g_valid"], uu["pct_nw"], uu["o_valid"], uu["contrib_f"], uu["contrib_nw"], codes_of(r["types"]), r["is_case"]))
    rows.append(r"\midrule" + "\n" + r"\multicolumn{12}{@{}l}{\emph{(b) Tate}} \\")
    for u, r in tate_units.sort_values("n", ascending=False).iterrows():
        rows.append(unit_row(u, r["n"], r["n_artists"], r["era_years"], r["peak_year_share"], r["top_dept_share"], r["top_artist_share"],
                             r["pct_f"], r["g_valid"], r["pct_nw"], r["o_valid"], r["contrib_f"], r["contrib_nw"], codes_of(r.get("patterns"))))
    head = ("Donor unit & Rec. & Art. & Era & Peak & Dept. & Artist & Fem. & Non-W. & Con. & Con. & Pat. \\\\\n"
            " & & & (yr) & yr (\\%) & (\\%) & (\\%) & (\\%) & (\\%) & F & NW & \\\\\n\\midrule\n")
    with open(out("rev_supp_units.tex"), "w", encoding="utf-8") as f:
        f.write("{\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n\\begin{longtable}{@{}p{0.30\\textwidth}rrrrrrrrrrl@{}}\n"
                "\\caption{Every donor unit with at least 100 records at MoMA (a) and at the Tate (b), by number of records. $^{\\ast}$Anchor case. "
                "`Era' is the span from first to last accession, `Peak yr' the share of records accessioned in the busiest year, `Dept.' the share in the largest department, "
                "and `Artist' the share by the largest artist. Female and non-Western shares are of coded records, and a dash marks fewer than 30 coded records (typically gifts by unidentified makers). "
                "`Con.' is the unit's contribution to the museum's gift-record share (percentage points). "
                "`Pat.' gives the patterns of Table~2 in the article that the unit matches: V volume gift, L founding-lineage gift, S single-artist gift, "
                "A anonymous-archive gift, I institutional block gift, C cross-departmental gift. Donor names are given as written in the credit lines, and Tate departments and origin are coded as described in Section~\\ref{sec:s_tate}.}\\label{tab:s_units}\\\\\n\\toprule\n"
                + head + "\\endfirsthead\n\\toprule\n" + head + "\\endhead\n\\bottomrule\n\\endlastfoot\n")
        for r in rows:
            f.write((r if isinstance(r, str) else " & ".join(r) + r" \\") + "\n")
        f.write("\\end{longtable}}\n")

    # S: gender denominators.
    rows = []
    for label, r in gden.iterrows():
        rows.append([tex_escape(label.replace("Acquired 1920s", "Acquired 1929").replace("All donor-identified gifts", "Gift channel")), fint(r["n"]), fint(r["female"]), fint(r["male"]), fint(r["mixed"]), fint(r["uncoded"]),
                     fdec(r["pct_uncoded"]), cell(r["pct_f_coded"]), fdec(r["pct_f_lower"]), fdec(r["pct_f_upper"])])
    write_table("rev_supp_gender.tex", "Gender coding of records by department, acquisition decade and case.",
                ["", "Records", "Women", "Men", "Mixed", "Uncoded", r"\% unc.", r"\% female", "Lower", "Upper"],
                rows, r"p{0.30\textwidth}rrrrrrrrr", "tab:s_gender", resize=True,
                note=r"`Women' and `Men' count records whose gender-coded makers are all women or all men, and `\% unc.' is the share of records with no binary code. `\% female' is women-only records over women-only plus men-only records. `Lower' counts every mixed and uncoded record as not female and `Upper' counts every one as female, over all records.")
    rows = [[tex_escape(k_), fint(r["records"]), fdec(r["pct"])] + [fint(r[d]) for d in MAIN_DEPARTMENTS] for k_, r in gkind.iterrows()]
    write_table("rev_supp_gender_kinds.tex", "What the records without a gender code name.",
                ["Kind of maker", "Records", r"\%", r"A\&D", r"D\&P", "M\\&P", r"P\&S", "Photo."], rows, "lrrrrrrr", "tab:s_gender_kinds",
                note=r"A\&D Architecture \& Design, D\&P Drawings \& Prints, M\&P Media and Performance, P\&S Painting \& Sculpture.")
    rows = [[tex_escape(r["rule"]), fint(round(r["numerator"])), fint(round(r["denominator"])), fdec(r["pct_f"], 2)] for _, r in galt.iterrows()]
    write_table("rev_supp_gender_rules.tex", "Female share of the gift channel under alternative counting rules.",
                ["Rule", "Numerator", "Denominator", r"\% female"], rows, r"p{0.62\textwidth}rrr", "tab:s_gender_rules")
    rows = [[tex_escape(m[0].upper() + m[1:]), fint(v) if float(v) == int(float(v)) and float(v) > 100 else (fint(v) if float(v) == int(float(v)) else fdec(v, 2))] for m, v in gval.items()]
    write_table("rev_supp_gender_validation.tex", "MoMA's artist-gender field compared with Wikidata for artists linked to a Wikidata item.",
                ["", "Value"], rows, r"p{0.7\textwidth}r", "tab:s_gender_validation")

    # S: nationality-to-region lookup.
    from coding import _REGION_MEMBERS, BASELINE_WESTERN_REGIONS
    rows = [[r_, "Western" if r_ in BASELINE_WESTERN_REGIONS else "Non-Western", tex_escape(", ".join(sorted(m_)))]
            for r_, m_ in _REGION_MEMBERS.items()]
    write_table("rev_supp_region_lookup.tex", "Nationality terms in MoMA's metadata assigned to each region, and the region's side under the baseline rule.",
                ["Region", "Baseline", "Nationality terms"], rows, r"p{0.2\textwidth}p{0.11\textwidth}p{0.6\textwidth}", "tab:s_lookup",
                size=r"\footnotesize")

    # S: region composition and schemes.
    short = {"North America": "N. Am.", "Western Europe": "W. Eur.", "Australia and New Zealand": "Aus./NZ",
             "Eastern Europe and former USSR": "E. Eur.", "Latin America and Caribbean": "Lat. Am.", "East Asia": "E. Asia",
             "South and Southeast Asia": "S./SE Asia", "Middle East and North Africa": "MENA", "Sub-Saharan Africa": "Sub-Sah. Afr.", "Indigenous peoples of North America and Australia": "Indig."}
    CASE_ABBR = {"Kleiner, Bell & Co.": "K", "Rockefeller lineage": "R", "Jean Pigozzi": "P", "Peter J. Cohen": "C",
                 "Judith Rothschild Foundation": "J", "Agnes Gund": "G"}
    for name, tab, cap in (("rev_supp_region_records.tex", regrec, "Records by region of the artist's nationality."),
                           ("rev_supp_region_artists.tex", regart, "Distinct artists by region of nationality.")):
        cols = [c for c in tab.columns if c != "n"]
        order = [u_ for u_ in tab.index if u_ not in CASE_ABBR] + [u_ for _, u_ in CASES if u_ in tab.index]
        head = ["Region"] + [("Gift channel" if u_ not in CASE_ABBR else CASE_ABBR[u_]) for u_ in order]
        rows = [[tex_escape(c)] + [fint(tab.loc[u_, c]) for u_ in order] for c in cols]
        rows.append(r"\midrule")
        rows.append(["Total"] + [fint(tab.loc[u_, "n"]) for u_ in order])
        note = "K Kleiner, Bell \\& Co., R Rockefeller lineage, P Pigozzi, C Cohen, J Judith Rothschild Foundation, G Gund."
        if name == "rev_supp_region_records.tex":
            note += (" `Mixed regions' counts records whose makers fall in more than one of the ten regions. "
                     f"Of these, {fint(museum['n_mixed_origin'])} have makers on both sides of the baseline Western and non-Western rule.")
        write_table(name, cap, head, rows, r"p{0.40\textwidth}" + "r" * len(order), "tab:s_" + name[9:-4], note=note)
    rows = [r"\multicolumn{8}{@{}l}{\emph{(a) Non-Western share of records (\%)}} \\"]
    for s_, r in schemes.iterrows():
        rows.append([tex_escape(s_), fdec(r["museum_pct_nw"])] + [fdec(r[f"{u} | pct_nw"]) for _, u in CASES])
    rows.append(r"\midrule" + "\n" + r"\multicolumn{8}{@{}l}{\emph{(b) Contribution to the gift-channel share (points); first column the six cases jointly}} \\")
    for s_, r in schemes.iterrows():
        rows.append([tex_escape(s_), fdec(r["six_contribution_pp"], 2)] + [csigned(r[f"{u} | contribution_pp"]) for _, u in CASES])
    write_table("rev_supp_region_schemes.tex",
                "Non-Western shares and contributions under the baseline rule and five alternative regional rules.",
                ["Rule", "Channel / six", "K", "R", "P", "C", "J", "G"], rows,
                r"p{0.38\textwidth}rrrrrrr", "tab:s_schemes",
                note="In panel (a) the second column is the gift-channel share and in panel (b) the contribution of the six cases together. "
                     "K Kleiner, Bell \\& Co., R Rockefeller lineage, P Pigozzi, C Cohen, J Judith Rothschild Foundation, G Gund.")

    # S: six cases, dependence among records.
    rows = []
    for p, u in CASES:
        r = six.loc[u]
        rows.append([CASE_TEX[u], fint(r["n_clusters"]), fdec(r["top_cluster_share"]),
                     cell(r["pct_f"]), f"[{cell(r['f_lo'])}, {cell(r['f_hi'])}]", cell(r["pct_f_wo_top"]),
                     cell(r["pct_nw"]), f"[{cell(r['nw_lo'])}, {cell(r['nw_hi'])}]", cell(r["pct_nw_wo_top"]),
                     fdec(r["exp_f"]), fdec(r["exp_nw"])])
    rows = []
    for key, lab in (("f", "female"), ("nw", "non-Western")):
        rows.append(r"\multicolumn{7}{@{}l}{\emph{(" + ("a" if key == "f" else "b") + ") " + lab.capitalize().replace("Non-western", "Non-Western") + r" share of records}} \\")
        for p, u in CASES:
            r = six.loc[u]
            rows.append([CASE_TEX[u], fint(r["n_clusters"]), fdec(r["top_cluster_share"]), cell(r[f"pct_{key}"]),
                         f"[{cell(r[f'{key}_lo'])}, {cell(r[f'{key}_hi'])}]", cell(r[f"pct_{key}_wo_top"]), fdec(r[f"exp_{key}"])])
        if key == "f":
            rows.append(r"\midrule")
    write_table("rev_supp_cases_dependence.tex",
                "The six cases: record shares, artist-resampling intervals, shares without the largest artist, and matched expectations.",
                ["Donor", "Clusters", r"Largest (\%)", r"Share (\%)", "Interval", "Without largest", "Expected"], rows, "lrrrrrr", "tab:s_dependence",
                note="Intervals are 2.5th and 97.5th percentiles over 5{,}000 resamples of the donor's artist clusters. "
                     "`Expected' is the share of other donors' gifts in the same departments and decades. `Largest' is the share of the donor's records that credit its largest identified artist, "
                     "and `Without largest' the share after those records are removed.")

    # S: sub-eras.
    rows = [[CASE_TEX[r["case"]], f"{int(r['start'])}--{int(r['end'])}" if r["start"] != r["end"] else str(int(r["start"])),
             fint(r["n"]), fint(r["n_artists"]), cell(r["pct_f"]), fint(r["g_valid"]), cell(r["pct_nw"]), fint(r["o_valid"]),
             fint(r["n_fund_led"])] for _, r in sube.iterrows()]
    write_table("rev_supp_suberas.tex", "Sub-eras of the six cases.",
                ["Donor", "Years", "Records", "Artists", r"\% female", "coded", r"\% non-W.", "coded", "Fund-led"], rows,
                "llrrrrrrr", "tab:s_suberas", resize=True,
                note="A year opens a new sub-era when the year-on-year change in the donor's record count is at least 1.5 standard deviations "
                     "from the donor's mean change and at least the larger of 10 records and 10\\% of the donor's peak year. At most three breaks are kept.")

    # S: Rockefeller.
    rows = [[tex_escape(pn), fint(r["n"]), fint(r["n_artists"]), f"{int(r['era_start'])}--{int(r['era_end'])}",
             cell(r["pct_f"]), cell(r["pct_nw"]), csigned(r["contrib_f"]), csigned(r["contrib_nw"])] for pn, r in rock.iterrows()]
    write_table("rev_supp_rockefeller.tex", "The Rockefeller family-line aggregate taken apart.",
                ["Part", "Records", "Artists", "Years", r"\% female", r"\% non-W.", "Contrib. F", "Contrib. NW"], rows,
                "lrrlrrrr", "tab:s_rockefeller", resize=True,
                note="Contributions are to the gift-channel record shares (percentage points).")
    rows = [[tex_escape(u), fint(r["n"]), f"{int(r['start'])}--{int(r['end'])}"] for u, r in rock_strings.sort_values("n", ascending=False).iterrows()]
    write_table("rev_supp_rockefeller_strings.tex", "Credit-line donor strings aggregated as the Rockefeller lineage.",
                ["Donor string", "Records", "Years"], rows, r"p{0.6\textwidth}rl", "tab:s_rockefeller_strings")

    thr_label = {"min_volume": "Minimum records, volume rule", "dept_single": "Top-department share, single-department rules (\\%)", "near_gap": "Allowed excess over gift-channel share (points)",
                 "era_years": "Active era, founding-lineage rule (years)", "top_artist": "Top-artist share, single-artist rule (\\%)",
                 "placeholder": "Unidentified-maker share, anonymous archive (\\%)", "peak_year": "Peak-year share, institutional block gift (\\%)",
                 "dept_spread": "Top-department share, cross-departmental rule (\\%)", "artist_gap": "Artist-level gap, cross-departmental rule (points)",
                 "max_top_artist": "Largest artist share, volume rule (\\%)"}
    rows = [[thr_label.get(r["threshold"], tex_escape(r["threshold"])), fdec(r["value"], 1).rstrip("0").rstrip("."), "yes" if r["default"] else "",
             fint(r["anchors_own_pattern"]), fint(r["clean"]), fint(r["ambiguous"]), fint(r["unclassified"]), fint(r["types_populated"])] for _, r in typsens.iterrows()]
    write_table("rev_supp_typology_sensitivity.tex", "Classification of the non-anchor units when one threshold is moved at a time.",
                ["Threshold", "Value", "Default", "Anchors kept", "One", "Several", "None", "Patterns used"], rows, r"p{0.36\textwidth}rlrrrrr",
                "tab:s_typsens", size=r"\footnotesize")
    rows = [[fint(r["k"]), fdec(r["kmeans_silhouette"], 3), fdec(r["ward_silhouette"], 3), fdec(r["shuffled_mean"], 3),
             fdec(r["shuffled_p95"], 3), fint(r["kmeans_smallest_cluster"])] for _, r in clus.iterrows()]
    write_table("rev_supp_clustering.tex", "Cluster structure of the donor feature space.",
                ["$k$", "$k$-means silhouette", "Ward silhouette", "Shuffled mean", "Shuffled 95th pct.", "Smallest cluster"], rows,
                "rrrrrr", "tab:s_clustering",
                note="Features are standardized log record count, active era, peak-year share, top-department share, top-artist share, "
                     "share of unidentified-maker records, female share and non-Western share. The shuffled reference permutes each feature independently across donors (100 draws). "
                     f"The adjusted Rand index between the six-cluster $k$-means solution and the pattern rules, over units matching exactly one pattern, is {clus['ari_rules_vs_k6_clean_units'].iloc[0]:.2f}.")

    # S: concentration and size classes.
    rows = []
    for m, r in conc.iterrows():
        shuf = "" if pd.isna(r.get("shuffled_mean")) else f"{fint(r['shuffled_mean'])} [{fint(r['shuffled_lo'])}, {fint(r['shuffled_hi'])}]"
        t10 = "" if pd.isna(r.get("top10_shuffled_mean")) else f"{fdec(r['top10_shuffled_mean'])} [{fdec(r['top10_shuffled_lo'])}, {fdec(r['top10_shuffled_hi'])}]"
        rows.append([tex_escape(m.capitalize().replace("Non-western", "Non-Western").replace("non-western", "non-Western")), fint(r["n_flagged"]), fint(r["units_supplying_any"]), fint(r["units_for_half"]), shuf,
                     fdec(r["top10_share"]), t10])
    write_table("rev_supp_concentration.tex", "Concentration of female and non-Western records and artists across donor units.",
                ["", "Count", "Units supplying any", "Units for half", "Label-shuffled", r"Ten leading sources (\%)", "Label-shuffled"],
                rows, "lrrrrrr", "tab:s_concentration", resize=True,
                note="Artists are those new to the collection, credited to the donor unit through which they first entered. The shuffled reference permutes the label across artists "
                     "(for records, across artist clusters) and reports the mean and the 2.5th and 97.5th percentiles. `Ten leading sources' are the ten units that supply the most, which differ from the ten largest units of Table~\\ref{tab:s_sizes}.")
    rows = []
    for c_, r in classes.iterrows():
        if c_.startswith("Spearman"):
            continue
        rows.append([c_.capitalize(), fint(r["units"]), fdec(r["pct_records"]), fdec(r["pct_f_records"]), fdec(r["pct_nw_records"]),
                     fdec(r["pct_new_artists"]), fdec(r["pct_new_f_artists"]), fdec(r["pct_new_nw_artists"])])
    write_table("rev_supp_size_classes.tex", "What donor units of different sizes supply, as percent of gift-channel records and of artists new to the collection through a named donor.",
                ["Units with", "Units", "Records", "Female rec.", "Non-W. rec.", "New artists", "New female", "New non-W."],
                rows, "lrrrrrrr", "tab:s_sizes")
    lead = first.sort_values("new_nw", ascending=False).head(8)
    rows = [[tex_escape(u), fint(r["n_records"]), fint(r["new_artists"]), fint(r["new_nw"])] for u, r in lead.iterrows()]
    lead = first.sort_values("new_f", ascending=False).head(8)
    rows2 = [[tex_escape(u), fint(r["n_records"]), fint(r["new_artists"]), fint(r["new_f"])] for u, r in lead.iterrows()]
    both_rows = [[a_[0], a_[2], a_[3], b_[0], b_[2], b_[3]] for a_, b_ in zip(rows, rows2)]
    write_table("rev_supp_first_source.tex", "Donor units through which the most non-Western artists (left) and the most female artists (right) first entered the collection.",
                [r"Donor unit", "New", r"Non-W.", r"Donor unit", "New", "Female"], both_rows,
                r"p{0.33\textwidth}rr@{\hspace{14pt}}p{0.33\textwidth}rr", "tab:s_first", size=r"\footnotesize",
                note="`New' counts the unit's artists who are new to the collection, and the last column of each half how many of them are non-Western or female.")
    rows = [["1929" if r["decade"] == 1920 else str(int(r["decade"])) + "s", fint(r["new_artists"]), fint(r["new_f"]), cell(r["pct_f"]), fint(r["units_with_new_f"]),
             fint(r["new_nw"]), cell(r["pct_nw"]), fint(r["units_with_new_nw"])] for _, r in decades.iterrows()]
    write_table("rev_supp_decades.tex", "Artists first entering the collection through a named-donor gift, by decade.",
                ["Decade", "New artists", "Female", r"\%", "Units with women", "Non-Western", r"\%", "Units with non-W."], rows,
                "lrrrrrrr", "tab:s_decades",
                note="Percentages are of the new artists with a gender code or with a recorded nationality, which are fewer than the new artists in the first column. The last row covers 2020 to 2024.")

    # S: random gifts of the same structure, four designs (summary and per-unit tables).
    share_label = {"female": "Female share", "non-Western": "Non-Western share",
                   "non-Western (Eastern Europe as Western)": "Non-Western share, Eastern Europe as Western"}
    design_label = {"rec_matched": "Record draw, matched", "rec_pooled": "Record draw, pooled",
                    "art_matched": "Artist draw, matched", "art_pooled": "Artist draw, pooled"}
    rows = []
    for (m_, sh_), t_ in nsum.groupby(["museum", "share"], sort=False):
        if rows:
            rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{m_}, {share_label[sh_].lower().replace('eastern europe', 'Eastern Europe').replace('western', 'Western').replace('non-Western', 'non-Western')} "
                    rf"({int(t_['units'].iloc[0])} units)}}}} \\")
        for r in t_.itertuples():
            rows.append([design_label[r.design], fint(r.raw_up), fint(r.raw_down), fint(r.bh_up), fint(r.bh_down),
                         fint(r.units - r.bh_up - r.bh_down), fint(r.cannot_flag_up), fint(r.cannot_flag_down)])
    with open(out("rev_supp_size_null.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n"
                "\\tbl{Donor units flagged by the comparison with random gifts of the same structure, under four designs.}\n"
                "{\\small\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
                " & \\multicolumn{2}{c}{Unadjusted} & \\multicolumn{3}{c}{Adjusted} & \\multicolumn{2}{c}{Cannot be flagged} \\\\\n"
                "Design & Above & Below & Above & Below & Neither & Above & Below \\\\\n\\midrule\n")
        for r in rows:
            f.write((r if isinstance(r, str) else " & ".join(r) + r" \\") + "\n")
        f.write("\\bottomrule\n\\end{tabular}}\n"
                "\\begin{tabnote}`Unadjusted' counts units with a two-sided $p$ below 0.05, and `Adjusted' applies the Benjamini--Hochberg procedure at 5\\% across the units of one share. "
                "A unit `cannot be flagged' in a direction when even the most extreme random gift in that direction (every maker drawing the same label) has an unadjusted $p$ of 0.05 or more. "
                "The record draw gives each maker the label of a record drawn from other donors' gifts and the artist draw the label of a maker drawn from them, "
                "matched to department and decade or pooled over the channel.\\end{tabnote}\n\\label{tab:s_size_null}\n\\end{table}\n")
    def flag_symbols(r):
        out_ = []
        for d_ in NULL_DESIGNS:
            out_.append("$\\cdot$" if r[f"{d_}_q"] >= 0.05 else ("$+$" if r[f"{d_}_dir"] == "up" else "$-$"))
        return "".join(out_)
    def detectable(r):
        up, dn = r["rec_matched_min_p_up"] < 0.05, r["rec_matched_min_p_down"] < 0.05
        return ("a" if up else "") + ("b" if dn else "") or "--"
    nm = nun[nun["museum"] == "MoMA"]
    fem = nm[nm["share"] == "female"].set_index("unit"); nwt = nm[nm["share"] == "non-Western"].set_index("unit")
    allu = sorted(set(fem.index) | set(nwt.index), key=lambda u_: -max(fem["records"].get(u_, 0), nwt["records"].get(u_, 0)))
    def part(t_, u_):
        if u_ not in t_.index:
            return ["--", "--", "", ""]
        r = t_.loc[u_]
        return [fdec(r["obs_share"]), fdec(r["rec_matched_expected"]), flag_symbols(r), detectable(r)]
    rows = []
    for u_ in allu:
        n_ = fem["records"].get(u_, nwt["records"].get(u_))
        rows.append([tex_escape(u_), fint(n_)] + part(fem, u_) + part(nwt, u_))
    head = (" & & \\multicolumn{4}{c}{Female share} & \\multicolumn{4}{c}{Non-Western share} \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(l){7-10}\n"
            "Donor unit & Rec. & Obs. & Exp. & Flags & Det. & Obs. & Exp. & Flags & Det. \\\\\n\\midrule\n")
    with open(out("rev_supp_null_units.tex"), "w", encoding="utf-8") as f:
        f.write("{\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n\\begin{longtable}{@{}p{0.36\\textwidth}rrrlcrrlc@{}}\n"
                "\\caption{Every donor unit in the comparison with random gifts, MoMA, baseline regional rule. "
                "`Obs.' is the unit's record share (\\%) and `Exp.' the mean share of random gifts under the matched record draw. "
                "`Flags' gives the adjusted result under the four designs in the order record matched, record pooled, artist matched, artist pooled "
                "($+$ above, $-$ below, $\\cdot$ not flagged). `Det.' gives the directions in which the matched record draw could flag the unit at all "
                "(a above, b below). In the share columns a dash marks a unit with too few coded records for that share, and in `Det.' a unit that cannot be flagged in either direction. Donor names are given as written in the credit lines.}\\label{tab:s_null_units}\\\\\n\\toprule\n"
                + head + "\\endfirsthead\n\\toprule\n" + head + "\\endhead\n\\bottomrule\n\\endlastfoot\n")
        for r in rows:
            f.write(" & ".join(r) + r" \\" + "\n")
        f.write("\\end{longtable}}\n")

    # S: associations, selection audit, Tate units.
    rows = [[tex_escape(re.sub(r"\|([^|]+)\|", r"absolute \1", m)), fint(r["n_units"]), csigned(r["spearman_rho"]), fdec(r["perm_p"], 3) if r["perm_p"] >= 0.001 else "$<$0.001"] for m, r in assoc.iterrows()]
    write_table("rev_supp_association.tex", "Association between a donor unit's record count and its gap or contribution (units with at least 100 records).",
                ["Measure", "Units", r"Spearman $\rho$", "Permutation $p$"], rows, "lrrr", "tab:s_assoc")
    rows = [[tex_escape(r["axis"]) if r["rank"] == 1 else "", fint(r["rank"]), tex_escape(r["unit"]) + (r"$^{\ast}$" if r["is_case"] else ""),
             fint(r["value"]) if float(r["value"]).is_integer() else fdec(r["value"]), fint(r["n"])] for _, r in sel.iterrows()]
    write_table("rev_supp_selection.tex", "The five most extreme donor units on each selection axis (units with at least 100 records). $^{\\ast}$Selected case.",
                ["Axis", "Rank", "Donor unit", "Value", "Records"], rows, r"p{0.3\textwidth}rp{0.42\textwidth}rr", "tab:s_selection",
                size=r"\footnotesize")

    # =============================================================== figures
    # Figure 1: volume versus gap, all units with 30+ coded records.
    agg = gifts.groupby("Unit")[["is_F", "g_valid", "is_NW", "o_valid"]].sum()
    agg["n"] = sizes
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True)
    labels = {"Kleiner, Bell & Co.": "Kleiner, Bell", "Rockefeller lineage": "Rockefeller", "Jean Pigozzi": "Pigozzi",
              "Peter J. Cohen": "Cohen", "Judith Rothschild Foundation": "Rothschild Fdn.", "Agnes Gund": "Gund",
              "Jack Shear": "Shear", "Patricia Phelps de Cisneros": "Cisneros", "Helen Kornblum": "Kornblum"}
    for ax, (num, den, base, ylab) in zip(axes, (("is_F", "g_valid", museum["pct_f"], "Female share minus gift-channel share (pp)"),
                                                 ("is_NW", "o_valid", museum["pct_nw"], "Non-Western share minus gift-channel share (pp)"))):
        d = agg[agg[den] >= 30].copy()
        d["gap"] = 100 * d[num] / d[den] - base
        small, large = d[d["n"] < 100], d[d["n"] >= 100]
        ax.scatter(small["n"], small["gap"], s=12, c=GRAY[4], edgecolors="none", label="Fewer than 100 records", zorder=2)
        ax.scatter(large["n"], large["gap"], s=24, facecolors="white", edgecolors=GRAY[0], linewidths=0.7,
                   label="At least 100 records", zorder=3)
        ax.axhline(0, color=GRAY[2], linewidth=0.8, zorder=1)
        offsets = FIG1_OFFSETS[num]
        for u, lab in labels.items():
            if u in d.index:
                dx, dy = offsets.get(u, (5, 5))
                ax.annotate(lab, xy=(d.loc[u, "n"], d.loc[u, "gap"]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=7.5, ha="left" if dx >= 0 else "right",
                            arrowprops=dict(arrowstyle="-", color=GRAY[2], linewidth=0.5, shrinkA=0, shrinkB=2))
        ax.set_xscale("log"); ax.set_xlim(right=9000); ax.set_xlabel("Records given (log scale)"); ax.set_ylabel(ylab)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "eps"):
        fig.savefig(out(f"fig_volume_vs_gap.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: record-level versus artist-level contribution, units with 100+ records.
    d = units[units["n"] >= 100].join(loo[["art_contrib_f", "art_contrib_nw"]], how="left").fillna({"art_contrib_f": 0, "art_contrib_nw": 0})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, (xc, yc, title) in zip(axes, (("contrib_f", "art_contrib_f", "Female share"), ("contrib_nw", "art_contrib_nw", "Non-Western share"))):
        ax.scatter(d[xc], d[yc], s=10 + 0.03 * d["n"], facecolors="white", edgecolors=GRAY[0], linewidths=0.7, zorder=3)
        lim = max(d[xc].abs().max(), d[yc].abs().max()) * 1.12
        ax.plot([-lim, lim], [-lim, lim], color=GRAY[3], linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(0, color=GRAY[4], linewidth=0.6, zorder=1); ax.axvline(0, color=GRAY[4], linewidth=0.6, zorder=1)
        for u, (dx, dy) in FIG2_OFFSETS[xc].items():
            if u in d.index:
                ax.annotate(labels[u], xy=(d.loc[u, xc], d.loc[u, yc]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=7.5, ha="left" if dx >= 0 else "right",
                            arrowprops=dict(arrowstyle="-", color=GRAY[2], linewidth=0.5, shrinkA=0, shrinkB=2))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Contribution counted as records (pp)"); ax.set_ylabel("Contribution counted as artists (pp)")
        ax.set_title(title)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "eps"):
        fig.savefig(out(f"fig_record_vs_artist.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------- copy
    os.makedirs(PAPER_OUT, exist_ok=True)
    copied = []
    for fname in sorted(os.listdir(out(""))):
        if fname.startswith(("rev_tab_", "rev_supp_")) or fname in ("rev_numbers.tex", "rev_numbers.csv") \
                or fname.startswith(("fig_volume_vs_gap.", "fig_record_vs_artist.")):
            shutil.copy2(out(fname), os.path.join(PAPER_OUT, fname)); copied.append(fname)
    print(f"{len(MACROS)} macros written; {len(copied)} files copied to {PAPER_OUT}")


if __name__ == "__main__":
    main()

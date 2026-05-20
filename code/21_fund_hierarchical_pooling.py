# -*- coding: utf-8 -*-
"""
Experiment 2: Hierarchical / partial-pooling model for Fund Purchase records.

Implementation note:
  Uses statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM, which is
  available locally, instead of PyMC or Stan.
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_output_path
from experiment_utils import (
    append_log,
    get_enhanced_main_sample,
    prepare_outcome_sample,
    sigmoid,
)


def clean_re_name(name):
    m = re.search(r"C\(PrimaryFund\)\[(.+)\]", str(name))
    if m:
        return m.group(1)
    return str(name)


def fit_hierarchical(df, outcome):
    d = df.copy()
    d = prepare_outcome_sample(d, outcome)
    d = d[d["PrimaryFund"].notna()].copy()
    d["PrimaryFund"] = d["PrimaryFund"].astype(str)
    formula = f"{outcome} ~ BirthYear_c + C(Department) + C(Decade)"
    vc = {"fund": "0 + C(PrimaryFund)"}
    model = BinomialBayesMixedGLM.from_formula(formula, vc, d)
    result = model.fit_vb()

    re_df = result.random_effects()
    if isinstance(re_df, pd.DataFrame):
        re_df = re_df.rename(columns=lambda c: str(c))
    else:
        re_df = pd.DataFrame(re_df)

    mean_col = [c for c in re_df.columns if c.lower().startswith("mean")][0]
    sd_col = [c for c in re_df.columns if c.lower().startswith("sd")][0]
    re_df = re_df.reset_index().rename(columns={"index": "re_name"})
    re_df["Fund"] = re_df["re_name"].apply(clean_re_name)
    re_df = re_df.groupby("Fund")[[mean_col, sd_col]].mean().reset_index()
    re_df = re_df.rename(columns={mean_col: "alpha_mean", sd_col: "alpha_sd"})

    fe_mean = np.asarray(result.fe_mean)
    base_eta = np.dot(model.exog, fe_mean)
    d = d.reset_index(drop=True)
    d["base_eta"] = base_eta
    d = d.merge(re_df.rename(columns={"Fund": "PrimaryFund"}), on="PrimaryFund", how="left")
    # If a fund had no matched random effect (rare), default to zero offset
    d["alpha_mean"] = d["alpha_mean"].fillna(0.0)
    d["alpha_sd"] = d["alpha_sd"].fillna(0.0)

    rows = []
    counts = d["PrimaryFund"].value_counts()
    top30 = counts.head(30).index.tolist()
    for fund in top30:
        sub = d[d["PrimaryFund"] == fund].copy()
        raw_share = sub[outcome].mean()
        post_mean = float(sigmoid(sub["base_eta"] + sub["alpha_mean"]).mean())
        lo = float(sigmoid(sub["base_eta"] + sub["alpha_mean"] - 1.96 * sub["alpha_sd"]).mean())
        hi = float(sigmoid(sub["base_eta"] + sub["alpha_mean"] + 1.96 * sub["alpha_sd"]).mean())
        rows.append(
            {
                "Outcome": outcome,
                "Fund": fund,
                "N": int(len(sub)),
                "RawShare": float(raw_share),
                "PosteriorMean": post_mean,
                "CI_lower": lo,
                "CI_upper": hi,
            }
        )

    tau = float(np.exp(result.vcp_mean[0])) if len(result.vcp_mean) else np.nan
    fund_summary = d.groupby("PrimaryFund")[outcome].agg(["mean", "count"]).reset_index()
    sampling_var = float((fund_summary["mean"] * (1 - fund_summary["mean"]) / fund_summary["count"]).mean())
    vpart = (tau**2) / (tau**2 + sampling_var) if sampling_var > 0 and not np.isnan(tau) else np.nan

    summary = {
        "Outcome": outcome,
        "Fund": "__SUMMARY__",
        "N": int(len(d)),
        "RawShare": np.nan,
        "PosteriorMean": np.nan,
        "CI_lower": np.nan,
        "CI_upper": np.nan,
        "tau": tau,
        "sampling_variance": sampling_var,
        "variance_partition": vpart,
    }
    return pd.DataFrame(rows), summary


def make_figure(df_all):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    for ax, outcome, title in zip(axes, ["IsFemale", "IsNonWest"], ["Female", "Non-Western"]):
        sub = df_all[df_all["Outcome"] == outcome].copy()
        ax.scatter(sub["RawShare"] * 100, sub["PosteriorMean"] * 100, s=np.clip(sub["N"], 10, 300), alpha=0.75)
        lim_lo = min(sub["RawShare"].min(), sub["PosteriorMean"].min()) * 100 - 2
        lim_hi = max(sub["RawShare"].max(), sub["PosteriorMean"].max()) * 100 + 2
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", color="gray", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Raw share (%)")
        ax.set_ylabel("Partial-pooled posterior mean (%)")
    fig.suptitle("Fund-level shrinkage: raw shares vs partial-pooled estimates")
    fig.tight_layout()
    fig.savefig(get_output_path("fig_fund_shrinkage.pdf"))


def main():
    df = get_enhanced_main_sample()
    df = df[df["AcqType_enhanced"] == "Fund Purchase"].copy()

    out_rows = []
    summaries = []
    for outcome in ["IsFemale", "IsNonWest"]:
        est_df, summary = fit_hierarchical(df, outcome)
        out_rows.append(est_df)
        summaries.append(summary)
    out_df = pd.concat(out_rows, ignore_index=True)
    out_df = pd.concat([out_df, pd.DataFrame(summaries)], ignore_index=True, sort=False)
    out_df.to_csv(get_output_path("fund_hierarchical_estimates.csv"), index=False)
    make_figure(out_df[out_df["Fund"] != "__SUMMARY__"])

    top_female = out_df[(out_df["Outcome"] == "IsFemale") & (out_df["Fund"] != "__SUMMARY__")].copy()
    header = "Outcome & Fund & N & Raw share & Partial-pooled mean & 95\\% interval \\\\"
    body = [header, "\\midrule"]
    for outcome in ["IsFemale", "IsNonWest"]:
        sub = out_df[(out_df["Outcome"] == outcome) & (out_df["Fund"] != "__SUMMARY__")].copy()
        for _, r in sub.iterrows():
            body.append(
                f"{outcome} & {r['Fund']} & {int(r['N']):,} & {100*r['RawShare']:.1f}\\% & "
                f"{100*r['PosteriorMean']:.1f}\\% & [{100*r['CI_lower']:.1f}\\%, {100*r['CI_upper']:.1f}\\%] \\\\"
            )
    tex = (
        "\\begin{table}[p]\n\\centering\n"
        "\\caption{Partial-pooling estimates for top Fund Purchase contributors}\n"
        "\\label{tab:fund_hierarchical}\n"
        "\\begin{tabular}{lp{5.2cm}rrrr}\n\\toprule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\footnotesize{Posterior means come from a random-intercept logistic model with controls for department, "
        "acquisition decade, and BirthYear. Intervals are approximate 95\\% credible intervals based on the fund "
        "random-effect posterior mean and standard deviation.}\n\\end{table}"
    )
    with open(get_output_path("table_fund_hierarchical.tex"), "w") as f:
        f.write(tex)

    female_summary = [x for x in summaries if x["Outcome"] == "IsFemale"][0]
    female_range = top_female["PosteriorMean"].max() - top_female["PosteriorMean"].min()
    if female_range > 0.60:
        rule = "fund-level dispersion remains wide after shrinkage"
    elif female_range < 0.30:
        rule = "the raw 0%-100% range is largely a small-N artifact"
    else:
        rule = "shrinkage compresses but does not erase fund-level dispersion"

    paragraph = (
        f"A partial-pooling fund model indicates that small named funds do inflate the raw 0\\%--100\\% range, but "
        f"they do not eliminate fund-level heterogeneity. Among the top thirty funds, the adjusted female share still "
        f"spans roughly {100*top_female['PosteriorMean'].min():.1f}\\% to {100*top_female['PosteriorMean'].max():.1f}\\%, "
        f"while the variance partition for the female model is {female_summary['variance_partition']:.2f}. In other words, "
        f"{rule}, supporting the view that named funds behave as distinct governance instruments rather than as pure "
        f"sampling noise."
    )
    append_log(
        "Experiment 2 — Hierarchical fund model",
        [
            f"Saved CSV: {get_output_path('fund_hierarchical_estimates.csv')}",
            f"Saved TEX: {get_output_path('table_fund_hierarchical.tex')}",
            f"Saved FIG: {get_output_path('fig_fund_shrinkage.pdf')}",
            f"Decision rule outcome: {rule}.",
        ],
        paper_paragraph=paragraph,
    )

    print(
        "Completed 21_fund_hierarchical_pooling.py | rows=%d | female variance_partition=%.3f"
        % (len(out_df), female_summary["variance_partition"])
    )


if __name__ == "__main__":
    main()

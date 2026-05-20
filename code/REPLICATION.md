# Replication guide

Replication for *Collections grow by volume but diversify by exception: donor
biographies in MoMA's gift channel, 1929-2024* (Museum Management and Curatorship).

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
```

## Run the pipeline

```bash
cd /home/jhbae/moma
python code/run_all.py
```

All results are written to `output/` (regenerated; not tracked in git). The
scripts read from `data/` and resolve paths relative to the project root.

## Data inputs (`data/`)

- `Artworks_full.csv` — raw MoMA collection export (public source:
  https://github.com/MuseumofModernArt/collection). Step 01 reads this. See
  `data/README.md` for how to download a fresh copy.
- `processed_moma_data.csv` — the preprocessed analysis table. **Not tracked in
  git** (large, derived); generate it with `python code/01_data_preprocessing.py`.
- `tate/processed_tate_data.csv`, `tate/artwork_data.csv`, `tate/artist_data.csv`
  — Tate sample for the cross-museum check (public source:
  https://github.com/tategallery/collection).

`N` and tabulated counts depend on the MoMA data snapshot. The paper's figures
correspond to the snapshot in `data/`; a different download date can yield
slightly different totals.

## Script-to-paper mapping

Run order is encoded in `run_all.py`. Each script writes to `output/`.

| Script | Produces | Paper element |
| --- | --- | --- |
| `01_data_preprocessing.py` | `data/processed_moma_data.csv` | sample construction (§3.1) |
| `32_donor_exploratory_profile.py` | `table_donor_exploratory_top20.tex` | Table 1 |
| `34_donor_biographies.py` | `donor_biography_profile.csv`, `cumulative_*.csv`, top-artist CSVs | §4 biographies, Table 2 |
| `35_donor_cross_comparison.py` | `donor_dept_footprint.csv`, `donor_demographic_gap.csv`, `six_donor_joint_counterfactual.csv` | §5 cross-donor, joint counterfactual |
| `36_donor_figures.py` | `fig_donor_*.{eps,pdf}`, `fig_six_donor_jointcf.{eps,pdf}` | Figures 1-5 |
| `41_strong_accept_analyses.py` | `fig_volume_vs_gap.{eps,pdf}`, `population_decorrelation.csv`, `catalogue_oos_*.csv` | §5.1 population decorrelation, Figure 6, §5.2 OOS validation |
| `40_concentration_canonical_recompute.py` | `concentration_canonical_recompute.csv` | §3.1 donor concentration |
| `33/37/39_*` (Tate) | `tate_*_v2.csv` | §5.3 Tate cross-museum check |
| `38_phase_j_verifications.py` | `symmetric_boundary_check.tex`, arithmetic checks | Table (symmetric boundary check), §5.3 |

Shared helpers: `config.py` (paths, baselines), `donor_bio_utils.py`,
`experiment_utils.py`, `tate_utils.py`, `utils.py`.

## Known limitation

`table_russian_soviet_sensitivity.tex` (the Russian/Soviet recoding sensitivity
table in the paper) is assembled by hand from the per-donor non-Western shares
under the baseline and alternative codings; there is no single script that emits
that `.tex` file. The underlying shares it tabulates are reproducible from the
biography and counterfactual outputs above (the 17-nationality coding rule is
documented in the paper's supplementary parser specification).

## LaTeX build

The paper source is a separate repository (`moma_paper/`). The figures and
`\input` tables produced here are copied into `moma_paper/output/` for the
LaTeX build; see that repository's `Makefile`.

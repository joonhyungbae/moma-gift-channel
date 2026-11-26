# Replication guide

Replication for *Counting women and non-Western artists in MoMA's named-donor gifts by record and by artist, 1929-2024*
(Museum Management and Curatorship).

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
```

## Run the pipeline

```bash
cd <repository root>
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

Coding rules live in one module, `coding.py`: positional parsing of the
constituent, gender and nationality lists; record-level and artist-level gender
coding; the nationality-to-region lookup with the baseline and alternative
Western / non-Western rules; and the credit-line parser (first-named donor
after `Gift of` / `Bequest of`, the name-first forms `X Gift` / `X Bequest`,
suffix retention, dedication and exchange clauses, co-donor / fund-led /
by-exchange flags). The supplement's Sections S1-S3 describe the same rules.

| Script | Produces | Paper element |
| --- | --- | --- |
| `01_data_preprocessing.py` | `data/processed_moma_data.csv` | sample construction (Section 3.1) |
| `50_rev_levels_baselines.py` | `rev_units.csv`, `rev_six_cases.csv`, `rev_museum.csv`, `rev_joint_counterfactual.csv`, `rev_volume_gap_association.csv`, `rev_suberas.csv` | records and artists, gap / contribution / matched gap, cluster resampling (Sections 3.3, 4, 5.1; Table 1) |
| `51_rev_coding_audits.py` | `rev_gender_*.csv`, `rev_region_*.csv`, `rev_rockefeller*.csv`, `rev_selection_audit.csv` | gender and origin coding (Section 3.2), regional rules, Rockefeller disaggregation, case selection (Section 3.4) |
| `52_rev_typology.py` | `rev_typology_*.csv` | six gift patterns, threshold sensitivity, cluster structure (Section 5.3; Table 2) |
| `53_rev_tate_patterns.py` | `rev_tate_units.csv`, `rev_tate_vs_moma_patterns.csv` | Tate comparison (Section 5.4; Table 3) |
| `54_rev_gender_validation.py` | `rev_gender_validation*.csv` | Wikidata check of the gender field (Section 3.2). Reads `data/wikidata_gender_cache.csv`; pass `--offline` to forbid network access |
| `55_rev_artist_concentration.py` | `rev_artist_*.csv`, `rev_first_entry_overview.csv`, `rev_ten_largest_decade_matched.csv`, `rev_rothschild_origin_bound.csv`, `rev_new_artists_by_decade.csv` | records versus artists, artists new to the collection by first accession in any channel, concentration, Rothschild bound (Section 5.2) |
| `57_rev_size_null.py` | `rev_null_units.csv`, `rev_null_summary.csv` | supplementary check: each large donor against random gifts of the same structure, four designs (record or artist draw, matched or pooled), with the number of makers and the smallest attainable p-value per unit (supplement Section S4) |
| `56_rev_paper_assets.py` | `rev_numbers.tex`, `rev_tab_*.tex`, `rev_supp_*.tex`, `fig_volume_vs_gap.*`, `fig_record_vs_artist.*` | every number, table and figure in the article and supplement |
| `58_rev_parser_audit_sample.py` (not in `run_all.py`) | `parser_audit_sample.csv` | fixed random sample of 300 records (200 inside the gift sample, 100 outside) with blank columns for checking the credit-line parser by hand |
| `32`-`41` (earlier scripts) | donor profiles, named-donor Tate check | named-donor recurrence sentence in Section 5.4; otherwise superseded by 50-57 |

Shared helpers: `config.py`, `coding.py`, `rev_utils.py`, `donor_bio_utils.py`,
`experiment_utils.py`, `tate_utils.py`, `utils.py`.

Every data-derived number in the article is a LaTeX macro defined in
`rev_numbers.tex`, so the text cannot drift from the pipeline.

## LaTeX build

The paper source is a separate repository (`moma_paper/`). Step 56 copies the number macros, tables and figures into `moma_paper/output/`
for the LaTeX build; see that repository's `Makefile`.

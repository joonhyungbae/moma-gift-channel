# Data

The analysis reads from this folder. Paths are resolved by `code/config.py`.

## MoMA collection (required)

The pipeline needs MoMA's collection export. This repository ships the exact
snapshot used in the paper as `Artworks_full.csv`, so no download is required
for byte-exact reproduction. To refresh from the public source instead:

1. Go to the MoMA collection repository:
   https://github.com/MuseumofModernArt/collection
2. Download `Artworks.csv` (use the web "Download raw file" button, or
   `git clone` the repository; if `Artworks.csv` arrives as a small Git LFS
   pointer, run `git lfs pull` to fetch the real file).
3. Place it here as `data/Artworks.csv`.

`code/config.py` uses `data/Artworks.csv` when present and otherwise falls back
to `data/Artworks_full.csv`. A file under 1 KB is treated as an LFS pointer and
skipped. `data/Artists_full.csv` is the matching artist table.

`processed_moma_data.csv` is **not** tracked in git (it is large and derived).
Generate it with:

```bash
python code/01_data_preprocessing.py     # writes data/processed_moma_data.csv
```

`N` and tabulated counts depend on the snapshot; a newer MoMA export can yield
slightly different totals than the paper.

## Tate collection (cross-museum check)

`tate/processed_tate_data.csv`, `tate/artwork_data.csv`, and `tate/artist_data.csv`
are derived from the public Tate collection
(https://github.com/tategallery/collection) and are shipped here because the
cross-museum check (paper §5.3) reads them directly.

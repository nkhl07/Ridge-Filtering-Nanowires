# Ridge Filtering for SEM Nanowire Segmentation

Per-pixel wire masks and per-wire measurements (count, length, width,
orientation family, junctions) for SEM images of TiS₃ nanowires.

The pipeline has two stages:

1. **Wire masks.** A 17-feature random-forest classifier (texture + ridge
   filters) runs on each frame after it is rescaled to 50k-equivalent scale.
   This is pipeline **v3**. The older classical pipeline (**v2**: sato ridge
   filter + directional closing) is still available with `--classical`.
2. **Wire extraction.** Skeleton → graph → one row per wire, using
   fingerprint-minutiae techniques (crossing numbers, spur pruning, bridge
   merging, collinear stub pairing at junctions).

## Key findings

- **Real nanowires are DARK bodies with bright rims.** 75% of labeled wire
  pixels are darker than the local background. A bright-only threshold gets
  11% recall; `sato(σ=1–3, black_ridges=True)` gets 52%.
- **The labels are hollow ~4 px pen outlines.** They must be hole-filled per
  orientation family before any coverage, area or thickness statistic.
- **Three orientation families:** blue ≈ 0°, green ≈ 60°, red ≈ 123.5°.
  Weight them by instance count, not pixel mass.
- **Scale comes from FEI TIFF metadata** (tag 34682 `PixelWidth`), never from
  the magnification in the filename. Five frames have filename magnifications
  that are 2× off: `12000×_59`, `20000×_58`, `50000×_16`, `65000×_60`,
  `6500×_57`.

Scores against the 2 hand-labeled frames (cross-frame validated, at matched
coverage):

| pipeline | recall | precision | F1 | junction recall |
|---|---|---|---|---|
| v1 sato, thr 0.10 | 0.517 | 0.484 | 0.500 | 0.574 |
| v2 sato thr 0.14 + directional close (9 px, 0/60/123°) | 0.570 | 0.505 | 0.534 | 0.672 |
| **v3 RF, p > 0.70 (default)** | **0.617** | **0.550** | **0.571** | **0.729** |
| v3 RF + directional close (hand-correction variant) | 0.780 | 0.525 | 0.622 | 0.909 |

About 44% of the RF's feature importance comes from local standard deviation:
wires are smooth against a mottled background. sato contributes only ~4%.
`meijering` and a steerable oriented-ridge bank both scored *worse* than sato.

**Validated scale band: rescale 0.625–1.0** (roughly 50k–80k native).
Outside it the masks degrade. Downscaled 100–200k frames over-fire on
background mottle. Upscaled ≤25k frames show wires as bright rods, so the
filter traces their dark halos instead. These frames are flagged in the
output JSON.

## Repository layout

```
scripts/ridge/
  gt_labels.py               parse the hand-labeled GT overlays (shared helper)
  measure_real_stats.py      coverage / thickness / length / orientation stats from GT
  tune_pseudo_mask.py        tune sato/frangi params against GT
  wire_classifier.py         v3 random-forest classifier (trains on first use, ~2 min)
  ridge_nanowire_images2.py  wire masks for every frame in NanowireImages2 (main entry point)
  wire_extract.py            skeleton → graph → per-wire measurements
  prep_handcorrect_frame.py  editable overlay of one in-band frame for hand-correction
  ridge_similar_cropped.py   v1 sato masks on the similar-cropped cache (kept for reproducibility)
slurm/                       SLURM job files for the two heavy stages
logs/                        logs from previous runs
requirements.txt
```

`data/` and `results/` are not tracked. Scripts find them relative to the
repo root (`ROOT = two parents above the script`), so keep the
`scripts/ridge/` layout.

## Data

Place the data under the repo root like this:

```
data/real/
  NanowireImages2/            62 raw FEI SEM TIFFs (e.g. SF251213M_50000×_50.tif)
  OneDrive_1_7-17-2026/       GT: TiS3_80000×_{1,2}.tif + "TiS3_80000×_{1,2} Labeled.tif"
  normalized_cache/           50k-normalized .npy cache  (only for prep_handcorrect_frame.py)
  similar_cropped_cache/      50k-normalized .npy cache  (only for ridge_similar_cropped.py)
```

The two caches are built by the companion CycleGAN project's dataloader. The
main pipeline (`ridge_nanowire_images2.py` → `wire_extract.py`) needs only
`NanowireImages2` and `OneDrive_1_7-17-2026`.

## Setup

Tested with Python 3.13.5.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

On VT ARC, use the cluster Python to build the venv:
`/apps/arch/software/Python/3.13.5-GCCcore-14.3.0/bin/python3.13 -m venv venv`.

The SLURM files use `./venv/bin/python` when it exists. Otherwise they use
`$PYTHON` if set, else `python3` on `PATH`.

## Usage

Run everything from the repo root.

### 1. Wire masks

```bash
sbatch slurm/ridge_full.sbatch                # all 62 frames, v3 RF
sbatch slurm/ridge_full.sbatch --classical    # v2 classical pipeline instead
sbatch slurm/ridge_full.sbatch 50000          # only frames whose name contains '50000'
```

Without SLURM, the equivalent is
`venv/bin/python scripts/ridge/ridge_nanowire_images2.py [--classical] [filter]`.
The first run trains and caches the RF at
`results/wire_classifier/rf_v3.joblib`.

Outputs go to `results/ridge_nanowire_images2/`:
- `<name>_50k_pseudo.png`: binary mask at 50k-equivalent scale
- `<name>_50k_overlay.png`: the frame in gray with the mask in magenta
- `ridge_coverage.json`: per-frame coverage, scale provenance, band flags and a summary.
  Filtered runs don't rewrite this file.

Memory: the largest frame (4010×) upscales to ~245 Mpx and needs ~55–60 GB,
so the job requests 128 GB. The pipeline is CPU-only.

### 2. Wire extraction

This stage needs the masks and `ridge_coverage.json` from step 1.

```bash
sbatch slurm/wire_extract.sbatch                                  # all in-band frames
sbatch slurm/wire_extract.sbatch --validate-gt                    # + measurement-level GT comparison
sbatch slurm/wire_extract.sbatch --frames SF251213M_65000×_24     # selected frames
```

Useful flags (see `--help` for all ~30 tuning parameters):
- `--all-bands`: also process frames outside the validated scale band.
  Raise `--mem` if you use it.
- `--out DIR`: write somewhere other than `results/wire_extract/`.
- `--no-junction-close`: skip directional closing before skeletonizing.

Outputs go to `results/wire_extract/`, one folder per frame:
- `wires.csv`: one row per wire chain
- `summary.json`: counts, stats, QC and the full parameter set
- `<frame>_instances.png`: instance-colored skeleton with junction and endpoint markers
- `<frame>_families.png`: orientation-family overlay (blue 0° / green 60° / red 123.5°)

Across all frames: `extract_summary.json`, plus `gt_validation.json` when
`--validate-gt` is used.

Nothing is split or deleted automatically; suspect chains are flagged and
kept.

### Analysis scripts (run directly, no SLURM)

```bash
venv/bin/python scripts/ridge/measure_real_stats.py      # -> results/real_stats/
venv/bin/python scripts/ridge/tune_pseudo_mask.py        # -> results/real_stats/pseudo_mask_params.json
venv/bin/python scripts/ridge/prep_handcorrect_frame.py  # -> results/handcorrect/
```

## Open items

- **Hand-correct an in-band frame.** Edit
  `results/handcorrect/SF251213M_50000×_50_50k_overlay.png`. It is the
  recall-oriented mask, so mostly erase excess magenta. Save it as
  `SF251213M_50000×_50_50k_corrected.png`. This gives in-band training and
  validation data for retraining the RF, which removes the 80k → in-band
  domain gap. Production coverage on that frame is 14.10%; the coverage
  correction factor is `corrected_coverage / 14.10%`.
- **Counting convention.** Connected components within one colour family.
  This splits junctions between different families and merges contacts within
  the same family. Needs confirmation with Dr. Pham.

## SLURM notes

The job files target VT ARC Falcon: partition `t4_normal_q`, QOS
`fal_t4_normal_short`, account `me_jiechen`. They request a GPU only because
Falcon has no CPU-only partition; the GPU sits idle. On another cluster, edit
the `#SBATCH` header lines.

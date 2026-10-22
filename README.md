# Ridge Filtering for SEM Nanowire Segmentation

Per-pixel wire masks and per-wire measurements (count, length, width,
orientation family, junctions) for SEM images of TiS₃ nanowires.

The pipeline must stay **unsupervised** (decided 2026-10-13). It has two
stages:

1. **Wire masks.** Each frame is rescaled to 50k-equivalent scale, then a
   sato ridge filter runs, the threshold is chosen per image, and directional
   closing is applied at orientations estimated from the same image. This is
   the **label-free** pipeline and the default. The older classical pipeline
   (**v2**: fixed threshold 0.14 + closing at fixed 0/60/123°) is kept as a
   comparison with `--classical`.
2. **Wire extraction.** Skeleton → graph → one row per wire, using
   fingerprint-minutiae techniques (crossing numbers, spur pruning, bridge
   merging, collinear stub pairing at junctions).

**GT is for evaluation only.** The two hand-labeled 80k frames are used only
to *score* masks (`eval_masks.py`, `wire_extract.py --validate-gt`). They are
never used to train, threshold or choose anything. No new manual labels will
be made.

v1 (sato, threshold 0.10, no closing) and v3 (a supervised random forest
trained on the GT frames) were removed on 2026-10-13. Their scripts and
results are archived in `~/ridge-filtering-archive/` (see Archive below).

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
- **Validated scale band: rescale 0.625–1.0** (roughly 50k–80k native).
  Outside it the masks degrade. Downscaled 100–200k frames over-fire on
  background mottle. Upscaled ≤25k frames show wires as bright rods, so the
  filter traces their dark halos instead. These frames are flagged in the
  output JSON.
- **Texture separates wires from background.** In the removed v3 random
  forest, local standard deviation carried 46.5% of the feature importance:
  wires are smooth against a mottled background. sato contributed only ~4%.
  This is a historical lesson for the planned U-Net, not a current pipeline
  component. `meijering` and a steerable oriented-ridge bank both scored
  *worse* than sato.

Scores: the old v2/v3 numbers (recall/precision/F1/junction recall tables in
earlier versions of this README) were hardcoded literals whose generating
code was lost, so they are unverified. Current scores come from
`results/eval/eval_masks.json`, written by `eval_masks.py` (see Evaluation
below).

## Label-free pipeline: what is per image and what is a prior

- **Per image (no labels):** the threshold (`skimage` `threshold_<method>`
  on the min-max normalised sato response) and up to 3 closing angles
  (orientation-histogram peaks of the above-threshold pixels; fewer when
  fewer families are visible).
- **Declared priors, fixed in advance:** 50k-equivalent scale, sato σ = 1–3
  with `black_ridges=True`, 9 px closing kernels, 3 orientation families,
  and `otsu` as the default threshold method. Downstream, `wire_extract.py`
  also uses the family centers 0/60/123.5° (`--family-centers`), but only to
  assign families and set the `off_family` QC flag. They do not affect the
  mask. The method was chosen before
  any GT evaluation. Do not switch it because another one scores better on
  GT.
- **Provenance of the priors:** σ = 1–3 and the 9 px closing length are
  label-era values. σ was tuned for v1, and the close length was compared on
  the GT frames. The `wire_extract.py` family centers 0/60/123.5° were
  measured from the GT labels (see Key findings). All three are frozen as
  priors and never re-tuned, so the evaluation is not fully GT-independent
  for these constants.

`ridge_coverage.json` records the per-frame `lf_method`, `lf_thr` and
`lf_angles_deg`, with the priors under `"priors"`. No GT scores are written
there.

## Repository layout

```
scripts/ridge/
  gt_labels.py               parse the hand-labeled GT overlays (shared helper)
  measure_real_stats.py      coverage / thickness / length / orientation stats from GT
  ridge_nanowire_images2.py  wire masks for every frame in NanowireImages2 (main entry point)
  labelfree.py               label-free mask (per-image threshold + closing angles)
  test_labelfree.py          synthetic-only tests for labelfree.py
  eval_masks.py              score masks against the 2 GT frames (evaluation only)
  wire_extract.py            skeleton → graph → per-wire measurements
slurm/
  ridge_full.sbatch          wire masks, all 62 frames
  eval_masks.sbatch          mask evaluation against GT
  wire_extract.sbatch        wire extraction
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
```

The pipeline (`ridge_nanowire_images2.py` → `wire_extract.py`) needs only
these two folders.

## Setup

Tested with Python 3.13.5.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

On VT ARC, use the cluster Python to build the venv:
`/apps/arch/software/Python/3.13.5-GCCcore-14.3.0/bin/python3.13 -m venv venv`.

The SLURM files use `$PYTHON` if set, else `./venv/bin/python` when it
exists, else `python3` on `PATH`.

## Usage

Run everything from the repo root. Don't run the pipeline on the login node.

### 1. Wire masks

```bash
sbatch slurm/ridge_full.sbatch                  # all 62 frames, label-free -> results/ridge_labelfree/
sbatch slurm/ridge_full.sbatch --lf-method li   # other threshold: otsu|li|triangle|yen|isodata
sbatch slurm/ridge_full.sbatch --classical      # v2 comparison -> results/ridge_v2/
sbatch slurm/ridge_full.sbatch 50000            # only frames whose name contains '50000'
```

Without SLURM, the equivalent is
`venv/bin/python scripts/ridge/ridge_nanowire_images2.py [--classical | --lf-method M] [--out DIR] [filter]`.
Unknown flags are an error.

Outputs (in `results/ridge_labelfree/` or `results/ridge_v2/`):
- `<name>_50k_pseudo.png`: binary mask at 50k-equivalent scale. Both
  pipelines write masks that are already direction-closed.
- `<name>_50k_overlay.png`: the frame in gray with the mask in magenta
- `ridge_coverage.json`: per-frame coverage, scale provenance, band flags and
  a summary, plus the label-free settings. Filtered runs don't rewrite this
  file.

Memory: the job requests 128 GB, sized for the largest frame (4010×), which
upscales to ~245 Mpx. The removed v3 feature stack peaked at ~56 GiB there.
Label-free sato needs less, but its peak hasn't been measured, so the
headroom stays. The pipeline is CPU-only.

### 2. Evaluation (GT, scoring only)

```bash
sbatch slurm/eval_masks.sbatch                                      # all methods
sbatch slurm/eval_masks.sbatch --methods lf_otsu,v2_fixed,v1_sato_thr0.10   # comma-separated subset
```

This scores masks against the 2 GT frames and writes
`results/eval/eval_masks.json`. The methods:

- `lf_otsu` is the pre-declared label-free default.
- The other `lf_*` methods and two ablations are reported next to it. They
  are never used to pick the default.
- `v2_fixed` is the v2 comparison.
- `v1_sato_thr0.10` is the **harness self-check (removed pipeline)**. It
  re-scores v1 and compares the result with
  `results/real_stats/pseudo_mask_params.json`.

If the self-check fails (`HARNESS CHECK FAIL`, exit status 3), don't trust
the rest of the output. A `zone convention` line under it compares v1's
junction recall with the 0.574 in `README_v3.md` sec. 6.2 of the CycleGAN
project (`~/Unsupervised-Nanowire-Segmentation-using-CycleGAN-Architecture-2/`,
not in this repo). WARN means the junction-zone
definition may differ. A contract check confirms that the label-free sato
response is identical to v2's. Leave `v1_sato_thr0.10` in any `--methods`
list, or the self-check doesn't run.

Quick check without data: `venv/bin/python scripts/ridge/eval_masks.py --selftest`
(synthetic frames only).

### 3. Wire extraction

This stage needs the masks and `ridge_coverage.json` from step 1. It needs no
special flags for label-free masks.

```bash
sbatch slurm/wire_extract.sbatch                                  # all in-band frames
sbatch slurm/wire_extract.sbatch --validate-gt                    # GT validation only (no production frames unless --frames is given)
sbatch slurm/wire_extract.sbatch --frames SF251213M_65000×_24     # selected frames
sbatch slurm/wire_extract.sbatch --masks-dir results/ridge_v2 --out results/wire_extract_v2
```

Both mask pipelines already apply directional closing. `wire_extract.py`
reads `ridge_coverage.json`, logs that the masks are closed, and skips its
own junction closing.

Useful flags (see `--help` for all ~30 tuning parameters):
- `--masks-dir DIR`: default `results/ridge_labelfree/`.
- `--out DIR`: default `results/wire_extract/`.
- `--all-bands`: also process frames outside the validated scale band.
  Raise `--mem` if you use it.
- `--junction-close`: opt in to pre-skeleton closing, for unclosed masks
  only. On label-free masks it uses each frame's `lf_angles_deg`; otherwise
  it uses `--close-angles` (default 0/60/123). With `--validate-gt`, both GT
  frame masks are closed at that GT frame's own label-free angles unless
  `--close-angles` is given, whatever `--masks-dir` holds. A frame with no
  estimated angles gets a warning and is left unclosed.
- `--validate-gt`: runs the label-free mask on the two 80k GT frames and
  compares extraction on it with extraction on the GT label masks
  (evaluation only). It uses the masks' `lf_method` when `--masks-dir` holds
  label-free masks, else `DEFAULT_METHOD` (otsu).

Outputs go to `results/wire_extract/`, one folder per frame:
- `wires.csv`: one row per wire chain
- `summary.json`: counts, stats, QC and the full parameter set
- `<frame>_instances.png`: instance-colored skeleton with junction and endpoint markers
- `<frame>_families.png`: orientation-family overlay (blue 0° / green 60° / red 123.5°)

Across all frames: `extract_summary.json`. With `--validate-gt` you also get
`gt_validation.json`, plus the GT-frame label-free masks in
`gt_pipeline/labelfree_<method>/`.

Nothing is split or deleted automatically; suspect chains are flagged and
kept.

### Other scripts (run directly, no SLURM)

```bash
OMP_NUM_THREADS=2 venv/bin/python scripts/ridge/test_labelfree.py   # synthetic tests, safe anywhere
venv/bin/python scripts/ridge/measure_real_stats.py                 # -> results/real_stats/ (reads GT)
```

## Open items

- **Hand-correction is CANCELLED.** No new manual labels will be made. The
  two existing GT frames stay evaluation-only.
- **Counting convention.** Connected components within one colour family.
  This splits junctions between different families and merges contacts within
  the same family. Needs confirmation with Dr. Pham.
- **Next steps:**
  1. Evaluate label-free vs v2 with `eval_masks.sbatch`.
  2. Noise2Void denoising.
  3. Self-training U-Net.

## Archive

`~/ridge-filtering-archive/` holds `v1_v3_results_2026-10-13.tar` with an
md5 manifest and a `README.txt`. The archive contains:

- the v3 model and the v3 masks for all 62 frames
- the v1 similar-cropped masks
- the cancelled hand-correction overlay
- the v3-derived `wire_extract` outputs
- copies of the removed scripts (`wire_classifier.py`, `tune_pseudo_mask.py`,
  `ridge_similar_cropped.py`, `prep_handcorrect_frame.py`)

The removed scripts are also in git history.

## SLURM notes

The job files target VT ARC Falcon: partition `t4_normal_q`, QOS
`fal_t4_normal_short`, account `me_jiechen`. They request a GPU only because
Falcon has no CPU-only partition; the GPU sits idle. On another cluster, edit
the `#SBATCH` header lines.

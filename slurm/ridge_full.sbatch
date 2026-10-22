#!/bin/bash
#SBATCH --job-name=ridge_full
#SBATCH --partition=t4_normal_q
#SBATCH --qos=fal_t4_normal_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ridge_full-%j.out
#SBATCH --error=logs/ridge_full-%j.err
#SBATCH --account=me_jiechen

# Ridge pseudo-masks for all 62 NanowireImages2 full frames.
# Default pipeline: label-free (sato + per-image auto threshold + directional
# closing at image-estimated angles) -> results/ridge_labelfree/. --classical
# runs the v2 comparison pipeline (fixed THR 0.14 + 0/60/123 closing) ->
# results/ridge_v2/. Unsupervised: the GT frames are evaluation-only
# (slurm/eval_masks.sbatch).
# CPU-only — t4 partition because Falcon has no CPU-only partition and t4 is
# the cheapest/least contended; the GPU idles.
# 128G was sized for the 4010x frame, which upscales to ~245 Mpx: the removed
# v3 RF's float64 feature stack peaked ~56 GiB there (32G OOM'd, job 510614).
# Label-free sato needs less but its peak is unmeasured, so the headroom is
# kept. Submit from the repo root; extra args pass through to the script:
#   sbatch slurm/ridge_full.sbatch                      # label-free (default)
#   sbatch slurm/ridge_full.sbatch --lf-method li       # other auto-threshold
#   sbatch slurm/ridge_full.sbatch --classical          # v2 comparison
#   sbatch slurm/ridge_full.sbatch --out results/ridge_lf_test
#   sbatch slurm/ridge_full.sbatch 50000                # only frames matching '50000'
# Unknown flags error.
# Python: $PYTHON if set, else ./venv/bin/python, else python3 on PATH.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/ridge/ridge_nanowire_images2.py ]; then
    echo "error: submit from the repo root (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
mkdir -p logs

if [ -z "$PYTHON" ]; then
    if [ -x venv/bin/python ]; then PYTHON=venv/bin/python; else PYTHON=python3; fi
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON" scripts/ridge/ridge_nanowire_images2.py "$@" || exit $?

echo "ridge_full done"

#!/bin/bash
#SBATCH --job-name=eval_masks
#SBATCH --partition=t4_normal_q
#SBATCH --qos=fal_t4_normal_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_masks-%j.out
#SBATCH --error=logs/eval_masks-%j.err
#SBATCH --account=me_jiechen

# Score wire masks against the 2 hand-labelled GT frames (EVALUATION ONLY --
# nothing in the label-free pipeline may be tuned on these numbers).
# CPU-only (sato + label-free thresholds) -- t4 partition because Falcon has
# no CPU-only partition and t4 is the cheapest/least contended; the GPU idles
# (same rationale as ridge_full.sbatch). The GT frames are only 640x960 at
# 50k-equivalent, so 32G is generous. Submit from the repo root; extra args
# pass through to the script (--methods is comma-separated):
#   sbatch slurm/eval_masks.sbatch                          # all methods
#   sbatch slurm/eval_masks.sbatch --methods lf_otsu,v2_fixed,v1_sato_thr0.10
# v1_sato_thr0.10 is the harness self-check (removed pipeline); leave it in
# the --methods list or the check does not run.
# Exit status 3 = harness self-check or sato contract check FAILED (JSON is
# still written).
# Output: results/eval/eval_masks.json (override with --out).
# Python: $PYTHON if set, else ./venv/bin/python, else python3 on PATH.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/ridge/eval_masks.py ]; then
    echo "error: submit from the repo root (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
mkdir -p logs

if [ -z "$PYTHON" ]; then
    if [ -x venv/bin/python ]; then PYTHON=venv/bin/python; else PYTHON=python3; fi
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON" scripts/ridge/eval_masks.py "$@" || exit $?

echo "eval_masks done"

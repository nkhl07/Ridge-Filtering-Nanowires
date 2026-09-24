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
# CPU-only pipeline (sato + RF classifier) — t4 partition because Falcon has
# no CPU-only partition and t4 is the cheapest/least contended; the GPU idles.
# 128G because the 4010x frame upscales to ~245 Mpx and its float64 feature
# stack peaks ~55-60G (32G OOM'd on it, job 510614). 16 CPUs matches the RF's
# n_jobs=16. Submit from the repo root; extra args pass through to the script:
#   sbatch slurm/ridge_full.sbatch                 # v3 RF pipeline (default)
#   sbatch slurm/ridge_full.sbatch --classical     # v2 sato+dirclose
#   sbatch slurm/ridge_full.sbatch 50000           # only frames matching '50000'
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

#!/bin/bash
#SBATCH --job-name=unet_pseudo
#SBATCH --partition=t4_normal_q
#SBATCH --qos=fal_t4_normal_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/unet_pseudo-%j.out
#SBATCH --error=logs/unet_pseudo-%j.err
#SBATCH --account=me_jiechen

# unet_v1 Phase 1: cleaned pseudo-labels from the label-free ridge teacher for
# the 21 in-band non-GT frames -> results/unet_v1/pseudo_labels/.
# CPU-only (sato + wire_extract graph) -- t4 partition because Falcon has no
# CPU-only partition (same rationale as ridge_full.sbatch); the GPU idles.
# In-band frames are at most 1024x1536 at 50k-equivalent, so 32G is generous.
# Runs the synthetic unit tests first and stops if they fail. Submit from the
# repo root; extra args pass through (e.g. --only 65000).
# Python: $PYTHON if set, else the CycleGAN repo's venv_l40s (torch + smp +
# PyYAML; core packages identical to requirements.txt).
cd "$SLURM_SUBMIT_DIR"
if [ ! -f unet/pseudo_labels.py ]; then
    echo "error: submit from the repo root (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
mkdir -p logs

PYTHON=${PYTHON:-$HOME/Unsupervised-Nanowire-Segmentation-using-CycleGAN-Architecture-2/venv/venv_l40s/bin/python}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON" -m unet.test_pseudo_labels || exit $?
"$PYTHON" -m unet.pseudo_labels --config configs/unet_v1.yaml "$@" || exit $?

echo "unet_pseudo done"

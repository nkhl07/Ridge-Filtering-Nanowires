#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/unet_train-%j.out
#SBATCH --error=logs/unet_train-%j.err
#SBATCH --account=me_jiechen

# unet_v1 Phase 2: train the U-Net student on the v1e pseudo-labels ->
# results/unet_v1/train/<run_name>/. Runs the synthetic unit tests first and
# stops if they fail. Submit from the repo root; extra args pass through:
#   sbatch slurm/unet_train.sbatch --epochs 5 --run-name diag5      # diagnostic
#   sbatch -p l40s_normal_q --qos=fal_l40s_normal_short --time=06:00:00 \
#       slurm/unet_train.sbatch                                      # full run
# 1 h covers the 5-epoch diagnostic; raise --time for the full run.
# Python: $PYTHON if set, else the CycleGAN repo's venv_l40s (torch + smp +
# tensorboard). resnet34 imagenet weights come from ~/.cache/huggingface
# (HF_HUB_OFFLINE=1: no network on compute nodes).
cd "$SLURM_SUBMIT_DIR"
if [ ! -f unet/train_unet.py ]; then
    echo "error: submit from the repo root (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
mkdir -p logs

PYTHON=${PYTHON:-$HOME/Unsupervised-Nanowire-Segmentation-using-CycleGAN-Architecture-2/venv/venv_l40s/bin/python}

export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" -m unet.test_train_unet || exit $?
"$PYTHON" -m unet.train_unet --config configs/unet_v1.yaml "$@" || exit $?

echo "unet_train done"

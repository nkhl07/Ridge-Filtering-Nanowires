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
# n_jobs=16. Extra args pass through to the script:
#   sbatch slurm/ridge_full.sbatch                 # v3 RF pipeline (default)
#   sbatch slurm/ridge_full.sbatch --classical     # v2 sato+dirclose
#   sbatch slurm/ridge_full.sbatch 50000           # only frames matching '50000'
cd /home/nmahanti/ridge-filtering
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

/home/nmahanti/Unsupervised-Nanowire-Segmentation-using-CycleGAN-Architecture-2/venv/venv_l40s/bin/python scripts/ridge/ridge_nanowire_images2.py "$@"

echo "ridge_full done"

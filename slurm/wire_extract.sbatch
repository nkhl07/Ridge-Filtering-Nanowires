#!/bin/bash
#SBATCH --job-name=wire_extract
#SBATCH --partition=t4_normal_q
#SBATCH --qos=fal_t4_normal_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wire_extract-%j.out
#SBATCH --error=logs/wire_extract-%j.err
#SBATCH --account=me_jiechen

# Wire extraction stage: skeleton -> graph -> per-wire measurements (Phase 1).
# CPU-only pipeline -- t4 partition because Falcon has no CPU-only partition
# and t4 is the cheapest/least contended; the GPU idles (same rationale as
# ridge_full.sbatch). Modest resources: in-band frames are at most 1024x1536
# and the heaviest step is the RF inference on the two 640x960 GT frames
# during --validate-gt. Bump --mem if ever run with --all-bands (the 4010x
# frame upscales to ~245 Mpx). Extra args pass through to the script:
#   sbatch slurm/wire_extract.sbatch                # all in-band frames
#   sbatch slurm/wire_extract.sbatch --validate-gt --frames SF251213M_65000×_24
#                                                   # 3-frame diagnostic
cd /home/nmahanti/ridge-filtering
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

/home/nmahanti/Unsupervised-Nanowire-Segmentation-using-CycleGAN-Architecture-2/venv/venv_l40s/bin/python scripts/ridge/wire_extract.py "$@"

echo "wire_extract done"

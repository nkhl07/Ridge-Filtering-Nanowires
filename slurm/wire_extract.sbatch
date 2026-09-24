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
# frame upscales to ~245 Mpx). Submit from the repo root; extra args pass
# through to the script:
#   sbatch slurm/wire_extract.sbatch                # all in-band frames
#   sbatch slurm/wire_extract.sbatch --validate-gt --frames SF251213M_65000×_24
#                                                   # 3-frame diagnostic
# Python: $PYTHON if set, else ./venv/bin/python, else python3 on PATH.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/ridge/wire_extract.py ]; then
    echo "error: submit from the repo root (got $SLURM_SUBMIT_DIR)" >&2
    exit 1
fi
mkdir -p logs

if [ -z "$PYTHON" ]; then
    if [ -x venv/bin/python ]; then PYTHON=venv/bin/python; else PYTHON=python3; fi
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON" scripts/ridge/wire_extract.py "$@" || exit $?

echo "wire_extract done"

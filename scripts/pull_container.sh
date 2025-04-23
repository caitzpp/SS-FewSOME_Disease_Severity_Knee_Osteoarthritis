#!/bin/bash
#SBATCH -p performance
#SBATCH -t 00:10:00
#SBATCH --job-name=pull_sif
#SBATCH --out=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/scripts/pull_out.log
#SBATCH --err=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/scripts/pull_err.log

SIF_FILENAME="my-python311-env.sif"
DOCKER_IMAGE="docker://caitlynzuppinger/my-python311-env:latest"

module load singularity  # or apptainer, depending on your cluster

# Pull only if not already present
if [ ! -f "$SIF_FILENAME" ]; then
  singularity pull "$SIF_FILENAME" "$DOCKER_IMAGE"
else
  echo "$SIF_FILENAME already exists, skipping pull."
fi

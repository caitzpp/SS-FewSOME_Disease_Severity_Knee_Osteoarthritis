#!/bin/bash
#SBATCH --job-name=pull_sif
#SBATCH --partition=performance
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --out=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/scripts/pull_out.log
#SBATCH --err=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/scripts/pull_err.log

# SIF_FILENAME="my-python311-env.sif"
# DOCKER_IMAGE="docker://caitlynzuppinger/my-python311-env:latest"

export SINGULARITY_TMPDIR=$HOME/tmp
export SINGULARITY_CACHEDIR=$HOME/.singularity

singularity pull my-python311-env.sif docker://caitlynzuppinger/my-python311-env:latest

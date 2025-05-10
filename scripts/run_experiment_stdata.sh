#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 62:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --job-name=Experiment_STData
#SBATCH --output=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Logs/training_%j.out     # Save stdout to file
#SBATCH --error=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Errors/training_%j.err      # Save stderr to file

PROJECT_FOLDER="$HOME/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome"
DATA_FOLDER="$HOME/data/raw/"
SIF_FILENAME="my-python311-env.sif"
FILENAME="main.py"

export XDG_CACHE_HOME=/workspace/cache
mkdir -p $XDG_CACHE_HOME

singularity exec --nv --no-home \
  -B "$PROJECT_FOLDER":/workspace \
  -B "$DATA_FOLDER":/data \
  -B "$HOME/VT9_SSFewSOME":/home2/c.zuppinger/VT9_SSFewSOME \
  "$SIF_FILENAME" python /workspace/"$FILENAME" \
  --data_path /data \
  --dir_path /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome \
  --train_ids_path /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/meta2/ \
  --device cuda \
  --save_models 2 \
  --train_ss 0 \
  --stage2 1 \
  --stage3 0 \
  --stage_severe_pred 0 \
  --model_name mod_st \
  --start_margin 0.5
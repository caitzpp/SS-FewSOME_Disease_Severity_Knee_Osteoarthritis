#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 62:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --job-name=Experiment_STData_SmallerIMG
#SBATCH --output=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Logs/training_%j.out
#SBATCH --error=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Errors/training_%j.err

PROJECT_FOLDER="$HOME/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome"
DATA_FOLDER="$HOME/data/raw/"
SIF_FILENAME="my-python311-env.sif"

# --- WandB config ---
# export WANDB_ENTITY=<YOUR_ENTITY>
# export WANDB_PROJECT=<YOUR_PROJECT>
export SWEEP_ID="compvis_cz/SS-Fewsome/pap7rxwv"
export WANDB_API_KEY=<YOUR_API_KEY>
export WANDB_DIR=/workspace/.wandb
export WANDB_CACHE_DIR=/workspace/cache
export XDG_CACHE_HOME=/workspace/cache

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# --- Fixed flags for main.py ---
FIXED_FLAGS=(
  --data_path /data/600x600_imgs
  --dir_path /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome
  --train_ids_path /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/meta2/
  --device cuda
  --eval_epoch 1
  --save_models 2
  --model_name mod_smallimg
  --use_wandb 1
  --train_ss 1
  --stage2 0
  --stage3 0
  --stage_severe_pred 0
)

# --- Launch inside Singularity ---
singularity exec --nv --no-home \
  -B "$PROJECT_FOLDER":/workspace \
  -B "$DATA_FOLDER":/data \
  -B "$HOME/VT9_SSFewSOME":/home2/c.zuppinger/VT9_SSFewSOME \
  "$SIF_FILENAME" bash -lc "
    cd /workspace
    python -m wandb agent \"$SWEEP_ID\" -- ${FIXED_FLAGS[*]}
"

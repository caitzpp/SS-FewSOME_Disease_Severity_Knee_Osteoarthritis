#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 12:00:00
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --job-name=Experiment_STData_SmallImg_samedata
#SBATCH --output=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Logs/training_%j.out     # Save stdout to file
#SBATCH --error=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Errors/training_%j.err      # Save stderr to file

PROJECT_FOLDER="$HOME/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome"
DATA_FOLDER="$HOME/data/raw/"
SIF_FILENAME="my-python311-env.sif"
FILENAME="main.py"

# --- WandB config ---
export WANDB_DIR=/workspace/.wandb
export WANDB_CACHE_DIR=/workspace/.wandb_cache
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

export XDG_CACHE_HOME=/workspace/cache
mkdir -p $XDG_CACHE_HOME

singularity exec --nv --no-home \
  -B "$PROJECT_FOLDER":/workspace \
  -B "$DATA_FOLDER":/data \
  -B "$HOME/VT9_SSFewSOME":/home2/c.zuppinger/VT9_SSFewSOME \
  "$SIF_FILENAME" python /workspace/"$FILENAME" \
  --data_path "/data/600x600_imgs" \
  --dir_path /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome \
  --train_ids_path /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/meta2/ \
  --device cuda \
  --eval_epoch 1 \
  --save_models 0 \
  --model_name mod_samedata \
  --use_wandb 1 \
  --train_ss 1 \
  --stage2 0 \
  --stage3 0 \
  --stage_severe_pred 0 \
  --lr 0.0005 \
  --bs 1 \
  --weight_decay 0.0001 \
  --beta1 0.85 \
  --beta2 0.999 \
  --eps 1e-08 \
  --use_same_image True \
  --wandb_agent compvis_cz/SS-Fewsome-SameData/wwq6advm \
  --sweep_count 20 \
  --seed 1001 \
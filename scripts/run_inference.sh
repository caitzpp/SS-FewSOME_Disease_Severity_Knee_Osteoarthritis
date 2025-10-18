#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 00:02:00
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --job-name=SS_Fewsome_Inference
#SBATCH --output=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Logs/inference_%j.out     # Save stdout to file
#SBATCH --error=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Errors/inference_%j.err      # Save stderr to file

PROJECT_FOLDER="$HOME/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome"
DATA_FOLDER="$HOME/data/raw/"
SIF_FILENAME="my-python311-env.sif"
FILENAME="inference.py"

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
  --task all \
  --save_models 0 \
  --model_name mod_smallimg2 \
  --use_wandb 0 \
  --train_ss 1 \
  --stage2 0 \
  --stage3 0 \
  --stage_severe_pred 0 \
  --lr 0.0005\
  --bs 1 \
  --weight_decay 0.0001 \
  --beta1 0.85 \
  --beta2 0.999 \
  --eps 1e-08 \
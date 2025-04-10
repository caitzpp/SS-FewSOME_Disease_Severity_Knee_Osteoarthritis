#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 03:00:00
#SBATCH --gpus=1
#SBATCH --job-name=MyTraining
#SBATCH --output=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Logs/training_%j.out     # Save stdout to file
#SBATCH --error=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Errors/training_%j.err      # Save stderr to file

cd /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis
source ./myenv/bin/activate
cd ./ss_fewsome
python main.py --data_path "/home2/c.zuppinger/data/raw/kaggle dataset" --save_models 2
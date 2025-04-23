#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 03:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --job-name=Experiment
#SBATCH --output=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Logs/training_%j.out     # Save stdout to file
#SBATCH --error=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Errors/training_%j.err      # Save stderr to file

cd /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis
source ./myenv/bin/activate
#python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python --version
cd ./ss_fewsome
#python main.py --data_path "/home2/c.zuppinger/data/raw/kaggle dataset" --dry_run --device cuda --save_models 2
#python ~/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/main.py --data_path "../../../data/raw/kaggle dataset" --device cuda --save_models 0 --stage2 0 --stage3 0 --stage_severe_pred 0 --ss_test 0

python main.py --data_path "../../../data/raw/kaggle dataset" --device cuda --save_models 0
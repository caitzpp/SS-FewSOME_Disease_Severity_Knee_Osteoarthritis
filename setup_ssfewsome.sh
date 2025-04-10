#!/bin/bash -l
#SBATCH --job-name=setup_ssfewsome
#SBATCH --output=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Logs/setup_output_%j.log
#SBATCH --error=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Errors/setup_error_%j.log
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=performance

cd /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis
python3 -m venv myenv
source myenv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
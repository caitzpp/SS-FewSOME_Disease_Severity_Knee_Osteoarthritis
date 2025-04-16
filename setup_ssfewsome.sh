#!/bin/bash -l
#SBATCH --job-name=setup_ssfewsome
#SBATCH --output=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Logs/setup_output_%j.log
#SBATCH --error=/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/Logs/Errors/setup_error_%j.log
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=performance

PYTHON_VERSION=3.11.8
INSTALL_DIR=$HOME/python-$PYTHON_VERSION

mkdir -p $HOME/build_python
cd $HOME/build_python

wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar xzf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION

./configure --prefix=$INSTALL_DIR --enable-optimizations
make -j4
make install

cd /home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis
$INSTALL_DIR/bin/python3.11 -m venv myenv
#python3 -m venv myenv
source myenv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
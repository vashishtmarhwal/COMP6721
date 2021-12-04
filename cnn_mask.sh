#!/encs/bin/tcsh

#####################################################################################################################
## Gurobi brings its own version of Python but that one does not contain any 3rd-party Python packages except Gurobi. 
## In order to use Gurobi together with popular Python packages like multiprocessing and others, we need to create a
## virtual Python environment, in which we can install gurobipy and other pakages.
##
## You can create the new virtual Python environment inside your home directory before scheduling your job,
## or create it inside $TMPDIR on fly as a part of your job
#####################################################################################################################

#$ -N MY_JOB
#$ -cwd
#$ -m bea
#$ -pe smp 8
#$ -l h_vmem=150G

##PUT YOUR MODULE LOADS HERE
## python 3.6, numpy 1.17.0, matplot.pyplot 3.3.4, cv2 4.5.4.58, seaborn 0.11.2, pandas 1.1.5, torch 1.10.0, torchvision 0.11.1, sklearn 0.22.1
module load python/3.5.1/default
module load opencv/4.5.4/default
module load pytorch/1.10.0/default

## Create a virtual Python environment (env) in $TMPDIR
python3.6 -m venv $TMPDIR/env
## Activate the new environment
source $TMPDIR/env/bin/activate.csh
## Install gurobipy module

cd $OPENCV_HOME
python3.6 setup.py build --build-base /tmp/${USER} install

cd $PYTORCH_HOME
python3.6 setup.py build --build-base /tmp/${USER} install


## return to workDir
cd $vashCOMP6721

## Now, instead of using 'gurobi.sh MY_PYTHON_SCRIPT.py', you can use
python main.py
## inside MY_PYTHON_SCRIPT.py, you can use
## from gurobipy import *
## import multiprocessing as mp
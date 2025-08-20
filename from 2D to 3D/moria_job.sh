#!/bin/zsh
#SBATCH --job-name=run1
#SBATCH -o current_runs/%x_%J.out
#SBATCH -e current_runs/%x_%J.err
#SBATCH --mem=128g
#SBATCH -c100
#SBATCH --time=48:0:0
#SBATCH --gres=gpu:0

cd /cs/labs/tsevi/amitaiovadia/miniconda3
conda activate py39_env
cd /cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict
/cs/labs/tsevi/amitaiovadia/miniconda3/envs/py39_env/bin/python3.9 /cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/run_predict_2D.py









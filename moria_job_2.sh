#!/bin/zsh
#SBATCH --job-name=run1
#SBATCH -o current_runs/%x_%J.out
#SBATCH -e current_runs/%x_%J.err
#SBATCH --mem=128g
#SBATCH -c10
#SBATCH --time=3:0:0
#SBATCH --gres=gpu:1

echo "started"

cd /cs/labs/tsevi/amitaiovadia/miniconda3
conda activate tf_39_env_2_14

echo "activated conda"
cd /cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict


# /cs/labs/tsevi/amitaiovadia/miniconda3/envs/py39_env/bin/python3.9 /cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/run_predict_2D.py
# /cs/labs/tsevi/amitaiovadia/miniconda3/envs/py39_env/bin/python3.9 /cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/estimate_analysis_errors.py
echo "run python"

/cs/labs/tsevi/amitaiovadia/miniconda3/envs/tf_39_env_2_14/bin/python3.9 /cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/estimate_analysis_errors.py

echo "finished working"






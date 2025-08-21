#!/bin/zsh
#SBATCH --job-name=predict_short_try
#SBATCH -o current_runs/%x_%J.out
#SBATCH -e current_runs/%x_%J.err
#SBATCH --mem=64g
#SBATCH -c10
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:1

echo "started"

cd /cs/labs/tsevi/lior.kotlar/pose-estimation
echo "activating environment"
source /cs/labs/tsevi/lior.kotlar/pose-estimation/trainenv/bin/activate

echo "run python"

/cs/labs/tsevi/lior.kotlar/pose-estimation/trainenv/bin/python3.11 /cs/labs/tsevi/lior.kotlar/pose-estimation/training_network/tensorflow/train.py "/cs/labs/tsevi/lior.kotlar/pose-estimation/training network/configurations/MODEL_18_POINTS_3_GOOD_CAMERAS/configuration_short.json"

echo "finished working"






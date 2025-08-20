#!/bin/zsh
#SBATCH --job-name=predict_short_try
#SBATCH -o current_runs/%x_%J.out
#SBATCH -e current_runs/%x_%J.err
#SBATCH --mem=64g
#SBATCH -c10
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:1

echo "started"

cd /cs/labs/tsevi/lior.kotlar/amitai-s-thesis
echo "activating environment"
source /cs/labs/tsevi/lior.kotlar/amitai-s-thesis/trainenv/bin/activate

echo "run python"

/cs/labs/tsevi/lior.kotlar/amitai-s-thesis/trainenv/bin/python3.11 /cs/labs/tsevi/lior.kotlar/amitai-s-thesis/training_network/tensorflow/train.py "/cs/labs/tsevi/lior.kotlar/amitai-s-thesis/training network/configurations/MODEL_18_POINTS_3_GOOD_CAMERAS/configuration_short.json"

echo "finished working"






import numpy as np
from matplotlib import pyplot as plt
import os
from predict_2D_sparse_box import Predictor2D
from run_predict_2D import Flight3DProcessing
import pickle
from tqdm import tqdm
from collections import defaultdict
from predictions_2Dto3D import From2Dto3D


def find_indexes(A, B):
    indexes = []
    for elem in B:
        try:
            index = A.index(elem)
            indexes.append(index)
        except ValueError:
            # In case the element is not found in A, this block will execute.
            pass
    return indexes


def test_models_combination(chosen_models, all_frames_scores, all_models_combinations):
    """
    if I use only this subset of models, what is the final movie score
    """
    left_wing_inds = list(np.arange(0, 7))
    right_wing_inds = list(np.arange(8, 15))
    head_tail_inds = [16, 17]
    side_wing_inds = [7, 15]

    left, right, head_tail, side = all_frames_scores
    all_subsets_models = Predictor2D.all_possible_combinations(chosen_models, fraq=0.1)
    models_comb_indices = find_indexes(all_models_combinations, all_subsets_models)
    num_frames = len(left)
    final_points_3D = np.zeros((num_frames, 18, 3))
    body_part_indices = [left_wing_inds, right_wing_inds, head_tail_inds, side_wing_inds]

    for i, body_part in enumerate([left, right, head_tail, side]):
        all_points = []
        all_scores = []
        all_best_combinations = []
        for frame in range(num_frames):
            frame_scores = body_part[frame]
            best_score = np.inf
            best_points = None
            best_combination = None
            for model_com_ind in models_comb_indices:
                comb_dict = frame_scores[model_com_ind]
                score = comb_dict['score']
                if score < best_score:
                    best_score = score
                    points = comb_dict['points_3D']
                    best_points = points
                    best_combination = comb_dict['model_combination']
            all_best_combinations.append(best_combination)
            all_points.append(best_points)
            all_scores.append(best_score)
        all_points = np.array(all_points)
        body_part = body_part_indices[i]
        final_points_3D[:, body_part, :] = all_points
    return final_points_3D


def run_ablation_study(movie_path, load_optimization_results=True, load_final_results=True):
    save_path_all_combinations_scores = os.path.join(movie_path, "all_combinations_model_scores_pkl.pkl")
    save_path_all_best_combinations_scores = os.path.join(movie_path, "all_best_combinations_scores_pkl.pkl")
    save_path_all_best_combinations_scores_smoothed = os.path.join(movie_path,
                                                                   "all_best_combinations_scores_smoothed_pkl.pkl")
    _, all_points_list = Flight3DProcessing.predict_3D_points_all_pairs(movie_path)
    all_points_list = [all_points_list[i][:, :, :6, :] for i in range(len(all_points_list))]

    if load_optimization_results:
        with open(save_path_all_combinations_scores, "rb") as f:
            all_frames_scores = pickle.load(f)
    else:
        final_score, best_points_3D, all_models_combinations, all_frames_scores = Predictor2D.find_3D_points_optimize_neighbors(
            all_points_list)
        with open(save_path_all_combinations_scores, "wb") as f:
            pickle.dump(all_frames_scores, f)

    all_models_combinations = Predictor2D.all_possible_combinations(np.arange(len(all_points_list)), fraq=0.1)

    if load_final_results:
        with open(save_path_all_best_combinations_scores, "rb") as f:
            model_scores = pickle.load(f)
        with open(save_path_all_best_combinations_scores_smoothed, "rb") as f:
            model_scores_smoothed = pickle.load(f)
    else:
        model_scores, model_scores_smoothed = [], []
        for combination in tqdm(all_models_combinations):
            best_points_3D = test_models_combination(combination, all_frames_scores, all_models_combinations)
            points_3D_smoothed = From2Dto3D.smooth_3D_points(best_points_3D)
            final_score = From2Dto3D.get_validation_score(best_points_3D)
            final_score_smoothed = From2Dto3D.get_validation_score(points_3D_smoothed)
            model_scores.append((combination, final_score))
            model_scores_smoothed.append((combination, final_score_smoothed))

        with open(save_path_all_best_combinations_scores, "wb") as f:
            pickle.dump(model_scores, f)
        with open(save_path_all_best_combinations_scores_smoothed, "wb") as f:
            pickle.dump(model_scores_smoothed, f)

    display_combinations_results(model_scores, movie_path, smoothed=False)
    display_combinations_results(model_scores_smoothed, movie_path, smoothed=True)


def display_combinations_results(model_scores, movie_path, smoothed=False):
    best_scores = defaultdict(lambda: [])  # Now store lists of combinations for each length
    model_scores_sum = defaultdict(lambda: [0, 0])  # [sum of scores, count]

    # Iterate through the model combinations and scores
    for combination, score in model_scores:
        length = len(combination)  # Get the length of the combination
        # Add the current combination and score
        best_scores[length].append((combination, score))
        # Keep only the top 8 scores for each length in descending order
        best_scores[length] = sorted(best_scores[length], key=lambda x: x[1])[:80]

        # Update the sum and count for each model in the combination
        for model in combination:
            model_scores_sum[model][0] += score
            model_scores_sum[model][1] += 1

    mean_scores = dict()
    for model, (total_score, count) in model_scores_sum.items():
        mean_score = total_score / count if count > 0 else 0
        mean_scores[model] = mean_score

    print("Mean Scores per Model:")
    print(mean_scores)
    print("\nBest Scores per Combination Length:")
    print(best_scores)

    add_text = " smoothed" if smoothed else ""

    # Define number of subplots based on number of unique lengths
    num_lengths = len(best_scores)
    fig, axes = plt.subplots(num_lengths, 1, figsize=(12, 20 * num_lengths))  # Adjust height per number of lengths

    if num_lengths == 1:
        axes = [axes]  # Ensure axes is always a list even for a single subplot

    # Create subplots for each combination length
    for idx, (length, ax) in enumerate(zip(sorted(best_scores.keys()), axes)):
        top_combinations = best_scores[length]  # Get the top combinations for this length
        top_scores = [score for _, score in top_combinations]
        top_comb_names = [", ".join(map(str, comb)) for comb, _ in top_combinations]

        bars = ax.bar(range(len(top_scores)), top_scores, width=0.5, color='skyblue')
        ax.set_xticks(range(len(top_scores)))
        ax.set_xticklabels(top_comb_names, rotation=45, ha="right", fontsize=4)
        ax.set_title(f"Top 8 Combinations for Length {length}{add_text}")
        ax.set_xlabel("Combinations")
        ax.set_ylabel(f"Score{add_text}")

        # Set the y-axis limit slightly higher to give room for the annotations
        y_max = max(top_scores) * 1.2  # 20% higher than the highest score
        ax.set_ylim(0, y_max)  # Set the y-axis limit

        # Format and display score annotations (multiplying by 1e5 and showing 1 digit after the decimal)
        for bar, score in zip(bars, top_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score * 1e5:.3f}",  # Multiply by 1e5 and format to 1 decimal point
                ha='center',
                va='bottom',
                fontsize=3,
                color='black'
            )

    # Adjust layout to give more space between subplots
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to fit the subplots with extra space
    plt.subplots_adjust(hspace=2)  # Increase space between subplots

    # Save the plot with all subplots
    save_path = os.path.join(movie_path, f"top_8_combinations_all_lengths{add_text}.png")
    fig.savefig(save_path, dpi=600)
    plt.close(fig)  # Close the figure

    # Save the original best score graph (Graph 1)
    fig1, ax1 = plt.subplots(figsize=(30, 10))
    best_lengths = sorted(best_scores.keys())  # Sort the combination lengths
    best_scores_values = [best_scores[length][0][1] for length in best_lengths]
    best_combinations = [best_scores[length][0][0] for length in best_lengths]

    bars1 = ax1.bar(range(len(best_lengths)), best_scores_values, color='skyblue')
    ax1.set_xticks(range(len(best_lengths)))
    ax1.set_xticklabels(best_lengths)
    ax1.set_title(f"Best Scores by Combination Length{add_text}")
    ax1.set_xlabel("Combination Length")
    ax1.set_ylabel(f"Best Score{add_text}")

    # Add score and combination annotations for best_scores
    for i, (bar, combination) in enumerate(zip(bars1, best_combinations)):
        combination_str = ', '.join(map(str, combination))
        # Score above the bar
        ax1.text(
            bar.get_x() + bar.get_width() / 2,  # x position
            bar.get_height(),  # y position
            f"{best_scores_values[i]:.4e}",  # formatted score text
            ha='center',  # horizontal alignment
            va='bottom',  # vertical alignment
            fontsize=8,
            color='black'
        )
        # Combination below the bar (mid-bar)
        ax1.text(
            bar.get_x() + bar.get_width() / 2,  # x position
            bar.get_height() / 2,  # y position (mid-bar)
            f"Combo: [{combination_str}]",  # combination text
            ha='center',  # horizontal alignment
            va='bottom',  # vertical alignment
            rotation=0,  # no rotation
            fontsize=8,
            color='black'
        )

    # Save the best scores graph as a high-resolution PNG image
    save_path_best_scores = os.path.join(movie_path, f"best_combination_per_ensemble_size{add_text}.png")
    fig1.savefig(save_path_best_scores, dpi=600)
    plt.close(fig1)  # Close the figure


if __name__ == '__main__':
    # movie_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/ablation study/mov1"
    # movie_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/ablation study/mov53"
    movie_path = r"G:\My Drive\Amitai\one halter experiments\ablation study\mov53"
    run_ablation_study(movie_path)

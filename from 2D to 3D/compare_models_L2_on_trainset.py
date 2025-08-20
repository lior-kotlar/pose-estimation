import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
import h5py


left_inds = np.arange(0, 8)
right_inds = np.arange(8, 16)


def tf_find_peaks(x):
    in_shape = tf.shape(x)
    flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])
    idx = tf.argmax(flattened, axis=1)
    rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
    cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])
    vals = tf.math.reduce_max(flattened, axis=1)
    pred = tf.stack([
        tf.cast(cols, tf.float32),
        tf.cast(rows, tf.float32),
    ], axis=1)
    return pred


def run_model_model_eval(model_type, model_path, dataset_path, dir_name):
    output_directory = os.path.join('compare_models', dir_name)
    errors_save_path = os.path.join(output_directory, 'errors.npy')
    new_config_path = os.path.join(output_directory, 'predict_2D_config.json')
    configuration_path = "predict_2D_config.json"
    ground_truth_confmaps = h5py.File(dataset_path, 'r')['/confmaps'][:].T
    ground_truth_2D = []
    for cam in range(4):
        gt = tf_find_peaks(ground_truth_confmaps[:, cam])
        gt = np.transpose(gt, axes=[0, 2, 1])
        gt = gt[..., :2]
        gt = gt[:, np.newaxis, :]
        ground_truth_2D.append(gt)
    ground_truth_2D = np.concatenate(ground_truth_2D, axis=1)
    with open(configuration_path) as C:
        config = json.load(C)
        config['model type'] = model_type
        config['box path'] = dataset_path
        config['is video'] = False
        config['wings pose estimation model path'] = model_path
        config["body parts to predict"] = 'WINGS_AND_BODY'
        with open(new_config_path, 'w') as file:
            json.dump(config, file, indent=4)

        predictor = Predictor2D(new_config_path, is_masked=True)
        predictor.run_predict_2D(save=False)
        predictions_2D = predictor.preds_2D

        ground_truth_2D_verified = np.zeros_like(ground_truth_2D)
        for frame in range(predictions_2D.shape[0]):
            for cam in range(predictions_2D.shape[1]):
                gt_orig_2D = ground_truth_2D[frame, cam]
                gt_flipped_2D = gt_orig_2D.copy()
                gt_flipped_2D[left_inds], gt_flipped_2D[right_inds] = gt_orig_2D[right_inds], gt_flipped_2D[left_inds]
                preds_2D = predictions_2D[frame, cam]

                if np.linalg.norm(preds_2D - gt_flipped_2D) < np.linalg.norm(preds_2D - gt_orig_2D):
                    gt_2D = gt_flipped_2D
                else:
                    gt_2D = gt_orig_2D
                ground_truth_2D_verified[frame, cam] = gt_2D

        errors = predictions_2D - ground_truth_2D_verified
        indices = np.array([i for i in range(errors.shape[2]) if i not in [7, 15]])
        new_errors = np.linalg.norm(errors[:, :, indices, :], axis=-1)
        print(f"{output_directory}\nmean error:{new_errors.mean()}\nstd:{new_errors.std()}")
        np.save(errors_save_path, new_errors)


def run_single_view_not_reprojected():
    model_type = "WINGS_AND_BODY_SAME_MODEL"
    model_path = r"models 5.0\per wing\MODEL_18_POINTS_PER_WING_Jun 09 not reprojected\best_model.h5"
    if cluster:
        dataset_path = "/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/training datasets/random_trainset_201_frames_18_joints.h5"
    else:
        dataset_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints.h5"
    dir_name = 'single_view_not_reprojected'
    run_model_model_eval(model_type, model_path, dataset_path, dir_name)


def run_single_view_reprojected():
    model_type = "WINGS_AND_BODY_SAME_MODEL"
    model_path = r"models 5.0/per wing/MODEL_18_POINTS_PER_WING_Jun 09_02 0.7-1.3 reprojected/best_model.h5"
    if cluster:
        dataset_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/training datasets/random_trainset_201_frames_18_joints_reprojected_masks.h5"
    else:
       dataset_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints_reprojected_masks.h5"
    dir_name = 'single_view_reprojected'
    run_model_model_eval(model_type, model_path, dataset_path, dir_name)


def run_multiview_reprojected():
    model_type = "ALL_CAMS_PER_WING"
    model_path = r"models 5.0/4 cameras/ALL_CAMS_18_POINTS_Jun 11_01/best_model.h5"
    if cluster:
        dataset_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/training datasets/random_trainset_201_frames_18_joints_reprojected_masks.h5"
    else:
        dataset_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints_reprojected_masks.h5"
    dir_name = 'multiview_reprojected'
    run_model_model_eval(model_type, model_path, dataset_path, dir_name)


def display_errors():
    dirs = ["multiview_reprojected", "single_view_not_reprojected", "single_view_reprojected"]
    base_path = "compare_models"

    # Prepare the header for the CSV file
    header = ['Model', 'Number of Errors', 'Std', 'Mean', 'Median', 'Maximum Error', 'error = 0 (%)',
              'error <= 1 (%)',
              'error <= 2 (%)', 'error <= 3 (%)', 'error <= 4 (%)',
              'error => 5 (%)', 'error => 10 (%)']

    # Open a CSV file to store the results
    csv_file_path = os.path.join(base_path, 'model_errors_summary.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # Process each directory/model
        for dir in dirs:
            errors_path = os.path.join(base_path, dir, 'errors.npy')
            errors = np.load(errors_path).flatten()

            num_errors = len(errors)
            std = np.std(errors)
            mean = np.mean(errors)
            median = np.median(errors)
            max_error = np.max(errors)  # Find the maximum error

            # Calculate ratios and convert to percentages
            ratio_0 = (np.sum(errors == 0) / num_errors) * 100
            ratio_1_minus = (np.sum(errors <= 1) / num_errors) * 100
            ratio_2_minus = (np.sum(errors <= 2) / num_errors) * 100
            ratio_3_minus = (np.sum(errors <= 3) / num_errors) * 100
            ratio_4_minus = (np.sum(errors <= 5) / num_errors) * 100
            ratio_5_plus = (np.sum(errors >= 5) / num_errors) * 100
            ratio_10_plus = (np.sum(errors >= 10) / num_errors) * 100

            # Round values to 2 decimal places for cleaner output
            row = [
                dir,
                num_errors,
                round(std, 2),
                round(mean, 2),
                round(median, 2),
                round(max_error, 2),
                round(ratio_0, 2),
                round(ratio_1_minus, 2),
                round(ratio_2_minus, 2),
                round(ratio_3_minus, 2),
                round(ratio_4_minus, 2),
                round(ratio_5_plus, 2),
                round(ratio_10_plus, 2)
            ]

            writer.writerow(row)

    print(f"Results have been saved to {csv_file_path}")


if __name__ == '__main__':
    display_errors()

    if False:
        cluster = False
        from predict_2D_sparse_box import Predictor2D

        run_multiview_reprojected()
        run_single_view_reprojected()
        run_single_view_not_reprojected()

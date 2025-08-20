# run using deep lab cut interpreter!!
# import h5py
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import cv2

train_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\as_numpy_arrays\train_box.npy"
confmaps_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\as_numpy_arrays\train_confmaps.npy"
RELEVANT_FEATURE_POINTS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17]


def tf_find_peaks(x):
    """ Finds the maximum value in each channel and returns the location and value.
    Args:
        x: rank-4 tensor (samples, height, width, channels)

    Returns:
        peaks: rank-3 tensor (samples, [x, y, val], channels)
    """

    # Store input shape
    in_shape = tf.shape(x)

    # Flatten height/width dims
    flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])

    # Find peaks in linear indices
    idx = tf.argmax(flattened, axis=1)

    # Convert linear indices to subscripts
    rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
    cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])

    # Dumb way to get actual values without indexing
    vals = tf.math.reduce_max(flattened, axis=1)

    # Return N x 3 x C tensor
    pred = tf.stack([
        tf.cast(cols, tf.float32),
        tf.cast(rows, tf.float32),
        ],
        axis=-1)
    return pred.numpy()


def load_dataset():
    box = np.load(train_path)[..., 1][..., np.newaxis]
    trainset = np.concatenate((box, box, box), axis=-1)
    return trainset


def load_labels():
    confmaps = np.load(confmaps_path)[..., RELEVANT_FEATURE_POINTS]
    peaks = tf_find_peaks(confmaps)
    return peaks


def load_from_mat(path, name, end=100):
    mat = scipy.io.loadmat(path)
    numpy_array = mat[name]
    numpy_array = np.transpose(numpy_array, (3, 2, 0, 1))
    return numpy_array[:end]

def save_labels_and_dataset():
    labels = load_labels()
    dataset = load_dataset()
    np.save("labels.npy", labels)
    np.save("dataset.npy", dataset)


def save_labeled_video():
    reprojected_points_2D_path = (r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled "
                                  r"dataset\estimated_positions.mat")
    ground_truth_2D_path = (r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled "
                            r"dataset\ground_truth_labels.mat")
    ground_truth_2D = load_from_mat(ground_truth_2D_path, name='ground_truth')
    num_frames = len(ground_truth_2D)
    num_joints = ground_truth_2D.shape[2]
    reprojected_points_2D = load_from_mat(reprojected_points_2D_path, name='positions',
                                                                      end=num_frames)
    box = h5py.File(r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\trainset_movie_1_370_520_ds_3tc_7tj.h5", 'r')['/box'][:num_frames]
    box = np.transpose(box, (0, 3, 2, 1))
    cams = [box[..., 1 + 3*i] for i in range(4)]
    for i, cam in enumerate(cams):
        cam = cam[..., np.newaxis]
        image_arrays = np.concatenate((cam, cam, cam), axis=-1)
        # Get the shape from the first image
        height, width = image_arrays[0].shape[:2]
        output_path = "labeled_cam_" + str(i + 1) + ".mp4"
        # Create video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

        # Write each frame
        for img in image_arrays:
            # OpenCV expects BGR format
            # If your arrays are in RGB format, convert them
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            out.write(img)

        # Release the video writer
        out.release()
    pass


if __name__ == '__main__':
    save_labeled_video()
    # save_labels_and_dataset()
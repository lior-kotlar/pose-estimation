import os
from PIL import Image, ImageFilter
import h5py
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter, shift
import matplotlib
matplotlib.use('TkAgg')
import cv2

def preprocess(X, permute=(0, 3, 2, 1)):
    """ Normalizes input data. """

    # Add singleton dim for single train_images
    if X.ndim == 3:
        X = X[None, ...]

    # Adjust dimensions
    if permute != None:
        X = np.transpose(X, permute)

    # Normalize
    if X.dtype == "uint8" or np.max(X) > 1:
        X = X.astype("float32") / 255

    return X


def load_dataset(data_path, X_dset="box", Y_dset="confmaps", permute=(0, 3, 2, 1)):
    """ Loads and normalizes datasets. """
    # Load
    with h5py.File(data_path, "r") as f:
        X = f[X_dset][:]
        Y = f[Y_dset][:]

    # Adjust dimensions
    X = preprocess(X, permute=None)
    Y = preprocess(Y, permute=None)
    if X.shape[0] != 2:
        X = X.T
    if Y.shape[0] != 2 or Y.shape[1] == 192:
        Y = Y.T
    X = X.T
    points_3D = h5py.File(data_path, "r")["/points_3D"][:]
    points_3D = np.transpose(points_3D, [1, 2, 0])[:X.shape[0]]
    points_3D_per_camera = np.repeat(np.expand_dims(points_3D, axis=1), X.shape[1], axis=1)
    return X, Y, points_3D


def zoom_image(img, scale):
    height, width = img.shape[:2]  # Get the height and width of the image
    center = (width / 2, height / 2)  # Get the center of the image

    # Generate a rotation matrix with no rotation, only scaling
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=0, scale=scale)

    # Apply the transformation (rotation and scaling)
    zoomed_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # plt.figure()
    # plt.imshow(zoomed_img)
    # plt.axis('equal')
    # plt.show()
    return zoomed_img


def augment(img, h_fl, v_fl, rotation_angle, shift_y_x, zoom_factor):
    if np.max(img) <= 1:
        img = np.uint8(img * 255)
    if h_fl:
        img = np.fliplr(img)
    if v_fl:
        img = np.flipud(img)
    img = zoom_image(img, zoom_factor)
    img = shift(img, shift_y_x)
    img_pil = Image.fromarray(img)
    img_pil = img_pil.rotate(rotation_angle, 3)
    img = np.asarray(img_pil)
    if np.max(img) > 1:
        img = img/255
    return img


def blur_channel(img_channel, sigma):
    """ a (imsize, imsize) numpy array """
    return gaussian_filter(img_channel, sigma=sigma)


def custom_augmentations(img):
    """get an image of shape (height, width, num_channels) and return augmented image"""
    # img_orig = img.copy()
    do_horizontal_flip = np.random.randint(2)
    do_vertical_flip = np.random.randint(2)
    rotation_angle = np.random.randint(-180, 180)
    shift_y_x = np.random.randint(-10, 10, 2)
    num_channels = img.shape[-1]
    zoom_factor = np.random.uniform(0.6, 1.3)
    for channel in range(num_channels):
        img[:, :, channel] = augment(img[:, :, channel], do_horizontal_flip,
                                     do_vertical_flip, rotation_angle, shift_y_x, zoom_factor)
    return img

def camera_matrix_augment(inputs):
    """

    Args:
        confmaps: a (192, 192, 10 * 4) confidence maps
        points_3D: the corresponding (10, 3) points 3D coordinates
    Returns: 4 augmented camera matrices of size (4, 4, 3)
    """
    confmaps, points_3D = inputs
    confmaps = custom_augmentations(confmaps)
    for cam in range(4):
        confmaps_cam = confmaps[:, :, (cam * 10) + np.arange(10)]
        points_2D = tf_find_peaks(confmaps_cam)
    return np.zeros((4,4,3))


def test_generators(data_path):
    box, confmaps, points_3D = load_dataset(data_path)
    image_size = confmaps.shape[-2]
    num_channels_img = box.shape[-1]
    num_channels_confmap = confmaps.shape[-1]
    num_cams = box.shape[1]

    # box = box.reshape([-1, image_size, image_size, num_channels_img])
    # confmaps = confmaps.reshape([-1, image_size, image_size, num_channels_confmap])
    box = box.reshape([-1, image_size, image_size, num_channels_img])
    confmaps = confmaps.reshape([-1, image_size, image_size, num_channels_confmap])

    # cam_boxes = []
    # cam_confmaps = []
    # cam_points_3D = []
    # for cam in range(num_cams):
    #     box_cam_i = box[:, cam, :, :, :]
    #     cam_confmaps_i = confmaps[:, cam, :, :, :]
    #     cam_boxes.append(box_cam_i)
    #     cam_confmaps.append(cam_confmaps_i)
    # box = np.concatenate(cam_boxes, axis=-1)
    # confmaps = np.concatenate(cam_confmaps, axis=-1)

    box = box[:100]
    confmaps = confmaps[:100]
    pass

    # box = box[..., :3]
    # points_3D = np.concatenate((points_3D[:, 0, ...],
    #                            points_3D[:, 1, ...],
    #                            points_3D[:, 2, ...],
    #                            points_3D[:, 3, ...]), axis=0)
    #
    # box = np.concatenate((box[:, 0, ...],
    #                            box[:, 1, ...],
    #                            box[:, 2, ...],
    #                            box[:, 3, ...]), axis=0)
    # confmaps = np.concatenate((confmaps[:, 0, ...],
    #                                 confmaps[:, 1, ...],
    #                                 confmaps[:, 2, ...],
    #                                 confmaps[:, 3, ...]), axis=0)
    seed = 0
    batch_size = 8
    # data_gen_args = dict(preprocessing_function=custom_augmentations,)


    data_gen_args = dict(rotation_range=45,
                         zoom_range=[0.8, 1.2],
                         horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=10,
                         height_shift_range=10,
                         interpolation_order=2,
                         shear_range=10)


    # data generator
    datagen_x = ImageDataGenerator(**data_gen_args)
    datagen_y = ImageDataGenerator(**data_gen_args)
    # prepare iterator
    datagen_x.fit(box[:, :, :, :], augment=True, seed=seed)
    datagen_y.fit(confmaps[:, :, :, :], augment=True, seed=seed)
    flow_box = datagen_x.flow(box, batch_size=batch_size, seed=seed)
    flow_conf = datagen_y.flow(confmaps, batch_size=batch_size, seed=seed)
    train_generator = zip(flow_box, flow_conf)
    #
    # # train_generator = ((x, tf_find_peaks(y)) for x, y in zip(flow_box, flow_conf))
    # # Create a custom generator for points_3D
    #
    # def points_3D_generator():
    #     for i in range(0, len(points_3D), batch_size):
    #         yield points_3D[i:i + batch_size]
    #
    # flow_points_3D = points_3D_generator()

    # train_generator = ((x, y, z) for x, y, z in zip(flow_box, flow_conf, flow_cam_mat))

    # for i, (X, Y, Z) in enumerate(train_generator):
    #     ind = 0
    #
    #     points_2D_cam_1 = tf_find_peaks(Y[:, :, :, 0:18])[ind].numpy().T[:, :-1]
    #     points_3D_ind = Z[ind][0]
    #
    #     points_2D_cam_2 = tf_find_peaks(Y[:, :, :, 18:18+18])[ind].numpy().T[:, :-1]
    #
    #     P1 = estimate_projection_matrix_dlt(points_3D_ind, points_2D_cam_1)
    #     P2 = estimate_projection_matrix_dlt(points_3D_ind, points_2D_cam_2)
    #
    #     tr_points_3d = cv2.triangulatePoints(P1, P2, points_2D_cam_1.T, points_2D_cam_2.T).T
    #     tr_points_3d = tr_points_3d[:, :-1] / tr_points_3d[:, -1:]
    #     error = np.mean(np.abs(points_3D_ind - tr_points_3d))
    #     pass

    for batch in train_generator:
        images, confmaps = batch
        confmaps = np.sum(confmaps, axis=-1)
        images = images[..., 1]
        fig, axes = plt.subplots(4, 2, figsize=(10, 15))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Display images with confmaps
        for i in range(8):
            combined_image = images[i] + confmaps[i]
            axes[i].imshow(combined_image, cmap='gray')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()



def experiment(data_path):
    box, confmaps, points_3D = load_dataset(data_path)
    image_size = confmaps.shape[-2]
    num_channels_img = box.shape[-1]
    num_channels_confmap = confmaps.shape[-1]
    num_cams = box.shape[1]

    # box = box.reshape([-1, image_size, image_size, num_channels_img])
    # confmaps = confmaps.reshape([-1, image_size, image_size, num_channels_confmap])
    box = box.reshape([-1, num_cams, image_size, image_size, num_channels_img])
    confmaps = confmaps.reshape([-1, num_cams, image_size, image_size, num_channels_confmap])

    cam_boxes = []
    cam_confmaps = []
    cam_points_3D = []
    for cam in range(num_cams):
        box_cam_i = box[:, cam, :, :, :]
        cam_confmaps_i = confmaps[:, cam, :, :, :]
        cam_boxes.append(box_cam_i)
        cam_confmaps.append(cam_confmaps_i)
    box = np.concatenate(cam_boxes, axis=-1)
    confmaps = np.concatenate(cam_confmaps, axis=-1)

    box = box[:20]
    confmaps = confmaps[:20]
    points_3D = points_3D[:20]

    seed = 0
    batch_size = 8

    datagen_cam_mat = ImageDataGenerator(**dict(preprocessing_function=camera_matrix_augment))
    datagen_cam_mat.fit((confmaps, points_3D), augment=True, seed=seed)
    flow_cam_mat = datagen_cam_mat.flow((confmaps, points_3D), batch_size=batch_size, seed=seed)

    data_gen_args = dict(preprocessing_function=custom_augmentations)
    datagen_x = ImageDataGenerator(**data_gen_args)
    datagen_y = ImageDataGenerator(**data_gen_args)
    # prepare iterator
    datagen_x.fit(box, augment=True, seed=seed)
    datagen_y.fit(confmaps, augment=True, seed=seed)
    flow_box = datagen_x.flow(box, batch_size=batch_size, seed=seed)
    flow_conf = datagen_y.flow(confmaps, batch_size=batch_size, seed=seed)
    train_generator = zip(flow_box, flow_conf, flow_cam_mat)

    for i, (X, Y, M) in enumerate(train_generator):
        ind = 0

        points_2D_cam_1 = tf_find_peaks(Y[:, :, :, 0:18])[ind].numpy().T[:, :-1]
        points_3D_ind = Z[ind]

        points_2D_cam_2 = tf_find_peaks(Y[:, :, :, 18:18+18])[ind].numpy().T[:, :-1]

        P1 = estimate_projection_matrix_dlt(points_3D_ind, points_2D_cam_1)
        P2 = estimate_projection_matrix_dlt(points_3D_ind, points_2D_cam_2)

        tr_points_3d = cv2.triangulatePoints(P1, P2, points_2D_cam_1.T, points_2D_cam_2.T).T
        tr_points_3d = tr_points_3d[:, :-1] / tr_points_3d[:, -1:]
        error = np.mean(np.abs(points_3D_ind - tr_points_3d))
        pass


def try_triangulation(confmaps, points_3D):
    ind = 5
    pts_3d = points_3D[ind, 0]
    confmaps_cam1 = confmaps[ind, :, :, 0:18]
    confmaps_cam2 = confmaps[ind, :, :, 18:18 + 18]
    print("check error sanity check")
    check_error(confmaps_cam1, confmaps_cam2, pts_3d)
    aug_confmaps_cam1 = custom_augmentations(confmaps_cam1.copy())
    aug_confmaps_cam2 = custom_augmentations(confmaps_cam2.copy())
    print("check error after augmentations")
    check_error(aug_confmaps_cam1, aug_confmaps_cam2, pts_3d)


def check_error(confmaps_cam1, confmaps_cam2, pts_3d):
    points_2d_cam1 = tf_find_peaks(confmaps_cam1[np.newaxis, :])[0].numpy().T[:, :-1]
    points_2d_cam2 = tf_find_peaks(confmaps_cam2[np.newaxis, :])[0].numpy().T[:, :-1]
    P1 = estimate_projection_matrix_dlt(pts_3d, points_2d_cam1)
    P2 = estimate_projection_matrix_dlt(pts_3d, points_2d_cam2)
    tr_points_3d = cv2.triangulatePoints(P1, P2, points_2d_cam1.T, points_2d_cam2.T).T
    tr_points_3d = tr_points_3d[:, :-1] / tr_points_3d[:, -1:]
    error = np.mean(np.abs(pts_3d - tr_points_3d))
    print('error 3d: ', error)


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
        vals
    ], axis=1)
    return pred

def estimate_projection_matrix_dlt(points_3d, points_2d):
    assert len(points_2d) == len(points_3d)
    assert len(points_2d) >= 6

    A = []

    for i in range(len(points_2d)):
        X, Y, Z = points_3d[i]
        x, y = points_2d[i]
        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y])

    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape(3, 4)
    P /= P[-1, -1]

    num_points = len(points_3d)
    points_3d_hom = np.column_stack((points_3d, np.ones((num_points, 1))))
    points_2d_reprojected_hom = np.dot(P, points_3d_hom.T).T
    points_2d_reprojected = points_2d_reprojected_hom[:, :-1] / points_2d_reprojected_hom[:, -1:]
    reprojection_error = np.mean(np.linalg.norm(points_2d_reprojected - points_2d, axis=-1))
    print("Reprojection error: ", reprojection_error)

    return P


if __name__ == '__main__':
    # data_path = "trainset_random_14_pts_yolo_masks.h5"
    data_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints_reprojected_masks.h5"
    test_generators(data_path)
    # experiment(data_path)
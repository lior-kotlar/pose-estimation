import h5py
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.use('TkAgg')
from skimage.morphology import convex_hull_image
from skimage.morphology import disk, erosion, dilation

import sys
sys.path.append(r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code") # add to the end of the path
sys.path.insert(0, r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code") # insert at the beginning of the path
import traingulate


def load_dataset(data_path, X_dset="box", Y_dset="confmaps", permute=(0, 3, 2, 1)):
    """ Loads and normalizes datasets. """
    # Load
    with h5py.File(data_path, "r") as f:
        X = f[X_dset][:]
        Y = f[Y_dset][:]
    if X.shape[0] != 2:
    #     X = np.transpose(X, [4, 3, 2, 1, 0])
        Y = np.transpose(Y, [4, 3, 2, 1, 0])
    return X, Y


def save_new_box(box, path):
    with h5py.File(path, "a") as f:
        del f["box"]
        ds_conf = f.create_dataset("box", data=box, compression="gzip", compression_opts=1)
        ds_conf.attrs["description"] = "box"
        ds_conf.attrs["dims"] = f"{box.shape}"


def get_masks(wings_detection_model, img_3_ch):
    net_input = img_3_ch
    if np.max(img_3_ch) <= 1:
        net_input = np.round(255 * img_3_ch)
    results = wings_detection_model(net_input)[0]
    masks = results.masks.masks.numpy()[:2, :, :]
    return masks


def get_reprojection_masks(triangulate, points_3D, cropzone):
    points_2D_reprojected = triangulate.get_reprojections(points_3D, cropzone)
    num_frames, num_cams, num_points, _ = points_2D_reprojected.shape
    image_size = 192
    reprojected_masks = np.zeros((num_frames, num_cams, image_size, image_size, 2))

    num_wings_points = num_points - 2
    num_points_per_wing = num_wings_points // 2
    left_inds = np.arange(0, num_points_per_wing)
    right_inds = np.arange(num_points_per_wing, num_wings_points)
    wings_pnts_inds = np.array([left_inds, right_inds])

    for frame in range(num_frames):
        for cam in range(num_cams):
            for wing in range(2):
                points_inds = wings_pnts_inds[wing, :]
                mask = np.zeros((image_size, image_size))
                wing_pnts = np.round(points_2D_reprojected[frame, cam, points_inds, :]).astype(int)
                mask[wing_pnts[:, 1], wing_pnts[:, 0]] = 1
                mask = convex_hull_image(mask)
                mask = dilation(mask, footprint=np.ones((5, 5)))
                reprojected_masks[frame, cam, :, :, wing] = mask
    return reprojected_masks


def add_reprojected_masks_to_trainset(trainset_path, from_2D_to_3D_json):

    box, confmaps = load_dataset(trainset_path)
    points_3D = h5py.File(trainset_path, "r")["/points_3D"][:]
    points_3D = np.transpose(points_3D, [1, 2, 0])
    cropzone = h5py.File(trainset_path, "r")["/cropZone"][:]
    pass
    with open(from_2D_to_3D_json) as C:
        config = json.load(C)
    triangulate = traingulate.Triangulate(config)
    masks = get_reprojection_masks(triangulate, points_3D, cropzone)
    box[..., 3:] = masks

    save_new_box(box, trainset_path)


if __name__ == '__main__':

    # movie_trainset_path = r"train_set_movie_14_pts_yolo_masks.h5"



    random_trainset_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints_reprojected_masks.h5"
    from_2D_to_3D_json = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\2D_to_3D_config.json"
    with h5py.File(random_trainset_path, "r") as f:
        # Iterate over the attribute names
        for name in f.keys():
            print(name)

    add_reprojected_masks_to_trainset(random_trainset_path, from_2D_to_3D_json)


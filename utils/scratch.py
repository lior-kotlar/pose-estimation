import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

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
        X = np.transpose(X, [5, 4, 3, 2, 1, 0])
    if Y.shape[0] != 2:
        Y = np.transpose(Y, [5, 4, 3, 2, 1, 0])
    return X, Y


def create_wings_syncronized_dataset(box_path_movie_1, path_syncronized_box, save_path):
    global path, data, f
    import scipy.io
    data = scipy.io.loadmat(f"{path_syncronized_box}")
    new_box = data["box"].T
    prev_box = h5py.File(box_path_movie_1, "r")["box"][:]
    num_frames = new_box.shape[0]
    cam1 = np.transpose(prev_box[:, 0:3, :, :], [0, 2, 3, 1])
    cam2 = np.transpose(prev_box[:, 3:6, :, :], [0, 2, 3, 1])
    cam3 = np.transpose(prev_box[:, 6:9, :, :], [0, 2, 3, 1])
    cam4 = np.transpose(prev_box[:, 9:12, :, :], [0, 2, 3, 1])
    new_cam1 = np.zeros((num_frames, 192, 192, 5))
    new_cam1[:, :, :, 0:3] = cam1
    new_cam2 = np.zeros((num_frames, 192, 192, 5))
    new_cam2[:, :, :, 0:3] = cam2
    new_cam3 = np.zeros((num_frames, 192, 192, 5))
    new_cam3[:, :, :, 0:3] = cam3
    new_cam4 = np.zeros((num_frames, 192, 192, 5))
    new_cam4[:, :, :, 0:3] = cam4
    new_cam1[:, :, :, 3:5] = np.transpose(new_box[:, 0, :, :, :], [0, 2, 3, 1])[:, :, :, 1:]
    new_cam2[:, :, :, 3:5] = np.transpose(new_box[:, 1, :, :, :], [0, 2, 3, 1])[:, :, :, 1:]
    new_cam3[:, :, :, 3:5] = np.transpose(new_box[:, 2, :, :, :], [0, 2, 3, 1])[:, :, :, 1:]
    new_cam4[:, :, :, 3:5] = np.transpose(new_box[:, 3, :, :, :], [0, 2, 3, 1])[:, :, :, 1:]
    box_to_save = np.concatenate([new_cam1, new_cam2, new_cam3, new_cam4], axis=-1)
    with h5py.File(save_path, "w") as f:
        ds_pos = f.create_dataset("box", data=box_to_save, compression="gzip",
                                  compression_opts=1)

def get_masks(img_3_ch, model):
    net_input = img_3_ch
    masks_2 = np.zeros((2, 192, 192))
    if np.max(img_3_ch) <= 1:
        net_input = np.round(255 * img_3_ch)
    results = model(net_input)[0]

    # find if the train_masks detected are overlapping
    boxes = results.boxes.boxes.numpy()

    masks_found = results.masks.masks.numpy()[[0, 1], :, :]
    # add train_masks
    for wing in range(2):
        mask = masks_found[wing, :, :]
        score = results.boxes.boxes[wing, 4]
        masks_2[wing, :, :] = mask
        # else:
        # print(f"score = {score}")
        # matplotlib.use('TkAgg')
        # img_3_ch[:, :, 2] += mask
        # plt.imshow(img_3_ch)
        # plt.show()
    return masks_2

def use_tracker_to_add_wings():
    from ultralytics import YOLO
    import h5py
    import cv2
    from PIL import Image
    import supervision as sv
    import matplotlib.pyplot as plt
    import matplotlib
    import torch

    matplotlib.use('TkAgg')
    wings_detection_model_path = "wings_segmentation/YOLO models/wings_detection_yolov8_weights_13_3.pt"
    # box_path_no_masks = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\dataset_movie_14_frames_1301_2300_ds_3tc_7tj.h5"
    box_path_no_masks = r"../datasets/movies datasets/movie 17/movie_17_1401_2000_ds_5tc_14tj.h5";
    box = h5py.File(box_path_no_masks, "r")["/box"][:]
    # inds = [3,4,5]
    inds = [11, 12, 13]
    box = box[:100, inds, :, :]
    box_tensor = torch.from_numpy(255 * box)

    box = np.transpose(box, [0, 3, 2, 1])
    box_list = [box[i, :, :, :] for i in range(box.shape[0])]
    box_list = [np.round(255 * img) for img in box_list]
    box_list = [np.ascontiguousarray(img) for img in box_list]  # make sure the arrays are contiguous

    model = YOLO(wings_detection_model_path)
    model.fuse()
    # Run object detection on saved video
    results = model.predict(source=box_list, save=False, save_txt=False, max_det=2, retina_masks=True)
    # results = model.track(source=box_list,stream=False, persist=False, max_det=2)
    for i, result in enumerate(results):
        masks = result.masks.data.numpy()
        orig_img = result.orig_img/255
        for wing in range(min(masks.shape[0], 2)):
            mask = masks[wing, :, :]
            orig_img[:, :, wing] += 1 * mask
            orig_img[:, :, wing + 1] += 1 * mask
        plt.imshow(orig_img)
        plt.show()



def save_as_video():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import h5py
    matplotlib.use('TkAgg')
    wings_detection_model_path = "wings_segmentation/YOLO models/wings_detection_yolov8_weights_13_3.pt"
    box_path_no_masks = r"../datasets/movies datasets/movie 14/dataset_movie_14_frames_1301_2300_ds_3tc_7tj.h5"
    box = h5py.File(box_path_no_masks, "r")["/box"][:]
    # box[frame, 3 + np.array([0, 1, 2]), :, :].T)]

    fig, ax = plt.subplots()

    # Create an empty plot
    im = ax.imshow(box[0, 3 + np.array([0, 1, 2]), :, :].T, cmap='gray')

    # Define the update function
    def update(i):
        # im.set_data(box[i, 3 + np.array([0, 1, 2]), :, :].T)
        im = ax.imshow(box[i, 3 + np.array([0, 1, 2]), :, :].T, extent=[0, 1, 0, 1])
        return im,

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=10, repeat=True)

    # Save the animation as an mp4 video
    ani.save('animation.gif', writer='pillow', bbox_inches='tight')


def load_dataset_before_train(path):
    from scipy.ndimage import binary_erosion
    with h5py.File(path, "r") as f:
        print(list(f.keys()))
        box = f['/box'][:]
        c1 = np.transpose(box[:, [0, 1, 2, 3, 4], ...], [0, 2, 3, 1])
        c2 = np.transpose(box[:, [5, 6, 7, 8, 9], ...], [0, 2, 3, 1])
        c3 = np.transpose(box[:, [10, 11, 12, 13, 14], ...], [0, 2, 3, 1])
        c4 = np.transpose(box[:, [15, 16, 17, 18, 19], ...], [0, 2, 3, 1])
        box = np.concatenate((c1, c2, c3, c4), axis=0)
        new_box = np.zeros_like(box)
        num_images = box.shape[0]
        for im_num in range(3000, num_images):
            for masks_num in range(2):
                mask = box[im_num, :, :, 3 + masks_num]
                fly = box[im_num, :, :, 1].astype(bool)
                new_mask = binary_erosion(mask, iterations=3)
                new_mask = np.bitwise_and(new_mask, fly)
                box[im_num, :, :, 3 + masks_num] = new_mask
                plt.imshow(box[im_num, :, :, [1, 1, 3 + masks_num]].T)
                plt.show()
                pass

        pass


def create_self_supervision_dataset(images_path, masks_path1, masks_path2, saving_dir):
    from PIL import Image
    import os
    import re
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')


    # Get all the file names in the images and masks directories
    image_files = sorted(os.listdir(images_path))
    mask_files1 = sorted(os.listdir(masks_path1))
    mask_files2 = sorted(os.listdir(masks_path2))

    # Initialize empty lists to store the images and masks
    images = []
    masks = []

    # Loop over all the files
    for image_file, mask_file1, mask_file2 in zip(image_files, mask_files1, mask_files2):
        # Load the image and mask
        image = np.array(Image.open(os.path.join(images_path, image_file))).astype(float)/255
        mask1 = np.array(Image.open(os.path.join(masks_path1, mask_file1))).astype(float)
        mask2 = np.array(Image.open(os.path.join(masks_path2, mask_file2))).astype(float)

        # Append the image and mask to the respective lists
        # plt.imshow(image[..., 1] + mask1 + mask2)
        # plt.show()
        # Find all numbers in the string
        number = re.findall(r'\d+', image_file)[0]
        # Convert the numbers to integers
        new_image = np.concatenate((image, mask1[..., np.newaxis], mask2[..., np.newaxis]), axis=-1)
        np.save(os.path.join(saving_dir, f"image_{number}.npy"), new_image)
        pass



def get_gaussian(mean, sigma, grid_size=[192, 192]):
    x, y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]))
    d = np.sqrt((x - mean[0]) ** 2 + (y - mean[1]) ** 2)
    # Gaussian function
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    return g


def try_soft_argmax():
    import torch
    confmaps = np.zeros((40, 192, 192, 10))
    true_xy = np.zeros((40, 10, 2))
    for i in range(40):
        for channel in range(10):
            mean_x, mean_y = np.random.uniform(0, 192), np.random.uniform(0, 192)
            true_xy[i, channel, :] = [mean_x, mean_y]
            confmaps[i, :, :, channel] = get_gaussian(mean=[mean_x, mean_y], sigma=3)

    heatmap = torch.from_numpy(confmaps).float()

    # Adjust the dimensions to [batch_size, num_channels, height, width]
    heatmap = heatmap.permute(0, 3, 1, 2)

    batch_size, num_channels, height, width = heatmap.shape

    # Create normalized grids for x and y coordinates
    y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, steps=height),
                                    torch.linspace(0, 1, steps=width))
    y_grid, x_grid = y_grid.to(heatmap.device), x_grid.to(heatmap.device)

    # Compute the weighted sums for x and y coordinates across all images and channels
    weighted_sum_x = (x_grid * heatmap).sum(dim=[2, 3])
    weighted_sum_y = (y_grid * heatmap).sum(dim=[2, 3])

    # Compute the sum of all weights (pixel values) for normalization
    total_weight = heatmap.sum(dim=[2, 3])

    # Calculate the centroid coordinates
    centroid_x = weighted_sum_x / total_weight
    centroid_y = weighted_sum_y / total_weight

    # Convert normalized coordinates to image dimensions
    centroid_x = centroid_x * (width - 1)
    centroid_y = centroid_y * (height - 1)

    # Combine the coordinates
    centroids = torch.stack([centroid_x, centroid_y], dim=-1)


def get_preds_2D(confmaps):
    from Augmentor import Augmentor
    num_frames, num_cams, _,_, num_joints = confmaps.shape
    preds_2D = np.zeros((num_frames, num_cams, num_joints, 2))
    for frame in range(num_frames):
        preds_2D[frame, ...] = Augmentor.tf_find_peaks(confmaps[frame])
    return preds_2D



def create_new_dataset():
    sys.path.append(r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code")
    sys.path.append(
        r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\tensorflow")
    from visualize import Visualizer
    from predict_2D_sparse_box import Predictor2D
    from Augmentor import Augmentor

    data_path1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints.h5"
    data_path2 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints_reprojected_masks.h5"

    destination_h5_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints_combined.h5"

    # box1 = h5py.File(data_path1, "r")["/box"][:]
    # confmaps1 = h5py.File(data_path1, "r")["/confmaps"][:].T
    # preds1 = get_preds_2D(confmaps1)
    # box2 = h5py.File(data_path2, "r")["/box"][:]

    # Paths to the HDF5 files
    data_path1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints.h5"
    data_path2 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\random_trainset_201_frames_18_joints_reprojected_masks.h5"
    destination_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\combined_dataset.h5"

    # Datasets to exclude from concatenation
    excluded_datasets = {'camera_centers', 'cameras_dlt_array', 'cameras_inv_dlt_array', 'exptID', 'rotation_matrix',
                         'skeleton'}

    # Open the source HDF5 files
    with h5py.File(data_path1, 'r') as file1, h5py.File(data_path2, 'r') as file2:
        # Open the destination HDF5 file
        with h5py.File(destination_path, 'w') as dest_file:
            # Assume both files have the same datasets structure
            for name in file1:
                if name in excluded_datasets:
                    # Directly copy excluded datasets from the first file
                    file1.copy(name, dest_file)
                else:
                    # Concatenate datasets across the 0 axis
                    data1 = file1[name][:]
                    data2 = file2[name][:]
                    if name == 'confmaps' or name == 'joints':
                        combined_data = np.concatenate([data1, data2], axis=-1)
                    elif name == 'points_3D':
                        combined_data = np.concatenate([data1, data2], axis=1)
                    else:
                        combined_data = np.concatenate([data1, data2], axis=0)
                    # Create a new dataset in the destination file with the combined data
                    dest_file.create_dataset(name, data=combined_data, compression="gzip",
                                      compression_opts=1)


if __name__ == "__main__":
    destination_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\train_pose_estimation\training_datasets\combined_dataset.h5"
    create_new_dataset()
    bb = h5py.File(destination_path, 'r')["/confmaps"][:]
    pass
    # try_soft_argmax()
    # images_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\create_annotations\input\train_images"
    # masks_path1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\create_annotations\input\train_masks\wings1"
    # masks_path2 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\create_annotations\input\train_masks\wings2"
    # saving_dir = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\create_annotations\input\as_numpy_arrays"
    # create_self_supervision_dataset(images_path, masks_path1, masks_path2, saving_dir)

    # use_tracker_to_add_wings()
    # save_as_video()

    # box_path_movie_1 = r"../datasets/movies datasets/roni movie 6/movie_6_1001_1500_ds_3tc_7tj.h5"
    # path_syncronized_box = r"../datasets/movies datasets/roni movie 6/sync_box.mat"
    # save_path = r"../datasets/movies datasets/roni movie 6/sync_box_to_predict.mat"
    # create_wings_syncronized_dataset(box_path_movie_1, path_syncronized_box, save_path)
    # path = r"C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\trainset 18 points\pre_train_1000_frames_5_channels_ds_3tc_7tj.h5"
    # load_dataset_before_train(path)

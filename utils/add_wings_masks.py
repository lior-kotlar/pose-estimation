import h5py
from ultralytics import YOLO
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import make_smoothing_spline
from scipy.signal import medfilt
from scipy.ndimage import center_of_mass, shift, gaussian_filter
import sys
import os
sys.path.append(r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code")
import visualize
NUM_CAMS = 4


class Add_Masks:
    def __init__(self, wings_detection_model_path,
                 box_path):
        self.wings_detection_model_path = wings_detection_model_path
        self.box_path = box_path
        self.box = self.get_box()
        self.wings_detection_model = self.get_wings_detection_model()
        self.cropzone = self.get_cropzone()
        self.num_frames, self.num_cams, _, self.im_size, self.num_times_channels = self.box.shape
        self.num_cams = NUM_CAMS
        self.scores = np.zeros((self.num_frames, self.num_cams, 2))

    def detect_masks_and_save(self):
        print("aligning time channels")
        self.align_time_channels()
        print("adding masks")
        self.add_masks()
        print("saving")
        self.save_masks_to_h5()

    def get_box(self):
        box = h5py.File(self.box_path, "r")["/box"][:]
        if len(box.shape) == 5:
            return box[..., :-2]
        assert len(box.shape) != 5
        box = np.transpose(box, (0, 3, 2, 1))
        x1 = np.expand_dims(box[:, :, :, 0:3], axis=1)
        x2 = np.expand_dims(box[:, :, :, 3:6], axis=1)
        x3 = np.expand_dims(box[:, :, :, 6:9], axis=1)
        x4 = np.expand_dims(box[:, :, :, 9:12], axis=1)
        box = np.concatenate((x1, x2, x3, x4), axis=1)
        return box

    def get_cropzone(self):
        return h5py.File(self.box_path, "a")["/cropzone"]

    @staticmethod
    def get_fly_cm(im_orig):
        im = gaussian_filter(im_orig, sigma=2)
        im[im < 0.8] = 0
        # im = binary_opening(im, iterations=1)
        CM = center_of_mass(im)
        return np.array(CM)

    def align_time_channels(self):
        all_shifts = np.zeros((self.num_frames, self.num_cams, 2, 2))
        all_shifts_smoothed = np.zeros((self.num_frames, self.num_cams, 2, 2))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                present = self.box[frame, cam, :, :, 1]
                cm_present = self.get_fly_cm(present)
                for i, time_channel in enumerate([0, 2]):
                    fly = self.box[frame, cam, :, :, time_channel]
                    CM = self.get_fly_cm(fly)
                    shift_to_do = cm_present - CM
                    all_shifts[frame, cam, i, :] = shift_to_do
        # do shiftes
        for cam in range(self.num_cams):
            for time_channel in range(all_shifts.shape[2]):
                for axis in range(all_shifts.shape[3]):
                    vals = all_shifts[:, cam, time_channel, axis]
                    A = np.arange(vals.shape[0])
                    filtered = medfilt(vals, kernel_size=11)
                    # all_shifts_smoothed[:, cam, time_channel, axis] = filtered
                    spline = make_smoothing_spline(A, filtered, lam=10000)
                    smoothed = spline(A)
                    all_shifts_smoothed[:, cam, time_channel, axis] = smoothed
                    pass

        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for i, time_channel in enumerate([0, 2]):
                    fly = self.box[frame, cam, :, :, time_channel]
                    shift_to_do = all_shifts_smoothed[frame, cam, i, :]
                    shifted_fly = shift(fly, shift_to_do, order=2)
                    self.box[frame, cam, :, :, time_channel] = shifted_fly
        # Visualizer.display_movie_from_box(self.box)

    def get_wings_detection_model(self):
        """ load a pretrained YOLOv8 segmentation model"""
        model = YOLO(self.wings_detection_model_path)
        model.fuse()
        return model

    def save_masks_to_h5(self):
        f = h5py.File(self.box_path, "a")
        try:
            masks_dset = f.create_dataset("scores", data=self.scores, compression="gzip",
                                      compression_opts=1)
            masks_dset.attrs["description"] = "a score from 0->1 for each mask (left right according to train_masks order)"
            masks_dset.attrs["dims"] = f"{self.scores.shape}"
        except:
            print("scores already exists")
        del f[r'/box']
        masks_dset = f.create_dataset("box", data=self.box, compression="gzip",
                                      compression_opts=1)
        f.close()

    def add_masks(self):
        """ Add train_masks to the dataset using yolov8 segmentation model """
        new_box = np.zeros((self.num_frames, self.num_cams, self.im_size, self.im_size, self.num_times_channels + 2))
        # visualize.Visualizer.display_movie_from_box(self.box)
        for cam in range(self.num_cams):
            print(f"finds wings for camera number {cam + 1}")
            img_3_ch_all = self.box[:, cam, :, :, :]
            new_box[:, cam, :, :, :self.num_times_channels] = img_3_ch_all
            # split to avoid memory limit
            n = 5
            img_3_ch_all_split = np.array_split(img_3_ch_all, n)
            results = []
            for i in range(n):
                img_3_ch_i = img_3_ch_all_split[i]
                img_3_ch_input = np.round(img_3_ch_i * 255)
                img_3_ch_input = [img_3_ch_input[i] for i in range(img_3_ch_input.shape[0])]
                results_i = self.wings_detection_model(img_3_ch_input)
                results.append(results_i)
            results = sum(results, [])
            masks = np.zeros((self.num_frames, self.im_size, self.im_size, 2))
            for frame in range(self.num_frames):
                masks_2 = np.zeros((self.im_size, self.im_size, 2))
                result = results[frame]
                boxes = result.boxes.data.numpy()
                inds_to_keep = self.eliminate_close_vectors(boxes, 10)
                num_wings_found = np.count_nonzero(inds_to_keep)
                if num_wings_found > 0:
                    masks_found = result.masks.data.numpy()[inds_to_keep, :, :]
                for wing in range(min(num_wings_found, 2)):
                    mask = masks_found[wing, :, :]
                    score = result.boxes.data[wing, 4]
                    masks_2[:, :, wing] = mask
                    self.scores[frame, cam, wing] = score
                masks[frame, :, :, :] = masks_2
            new_box[:, cam, :, :, self.num_times_channels:] = masks
        self.box = new_box

    @staticmethod
    def eliminate_close_vectors(matrix, threshold):
        # calculate pairwise Euclidean distances
        distances = cdist(matrix, matrix, 'euclidean')
        # create a mask to identify which vectors to keep
        inds_to_del = np.ones(len(matrix), dtype=bool)
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if distances[i, j] < threshold:
                    # eliminate one of the vectors
                    inds_to_del[j] = False

        # return the new matrix with close vectors eliminated
        return inds_to_del


def add_masks_all_movies(base_path, model_path):
    # Loop through the directory and its subdirectories
    file_list = []
    for root, dirs, files in os.walk(base_path):
        # Loop through the files
        for file in files:
            # Check if the file name starts with "movie_"
            if file.startswith("movie_"):
                # Join the root and file name to get the full path
                full_path = os.path.join(root, file)
                # Append the full path to the list
                file_list.append(full_path)

    for movie_path in file_list:
        print(movie_path)
        add_masks_obj = Add_Masks(model_path, movie_path)
        add_masks_obj.detect_masks_and_save()


if __name__ == "__main__":
    # box_path = r"G:\My Drive\Amitai\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies\mov1\movie_1_201_1045_ds_3tc_7tj.h5"
    #
    # add_masks_obj = Add_Masks(model_path, box_path)
    # add_masks_obj.detect_masks_and_save()
    model_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\YOLO models\yolo_weights_6_1_24.pt"
    base_path = r"G:\My Drive\Amitai\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies"
    add_masks_all_movies(base_path, model_path)

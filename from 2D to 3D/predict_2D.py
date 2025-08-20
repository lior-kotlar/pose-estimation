import matplotlib.pyplot as plt
import scipy.ndimage
import scipy
import yaml
import h5py
import json
import h5py
import numpy as np
from visualize import Visualizer
import os
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from scipy.interpolate import make_smoothing_spline
from skimage import util, measure
import tensorflow as tf
# from training import preprocess
# from leap.utils import find_weights, find_best_weights, preprocess
# from leap.layers import Maxima2D
from scipy.spatial.distance import cdist
from scipy.io import loadmat
# imports of the wings1 detection
from time import time
from ultralytics import YOLO
# import open3d as o3d
from scipy.signal import medfilt
from scipy.ndimage import binary_dilation, binary_closing, center_of_mass, shift, gaussian_filter, binary_opening
from datetime import date
import shutil

# from scipy.spatial.distance import pdist
# from scipy.ndimage.measurements import center_of_mass
# from scipy.spatial import ConvexHull
# import matplotlib
# import cv2
# import preprocessor
from constants import *
import predictions_2Dto3D


class Predictor2D:
    def __init__(self, configuration_path):

        with open(configuration_path) as C:
            config = json.load(C)
            self.config = config
            self.box_path = config["box path"]
            self.wings_pose_estimation_model_path = config["wings pose estimation model path"]
            self.wings_pose_estimation_model_path_second_pass = config["wings pose estimation model path second path"]
            self.head_tail_pose_estimation_model_path = config["head tail pose estimation model path"]
            # self.out_path = config["out path"]
            self.wings_detection_model_path = config["wings detection model path"]
            self.model_type = config["model type"]
            self.model_type_second_pass = config["model type second pass"]
            self.is_video = bool(config["is video"])
            self.batch_size = config["batch size"]
            self.points_to_predict = config["body parts to predict"]
            self.num_cams = config["number of cameras"]
            self.num_times_channels = config["number of time channels"]
            self.mask_increase_initial = config["mask increase initial"]
            self.mask_increase_reprojected = config["mask increase reprojected"]
            self.predict_again_using_reprojected_masks = bool(config["use reprojected masks"])
            self.base_output_path = config["base output path"]
            self.json_2D_3D_path = config["2D to 3D config path"]

        first_frame = 0
        last_frame = -1
        self.box = self.get_box()[first_frame:last_frame]
        self.masks_flag = True if self.box.shape[-1] == 5 else False
        # Visualizer.display_movie_from_box(np.copy(self.box))
        self.cropzone = self.get_cropzone()[first_frame:last_frame]
        self.im_size = self.box.shape[2]
        self.num_frames = self.box.shape[0]
        self.num_pass = 0

        self.wings_pose_estimation_model = \
            Predictor2D.get_pose_estimation_model(self.wings_pose_estimation_model_path)
        if self.model_type != WINGS_AND_BODY_SAME_MODEL:
            self.head_tail_pose_estimation_model = \
                Predictor2D.get_pose_estimation_model(self.head_tail_pose_estimation_model_path)
        self.wings_detection_model = self.get_wings_detection_model()
        self.scores = np.zeros((self.num_frames, self.num_cams, 2))
        self.predict_method = self.choose_predict_method()

        self.run_name = self.get_run_name()

        self.total_runtime = None
        self.prediction_runtime = None
        self.predicted_points = None
        self.out_path_h5 = None

    def run_predict_2D(self):
        """
        creates an array of pose estimation predictions
        """
        t0 = time()
        self.run_path = self.create_run_folders()
        self.preprocess_box()
        if not self.masks_flag:
            self.clean_images()
            self.align_time_channels()
        # Visualizer.display_movie_from_box(np.copy(self.box))
        self.preprocess_masks()
        preprocessing_time = time() - t0
        preds_time = time()
        print("preprocess [%.1fs]" % preprocessing_time)
        self.predicted_points = self.predict_method()
        print("finish predict")
        self.prediction_runtime = 0
        self.save_predictions_to_h5()
        # Visualizer.show_predictions_all_cams(self.box, self.predicted_points)
        if self.predict_again_using_reprojected_masks:
            self.num_pass += 1
            self.model_type = self.model_type_second_pass
            self.predict_method = self.choose_predict_method()
            return_model_peaks = False if self.model_type == ALL_CAMS_PER_WING else True
            self.wings_pose_estimation_model = \
                self.get_pose_estimation_model(self.wings_pose_estimation_model_path_second_pass,
                                               return_model_peaks=return_model_peaks)
            json_2D_to_3D_config = self.create_2D_3D_config()
            print("starting reprojection masks")
            predictor = predictions_2Dto3D.From2Dto3D(load_from=CONFIG,
                                                      configuration_path=json_2D_to_3D_config)
            points_3D = predictor.choose_best_score_2_cams()
            smoothed_3D = predictor.smooth_3D_points(points_3D)
            reprojection_masks = predictor.get_reprojection_masks(smoothed_3D,
                                                                  self.mask_increase_reprojected)
            predictor.box[..., [3, 4]] = reprojection_masks
            self.box = predictor.box
            print("created reprojection masks")
            self.predicted_points = self.predict_method()
            self.save_predictions_to_h5()

        # Visualizer.show_predictions_all_cams(self.box, self.predicted_points)
        self.prediction_runtime = time() - preds_time
        self.total_runtime = time() - t0
        print("Predicted [%.1fs]" % self.prediction_runtime)
        print("Prediction performance: %.3f FPS" % (self.num_frames * self.num_cams / self.prediction_runtime))

    def clean_images(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for channel in range(3):
                    image = self.box[frame, cam, :, :, channel]
                    binary = np.where(image >= 0.1, 1, 0)
                    label = measure.label(binary)
                    props = measure.regionprops(label)
                    sizes = [prop.area for prop in props]
                    largest = np.argmax(sizes)
                    fly_component = np.where(label == largest + 1, 1, 0)
                    image = image * fly_component
                    self.box[frame, cam, :, :, channel] = image

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
                    try:
                        spline = make_smoothing_spline(A, filtered, lam=10000)
                        smoothed = spline(A)
                    except:
                        smoothed = filtered
                        print(f"spline failed in cam {cam} time channel {time_channel} and axis {axis}")
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

    @staticmethod
    def get_fly_cm(im_orig):
        im = gaussian_filter(im_orig, sigma=2)
        im[im < 0.8] = 0
        # im = binary_opening(im, iterations=1)
        CM = center_of_mass(im)
        return np.array(CM)

    def preprocess_masks(self):
        if self.points_to_predict == WINGS or self.points_to_predict == WINGS_AND_BODY:
            if not self.masks_flag:
                self.add_masks()
            self.adjust_masks_size()
            if self.is_video:
                self.fix_masks()

    def get_run_name(self):
        box_path_file = os.path.basename(self.box_path)
        name, ext = os.path.splitext(box_path_file)
        run_name = f"{name}_{self.model_type}_{date.today().strftime('%b %d')}"
        return run_name

    def create_2D_3D_config(self):
        json_path = self.json_2D_3D_path
        with open(json_path, "r") as jsonFile:
            data = json.load(jsonFile)

        # Change the values of some variables
        data["2D predictions path"] = self.out_path_h5
        data["align right left"] = 1

        new_json_path = os.path.join(self.run_path, "2D_to_3D_config.json")
        # Save the JSON string to a different file
        with open(new_json_path, "w") as jsonFile:
            json.dump(data, jsonFile)
        return new_json_path

    def create_run_folders(self):
        """ Creates subfolders necessary for outputs of vision. """
        run_path = os.path.join(self.base_output_path, self.run_name)

        initial_run_path = run_path
        i = 1
        while os.path.exists(run_path):  # and not is_empty_run(run_path):
            run_path = "%s_%02d" % (initial_run_path, i)
            i += 1

        if os.path.exists(run_path):
            shutil.rmtree(run_path)

        os.makedirs(run_path)
        print("Created folder:", run_path)

        return run_path

    def save_predictions_to_h5(self):
        """ save the predictions and the box of train_images to h5 file"""
        if self.num_pass > 0:
            name = "predicted_points_and_box_reprojected.h5"
        else:
            name = "predicted_points_and_box.h5"
        self.out_path_h5 = os.path.join(self.run_path, name)
        with open(f"{self.run_path}/configuration_short.json", 'w') as file:
            json.dump(self.config, file, indent=4)
        with h5py.File(self.out_path_h5, "w") as f:
            f.attrs["num_frames"] = self.box.shape[0]
            f.attrs["img_size"] = self.im_size
            f.attrs["box_path"] = self.box_path
            f.attrs["box_dset"] = "/box"
            f.attrs["pose_estimation_model_path"] = self.wings_pose_estimation_model_path
            f.attrs["wings_detection_model_path"] = self.wings_detection_model_path

            positions = self.predicted_points[..., :2]
            confidence_val = self.predicted_points[..., 2]

            ds_pos = f.create_dataset("positions_pred", data=positions.astype("int32"), compression="gzip",
                                      compression_opts=1)
            ds_pos.attrs["description"] = "coordinate of peak at each sample"
            ds_pos.attrs["dims"] = "(sample, joint, [x, y])"

            ds_conf = f.create_dataset("conf_pred", data=confidence_val.squeeze(), compression="gzip",
                                       compression_opts=1)
            ds_conf.attrs["description"] = "confidence map value in [0, 1.0] at peak"
            ds_conf.attrs["dims"] = "(frame, cam, joint)"

            ds_conf = f.create_dataset("box", data=self.box, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "The predicted box and the wings1 if the wings1 were detected"
            ds_conf.attrs["dims"] = f"{self.box.shape}"

            if self.points_to_predict == WINGS or self.points_to_predict == WINGS_AND_BODY:
                ds_conf = f.create_dataset("scores", data=self.scores, compression="gzip", compression_opts=1)
                ds_conf.attrs["description"] = "the score (0->1) assigned to each wing during wing detection"
                ds_conf.attrs["dims"] = f"{self.scores.shape}"

            ds_conf = f.create_dataset("cropzone", data=self.cropzone, compression="gzip", compression_opts=1)
            ds_conf.attrs["description"] = "cropzone of every image for 2D to 3D projection"
            ds_conf.attrs["dims"] = f"{self.cropzone.shape}"
            # f.attrs["prediction_runtime_secs"] = self.prediction_runtime
            # f.attrs["total_runtime_secs"] = self.total_runtime

    def choose_predict_method(self):
        if self.points_to_predict == WINGS:
            return self.predict_wings
        elif self.points_to_predict == BODY:
            return self.predict_body
        elif self.points_to_predict == WINGS_AND_BODY:
            if self.model_type == WINGS_AND_BODY_SAME_MODEL:
                return self.predict_wings_and_body_same_model
            if self.model_type == ALL_POINTS or self.model_type == ALL_POINTS_REPROJECTED_MASKS:
                return self.predict_all_points
            if self.model_type == ALL_CAMS_PER_WING:
                return self.predict_all_cams_per_wing
            return self.predict_wings_and_body

    def predict_all_cams_per_wing(self):
        n = 8
        print(f"started predicting projected masks, split box into {n} parts")
        splited_box = np.array_split(self.box, n)
        all_points = []
        for i in range(n):
            print(f"predicting part number {i + 1}")
            box = splited_box[i]
            all_points_i = []
            for wing in range(2):
                input_wing = box[..., [0, 1, 2, self.num_times_channels + wing]]
                input_wing = np.concatenate((input_wing[:, 0, ...],
                                             input_wing[:, 1, ...],
                                             input_wing[:, 2, ...],
                                             input_wing[:, 3, ...]), axis=-1)
                output = self.wings_pose_estimation_model(input_wing)
                peaks = self.tf_find_peaks(output)
                peaks_list = [peaks[..., 0:10],
                              peaks[..., 10:20],
                              peaks[..., 20:30],
                              peaks[..., 30:40]]
                for cam in range(self.num_cams):
                    peaks_list[cam] = np.expand_dims(peaks_list[cam], axis=1)
                peaks_wing = np.concatenate(peaks_list, axis=1)
                all_points_i.append(peaks_wing)
            all_points_i = np.concatenate((all_points_i[0], all_points_i[1]), axis=-1)
            tail_points = all_points_i[..., [8, 18]]
            tail_points = np.expand_dims(np.mean(tail_points, axis=-1), axis=-1)
            head_points = all_points_i[..., [9, 19]]
            head_points = np.expand_dims(np.mean(head_points, axis=-1), axis=-1)
            wings_points = all_points_i[..., [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]]
            wings_and_body_pnts = np.concatenate((wings_points, tail_points, head_points), axis=-1)
            wings_and_body_pnts = np.transpose(wings_and_body_pnts, [0, 1, 3, 2])
            all_points.append(wings_and_body_pnts)
        all_wing_and_body_points = np.concatenate(all_points, axis=0)
        print("done predicting projected masks")
        return all_wing_and_body_points

    def predict_all_points(self):
        all_points = []
        for cam in range(self.num_cams):
            input = self.box[:, cam, ...]
            points_cam_i, _, _, _ = self.predict_Ypk(input, self.batch_size, self.wings_pose_estimation_model)
            all_points.append(points_cam_i[np.newaxis, ...])
        wings_and_body_pnts = np.concatenate(all_points, axis=0)
        wings_and_body_pnts = np.transpose(wings_and_body_pnts, [1, 0, 3, 2])
        return wings_and_body_pnts

    def predict_wings_and_body_same_model(self):
        all_pnts = self.predict_wings()
        tail_points = all_pnts[:, :, [8, 18], :]
        tail_points = np.expand_dims(np.mean(tail_points, axis=2), axis=2)
        head_points = all_pnts[:, :, [9, 19], :]
        head_points = np.expand_dims(np.mean(head_points, axis=2), axis=2)
        wings_points = all_pnts[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17], :]
        wings_and_body_pnts = np.concatenate((wings_points, tail_points, head_points), axis=2)
        return wings_and_body_pnts

    def predict_wings_and_body(self):
        wings_points = self.predict_wings()
        body_points = self.predict_body()
        wings_and_body_pnts = np.concatenate((wings_points, body_points), axis=2)
        return wings_and_body_pnts

    def predict_wings(self):
        Ypks = []
        for cam in range(self.num_cams):
            print(f"predict camera {cam + 1}")
            Ypks_per_wing = []
            for wing in range(2):
                input = np.transpose(self.box[:, cam, :, :, [0, 1, 2, self.num_times_channels + wing]], [1, 2, 3, 0])
                # split for memory limit
                n = 1
                input_split = np.array_split(input, n)
                Ypk = []
                for i in range(n):
                    Ypk_i, _, _, _ = self.predict_Ypk(input_split[i], self.batch_size, self.wings_pose_estimation_model)
                    Ypk.append(Ypk_i)
                Ypk = np.concatenate(Ypk, axis=0)
                Ypks_per_wing.append(Ypk)
            Ypk_cam = np.concatenate((Ypks_per_wing[0], Ypks_per_wing[1]), axis=-1)
            Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
            Ypks.append(Ypk_cam)
        Ypk_all = np.concatenate(Ypks, axis=1)
        Ypk_all = np.transpose(Ypk_all, [0, 1, 3, 2])
        return Ypk_all

    def predict_body(self):
        Ypks = []
        for cam in range(self.num_cams):
            input = self.box[:, cam, :, :, :self.num_times_channels]
            Ypk_cam, _, _, _ = self.predict_Ypk(input, self.batch_size, self.head_tail_pose_estimation_model)
            Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
            Ypks.append(Ypk_cam)
        Ypk_all = np.concatenate(Ypks, axis=1)
        Ypk_all = np.transpose(Ypk_all, [0, 1, 3, 2])
        return Ypk_all

    def add_masks(self):
        """ Add train_masks to the dataset using yolov8 segmentation model """
        new_box = np.zeros((self.num_frames, self.num_cams, self.im_size, self.im_size, self.num_times_channels + 2))
        for cam in range(self.num_cams):
            print(f"finds wings for camera number {cam+1}")
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

    def adjust_masks_size(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                mask_1 = self.box[frame, cam, :, :, self.num_times_channels]
                mask_2 = self.box[frame, cam, :, :, self.num_times_channels + 1]
                mask_1 = self.adjust_mask(mask_1, self.mask_increase_initial)
                mask_2 = self.adjust_mask(mask_2, self.mask_increase_initial)
                self.box[frame, cam, :, :, self.num_times_channels] = mask_1
                self.box[frame, cam, :, :, self.num_times_channels + 1] = mask_2

    def fix_masks(self):  # todo find out if there are even train_masks to be fixed
        """
            goes through each frame, if there is no mask for a specific wing, unite train_masks of the closest times before and after
            this frame.
            :param X: a box of size (num_frames, 20, 192, 192)
            :return: same box
            """
        search_range = 5
        problematic_masks = []
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for mask_num in range(2):
                    mask = self.box[frame, cam, :, :, self.num_times_channels + mask_num]
                    if np.all(mask == 0):  # check if all 0:
                        problematic_masks.append((frame, cam, mask_num))
                        # find previous matching mask
                        prev_mask = np.zeros(mask.shape)
                        next_mask = np.zeros(mask.shape)
                        for prev_frame in range(frame - 1, max(0, frame - search_range - 1), -1):
                            prev_mask_i = self.box[prev_frame, cam, :, :, self.num_times_channels + mask_num]
                            if not np.all(prev_mask_i == 0):  # there is a good mask
                                prev_mask = prev_mask_i
                                break
                        # find next matching mask
                        for next_frame in range(frame + 1, min(self.num_frames, frame + search_range)):
                            next_mask_i = self.box[next_frame, cam, :, :, self.num_times_channels + mask_num]
                            if not np.all(next_mask_i == 0):  # there is a good mask
                                next_mask = next_mask_i
                                break
                        # combine the 2 train_masks

                        new_mask = prev_mask + next_mask  # todo changed it from : prev_mask + next_mask
                        new_mask[new_mask >= 1] = 1

                        sz_prev_mask = np.count_nonzero(prev_mask)
                        sz_next_mask = np.count_nonzero(next_mask)
                        sz_new_mask = np.count_nonzero(new_mask)
                        if sz_prev_mask + sz_next_mask == sz_new_mask:
                            # it means that the train_masks are not overlapping
                            new_mask = prev_mask if sz_prev_mask > sz_next_mask else next_mask

                        # replace empty mask with new mask
                        self.box[frame, cam, :, :, self.num_times_channels + mask_num] = new_mask

    def get_box(self):
        box = h5py.File(self.box_path, "r")["/box"]
        if len(box.shape) == 5:
            print("box already has wings masks")
            return box
        box = np.transpose(box, (0, 3, 2, 1))
        x1 = np.expand_dims(box[:, :, :, 0:3], axis=1)
        x2 = np.expand_dims(box[:, :, :, 3:6], axis=1)
        x3 = np.expand_dims(box[:, :, :, 6:9], axis=1)
        x4 = np.expand_dims(box[:, :, :, 9:12], axis=1)
        box = np.concatenate((x1, x2, x3, x4), axis=1)
        return box

    def get_cropzone(self):
        return h5py.File(self.box_path, "r")["/cropzone"]

    def get_wings_detection_model(self):
        """ load a pretrained YOLOv8 segmentation model"""
        if self.num_pass == 0:
            model = YOLO(self.wings_detection_model_path)
            model.fuse()
        elif self.num_pass == 1:
            model = YOLO(self.wings_pose_estimation_model_path_second_pass)
            model.fuse()
        return model

    def preprocess_box(self):
        # Add singleton dim for single train_images
        if self.box.ndim == 3:
            self.box = self.box[None, ...]
        if self.box.dtype == "uint8" or np.max(self.box) > 1:
            self.box = self.box.astype("float32") / 255


    @staticmethod
    def adjust_mask(mask, radius=3):
        mask = binary_closing(mask).astype(int)
        mask = binary_dilation(mask, iterations=radius).astype(int)
        return mask

    @staticmethod
    def get_pose_estimation_model(pose_estimation_model_path, return_model_peaks=True):
        """ load a pretrained LEAP pose estimation model model"""
        model = keras.models.load_model(pose_estimation_model_path)
        if return_model_peaks:
            model = Predictor2D.convert_to_peak_outputs(model, include_confmaps=False)
        print("weights_path:", pose_estimation_model_path)
        print("Loaded model: %d layers, %d params" % (len(model.layers), model.count_params()))
        return model

    @staticmethod
    def convert_to_peak_outputs(model, include_confmaps=False):
        """ Creates a new Keras model with a wrapper to yield channel peaks from rank-4 tensors. """
        if type(model.output) == list:
            confmaps = model.output[-1]
        else:
            confmaps = model.output

        if include_confmaps:
            return keras.Model(model.input, [Lambda(Predictor2D.tf_find_peaks)(confmaps), confmaps])
        else:
            return keras.Model(model.input, Lambda(Predictor2D.tf_find_peaks)(confmaps))


    @staticmethod
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
            vals],
            axis=1)
        return pred

    @staticmethod
    def predict_Ypk(X, batch_size, model_peaks, save_confmaps=False):
        """ returns a predicted dataset"""
        confmaps, confmaps_min, confmaps_max = None, None, None
        if save_confmaps:
            Ypk, confmaps = model_peaks.predict(X, batch_size=batch_size)

            # Quantize
            confmaps_min = confmaps.min()
            confmaps_max = confmaps.max()

            # Reshape
            confmaps = np.transpose(confmaps, (0, 3, 2, 1))
        else:
            Ypk = model_peaks.predict(X, batch_size=batch_size)
        return Ypk, confmaps, confmaps_min, confmaps_max

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


def config_1(config):
    # 3 good cameras 1
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 0
    return config


def config_2(config):
    # 3 good cameras 1
    config["wings pose estimation model path"] = r"models/3 good cameras/MODEL_18_POINTS_3_GOOD_CAMERAS_Jan 03/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 0
    return config


def config_3(config):
    # 3 good cameras 2
    config["wings pose estimation model path"] = r"models/3 good cameras/MODEL_18_POINTS_3_GOOD_CAMERAS_Jan 03_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 0
    return config


def config_4(config):
    # 2 passes reprojected masks
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = "models/per wing/MODEL_18_POINTS_PER_WING_Jan 20/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 1
    return config


def config_5(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/4 cameras/concatenated encoder/ALL_CAMS_18_POINTS_Jan 19_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again using reprojected masks"] = 1
    return config


def config_6(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/4 cameras/concatenated encoder/ALL_CAMS_18_POINTS_Jan 20_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again using reprojected masks"] = 1
    return config

def predict_all_movies(base_path, config_path_2D):
    import predictions_2Dto3D
    file_list = []
    # movies_dir = 'movies'
    # os.listdir(base_path)
    dirs = ['mov50']
    for sub_dir in dirs:
        # Join the subdirectory name with the movies_dir path
        sub_dir_path = os.path.join(base_path, sub_dir)
        # Check if the subdirectory is actually a directory
        if os.path.isdir(sub_dir_path):
            # Loop over all the files in the subdirectory
            for file in os.listdir(sub_dir_path):
                # Check if the file name starts with 'movie' and ends with '.h5'
                if file.startswith('movie') and file.endswith('.h5'):
                    # Join the file name with the subdirectory path
                    file_path = os.path.join(sub_dir_path, file)
                    # Append the full path of the file to the list
                    file_list.append(file_path)

    # config_functions = [config_1, config_2, config_3, config_4, config_5, config_6]
    config_functions = [config_1, config_2, config_3]
    for movie_path in file_list:
        print(movie_path)
        for model in range(len(config_functions)):
            dir_path = os.path.dirname(movie_path)
            with open(config_path_2D) as C:
                config_2D = json.load(C)
                config_2D["box path"] = movie_path
                config_2D["base output path"] = dir_path
                config_func = config_functions[model]
                config_2D = config_func(config_2D)
            new_config_path = os.path.join(dir_path, 'configuration predict 2D.json')
            with open(new_config_path, 'w') as file:
                json.dump(config_2D, file, indent=4)
            predictor = Predictor2D(new_config_path)
            predictor.run_predict_2D()



if __name__ == '__main__':
    config_path = r"from 2D to 3D/predict_2D_config.json"
    predictor = Predictor2D(config_path)
    predictor.run_predict_2D()

    # predict roni data
    # config_path = r"predict_2D__pytorch_config.json"  # get the first argument
    # base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data"
    # predict_all_movies(base_path, config_path)

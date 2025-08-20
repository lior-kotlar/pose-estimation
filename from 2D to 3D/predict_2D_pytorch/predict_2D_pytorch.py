import json
import h5py
import numpy as np
from visualize2 import Visualizer
import os
from scipy.interpolate import make_smoothing_spline
from skimage import measure
# from training import preprocess
# from leap.utils import find_weights, find_best_weights, preprocess
# from leap.layers import Maxima2D
from scipy.spatial.distance import cdist
# imports of the wings1 detection
from time import time
from ultralytics import YOLO
from scipy.signal import medfilt
from scipy.ndimage import binary_dilation, binary_closing, center_of_mass, shift, gaussian_filter
from datetime import date
import shutil
import torch
from utils_pytorch import torch_find_peaks_argmax, DecomposeCamera
from BoxSparse import BoxSparse
from constants import *
from traingulate2 import Triangulate


# Now you can import triangulation.py as if it were in the same directory
WHICH_TO_FLIP = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).astype(bool)
ALL_COUPLES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

class Predictor2D:
    def __init__(self, configuration_path):

        self.ts = None
        self.Rs = None
        self.Ks = None
        self.predicted_points = None
        self.masks_flag = None
        self.run_path = None
        with open(configuration_path) as C:
            config = json.load(C)
            self.config = config
            self.triangulate = Triangulate(self.config)
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
            self.predict_again_using_reprojected_masks = bool(config["predict again using reprojected masks"])
            self.base_output_path = config["base output path"]
            self.calibration_data_path = self.config["calibration data path"]
            self.predict_method = self.choose_predict_method()

        self.box_sparse = BoxSparse(self.box_path)
        self.cropzone = self.get_cropzone()
        self.im_size = self.box_sparse.shape[2]
        self.num_frames = self.box_sparse.shape[0]
        # self.masks_flag = True if self.box_sparse.shape[-1] == 5 else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wings_pose_estimation_model = self.load_model()
        self.wings_detection_model = self.get_wings_detection_model()
        self.run_name = self.get_run_name()
        self.camera_matrices = self.load_camera_matrices()
        self.Ks, self.Rs, self.ts = self.decompose_camera_matrices(self.camera_matrices)
        self.Ps_cropped, self.inv_Ps_cropped = self.get_cropped_camera_matrices(self.Ks, self.Rs, self.ts, self.cropzone)
        self.all_cropped_Ps, self.all_cropped_invPs = self.get_cropped_camera_matrices(self.Ks, self.Rs, self.ts, self.cropzone)
        self.num_joints = 18
        self.num_wings_points = self.num_joints - 2
        self.num_points_per_wing = self.num_wings_points // 2
        self.left_inds = np.arange(0, self.num_points_per_wing)
        self.right_inds = np.arange(self.num_points_per_wing, self.num_wings_points)
        self.left_mask_ind, self.right_mask_ind = 3, 4
        pass



    def run_predict_2D(self):
        """
        creates an array of pose estimation predictions
        """
        t0 = time()
        # self.run_path = self.create_run_folders()
        self.clean_images()
        self.align_time_channels()
        self.preprocess_masks()
        # box = self.box_sparse.retrieve_dense_box()
        # Visualizer.display_movie_from_box(box)
        self.predicted_points = self.predict_method()
        self.preds_2D = self.predicted_points[..., :-1]

        self.conf_preds = self.predicted_points[..., -1]
        # self.enforce_3D_consistency()
        # self.points_3D_all, self.reprojection_errors, self.triangulation_errors = (
        #     self.get_all_3D_pnts_pairs(self.preds_2D, self.cropzone))

        box = self.box_sparse.retrieve_dense_box()
        Visualizer.show_predictions_all_cams(box, self.preds_2D)
        pass


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

    def enforce_3D_consistency(self):
        chosen_camera = 0
        cameras_to_check = np.arange(0, 4)
        cameras_to_check = cameras_to_check[np.where(cameras_to_check != chosen_camera)]
        for frame in range(self.num_frames):
            # step 1
            if frame > 0:
                switch_flag = self.deside_if_switch(chosen_camera, frame)
                if switch_flag:
                    self.flip_camera(chosen_camera, frame)

            # step 2
            cameras_to_flip = self.find_which_cameras_to_flip(cameras_to_check, frame)
            # print(f"frame {frame}, camera to flip {cameras_to_flip}")
            for cam in cameras_to_flip:
                self.flip_camera(cam, frame)

    def find_which_cameras_to_flip(self, cameras_to_check, frame):
        num_of_options = len(WHICH_TO_FLIP)
        switch_scores = np.zeros(num_of_options, )
        cropzone = self.cropzone[frame][np.newaxis, ...]
        for i, option in enumerate(WHICH_TO_FLIP):
            points_2D = np.copy(self.preds_2D[frame])
            cameras_to_flip = cameras_to_check[option]
            for cam in cameras_to_flip:
                left_points = points_2D[cam, self.left_inds, :]
                right_points = points_2D[cam, self.right_inds, :]
                points_2D[cam, self.left_inds, :] = right_points
                points_2D[cam, self.right_inds, :] = left_points
            points_2D = points_2D[np.newaxis, ...]
            _, reprojection_errors, _ = self.get_all_3D_pnts_pairs(points_2D, cropzone)
            score = np.mean(reprojection_errors)
            switch_scores[i] = score
        cameras_to_flip = cameras_to_check[WHICH_TO_FLIP[np.argmin(switch_scores)]]
        return cameras_to_flip



    def deside_if_switch(self, chosen_camera, frame):
        cur_left_points = self.preds_2D[frame, chosen_camera, self.left_inds, :]
        cur_right_points = self.preds_2D[frame, chosen_camera, self.right_inds, :]
        prev_left_points = self.preds_2D[frame - 1, chosen_camera, self.left_inds, :]
        prev_right_points = self.preds_2D[frame - 1, chosen_camera, self.right_inds, :]
        l2l_dist = np.linalg.norm(cur_left_points - prev_left_points)
        r2r_dist = np.linalg.norm(cur_right_points - prev_right_points)
        r2l_dist = np.linalg.norm(cur_right_points - prev_left_points)
        l2r_dist = np.linalg.norm(cur_left_points - prev_right_points)
        do_switch = l2l_dist + r2r_dist > r2l_dist + l2r_dist
        return do_switch

    def flip_camera(self, camera_to_flip, frame):
        left_points = self.preds_2D[frame, camera_to_flip, self.left_inds, :]
        right_points = self.preds_2D[frame, camera_to_flip, self.right_inds, :]
        self.preds_2D[frame, camera_to_flip, self.left_inds, :] = right_points
        self.preds_2D[frame, camera_to_flip, self.right_inds, :] = left_points
        # switch train_masks in box
        left_mask = self.box_sparse.get_frame_camera_channel_dense(frame, camera_to_flip, self.left_mask_ind)
        right_mask = self.box_sparse.get_frame_camera_channel_dense(frame, camera_to_flip, self.right_mask_ind)
        self.box_sparse.set_frame_camera_channel_dense(frame, camera_to_flip, self.left_mask_ind, right_mask)
        self.box_sparse.set_frame_camera_channel_dense(frame, camera_to_flip, self.right_mask_ind, left_mask)
        # switch confidence scores
        left_conf_scores = self.conf_preds[frame, camera_to_flip, self.left_inds]
        right_conf_scores = self.conf_preds[frame, camera_to_flip, self.right_inds]
        self.conf_preds[frame, camera_to_flip, self.left_inds] = left_conf_scores
        self.conf_preds[frame, camera_to_flip, self.right_inds] = right_conf_scores

    def load_camera_matrices(self):
        camera_matrices = h5py.File(self.calibration_data_path, "r")["/camera_matrices"][:].T
        return camera_matrices

    @staticmethod
    def get_cropped_camera_matrices(Ks, Rs, ts, cropzones):

        num_frames, num_cams = cropzones.shape[0], cropzones.shape[1]
        all_cropped_Ps = []
        all_cropped_invPs = []
        for frame in range(num_frames):
            cropped_Ps = []
            cropped_invPs = []
            for cam in range(num_cams):
                y_crop, x_crop = cropzones[frame, cam, :]
                K = Ks[cam]
                K /= K[-1, -1]
                dx = x_crop
                dy = 800 + 1 - y_crop - 192
                K_prime = K.copy()
                K_prime[0, 2] -= dx  # adjust x-coordinate of the principal point
                K_prime[1, 2] -= dy  # adjust y-coordinate of the principal point
                R = Rs[cam]
                t = ts[cam]
                P_prime = K_prime @ np.column_stack((R, t))
                P_prime /= np.linalg.norm(P_prime)
                cropped_Ps.append(P_prime)
                inv_P_prime = np.linalg.pinv(P_prime)
                inv_P_prime /= np.linalg.norm(inv_P_prime)
                cropped_invPs.append(inv_P_prime)
            cropped_Ps, cropped_invPs = np.array(cropped_Ps), np.array(cropped_invPs)
            all_cropped_Ps.append(cropped_Ps)
            all_cropped_invPs.append(cropped_invPs)
        all_cropped_Ps, all_cropped_invPs = np.array(all_cropped_Ps), np.array(all_cropped_invPs)
        return all_cropped_Ps, all_cropped_invPs

    @staticmethod
    def decompose_camera_matrices(camera_matrices):
        Ks = []
        Rs = []
        ts = []
        for P in camera_matrices:
            K, Rc_w, Pc, pp, pv = DecomposeCamera(P)
            t = (-Rc_w @ Pc)[:, np.newaxis]
            Ks.append(K)
            Rs.append(Rc_w)
            ts.append(t)
        return Ks, Rs, ts

    def choose_predict_method(self):
        if self.model_type == ALL_CAMS_PER_WING:
            return self.predict_4_cameras
        return self.predict_per_wing_per_cam

    def predict_per_wing_per_cam(self):
        all_pnts = self.predict_wings()
        tail_points = all_pnts[:, :, [8, 18], :]
        tail_points = np.expand_dims(np.mean(tail_points, axis=2), axis=2)
        head_points = all_pnts[:, :, [9, 19], :]
        head_points = np.expand_dims(np.mean(head_points, axis=2), axis=2)
        wings_points = all_pnts[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17], :]
        wings_and_body_pnts = np.concatenate((wings_points, tail_points, head_points), axis=2)
        return wings_and_body_pnts

    def predict_wings(self, n=100):
        Ypks = []
        for cam in range(self.num_cams):
            print(f"predict camera {cam + 1}")
            Ypks_per_wing = []
            for wing in range(2):
                input = self.box_sparse.get_camera_dense(cam, channels=[0, 1, 2, self.num_times_channels + wing])
                Ypk = self.predict_Ypk(input)
                Ypks_per_wing.append(Ypk)
            Ypk_cam = np.concatenate((Ypks_per_wing[0], Ypks_per_wing[1]), axis=1)
            Ypk_cam = np.expand_dims(Ypk_cam, axis=1)
            Ypks.append(Ypk_cam)
        Ypk_all = np.concatenate(Ypks, axis=1)
        return Ypk_all

    def predict_Ypk(self, input_to_net):
        input_to_net = torch.from_numpy(input_to_net).float()
        input_to_net = torch.permute(input_to_net, [0, 3, 2, 1])
        input_to_net = input_to_net.to(self.device)
        n_samples = input_to_net.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        ypk = []
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                # Extract the current batch from your dataset
                input_batch = input_to_net[start_idx:end_idx]
                # Feed the batch to your model
                outputs = self.wings_pose_estimation_model(input_batch)
                outputs = outputs.detach().cpu().numpy()
                outputs = np.transpose(outputs, [0, 2, 3, 1])
                ypk_batch = torch_find_peaks_argmax(outputs).clone()
                ypk_batch[..., [0, 1]] = ypk_batch[..., [1, 0]]
                ypk.append(ypk_batch)
        ypk = np.concatenate(ypk, axis=0)
        return ypk



    def preprocess_masks(self):
        self.add_masks()
        self.adjust_masks_size()
        if self.is_video:
            self.fix_masks()

    def add_masks(self, n=5):
        """ Add train_masks to the dataset using yolov8 segmentation model """
        for cam in range(self.num_cams):
            print(f"finds wings for camera number {cam+1}")
            img_3_ch_all = self.box_sparse.get_camera_dense(cam, [0, 1, 2])
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
                boxes = result.boxes.cpu().data.numpy()
                inds_to_keep = self.eliminate_close_vectors(boxes, 10)
                num_wings_found = np.count_nonzero(inds_to_keep)
                if num_wings_found > 0:
                    masks_found = result.masks.cpu().data.numpy()[inds_to_keep, :, :]
                else:
                    assert f"no masks found for this frame {frame} and camera {cam}"
                for wing in range(min(num_wings_found, 2)):
                    mask = masks_found[wing, :, :]
                    masks_2[:, :, wing] = mask
                masks[frame, :, :, :] = masks_2
            self.box_sparse.set_camera_dense(camera_idx=cam, dense_camera_data=masks, channels=[3, 4])

    def adjust_masks_size(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                mask_1 = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_times_channels)
                mask_2 = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_times_channels + 1)
                mask_1 = self.adjust_mask(mask_1, self.mask_increase_initial)
                mask_2 = self.adjust_mask(mask_2, self.mask_increase_initial)
                self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels, mask_1)
                self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + 1, mask_2)


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
                    # mask = self.box[frame, cam, :, :, self.num_times_channels + mask_num]
                    mask = self.box_sparse.get_frame_camera_channel_dense(frame, cam, self.num_times_channels + mask_num)
                    if np.all(mask == 0):  # check if all 0:
                        problematic_masks.append((frame, cam, mask_num))
                        # find previous matching mask
                        prev_mask = np.zeros(mask.shape)
                        next_mask = np.zeros(mask.shape)
                        for prev_frame in range(frame - 1, max(0, frame - search_range - 1), -1):
                            prev_mask_i = self.box_sparse.get_frame_camera_channel_dense(prev_frame, cam, self.num_times_channels + mask_num)
                            if not np.all(prev_mask_i == 0):  # there is a good mask
                                prev_mask = prev_mask_i
                                break
                        # find next matching mask
                        for next_frame in range(frame + 1, min(self.num_frames, frame + search_range)):
                            next_mask_i = self.box_sparse.get_frame_camera_channel_dense(next_frame, cam, self.num_times_channels + mask_num)
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
                        self.box_sparse.set_frame_camera_channel_dense(frame, cam, self.num_times_channels + mask_num, new_mask)

    @staticmethod
    def adjust_mask(mask, radius=3):
        mask = binary_closing(mask).astype(int)
        mask = binary_dilation(mask, iterations=radius).astype(int)
        return mask

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

    def clean_images(self):
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                for channel in range(self.num_times_channels):
                    image = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel)
                    # image = self.box[frame, cam, :, :, channel]
                    binary = np.where(image >= 0.1, 1, 0)
                    label = measure.label(binary)
                    props = measure.regionprops(label)
                    sizes = [prop.area for prop in props]
                    largest = np.argmax(sizes)
                    fly_component = np.where(label == largest + 1, 1, 0)
                    image = image * fly_component
                    # self.box[frame, cam, :, :, channel] = image
                    self.box_sparse.set_frame_camera_channel_dense(frame, cam, channel, image)

    def align_time_channels(self):
        all_shifts = np.zeros((self.num_frames, self.num_cams, 2, 2))
        all_shifts_smoothed = np.zeros((self.num_frames, self.num_cams, 2, 2))
        for frame in range(self.num_frames):
            for cam in range(self.num_cams):
                # present = self.box[frame, cam, :, :, 1]
                present = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=1)
                cm_present = self.get_fly_cm(present)
                for i, time_channel in enumerate([0, 2]):
                    # fly = self.box[frame, cam, :, :, time_channel]
                    fly = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=time_channel)
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
                    # fly = self.box[frame, cam, :, :, time_channel]
                    fly = self.box_sparse.get_frame_camera_channel_dense(frame, cam, channel_idx=time_channel)
                    shift_to_do = all_shifts_smoothed[frame, cam, i, :]
                    shifted_fly = shift(fly, shift_to_do, order=2)
                    # self.box[frame, cam, :, :, time_channel] = shifted_fly
                    self.box_sparse.set_frame_camera_channel_dense(frame, cam, time_channel, shifted_fly)

        # box = self.box_sparse.retrieve_dense_box()
        # Visualizer.display_movie_from_box(box[..., :-2])
    @staticmethod
    def get_fly_cm(im_orig):
        im = gaussian_filter(im_orig, sigma=2)
        im[im < 0.8] = 0
        # im = binary_opening(im, iterations=1)
        CM = center_of_mass(im)
        return np.array(CM)

    def load_model(self):
        model = torch.jit.load(self.wings_pose_estimation_model_path, map_location=self.device)
        model.eval()
        return model

    def get_run_name(self):
        box_path_file = os.path.basename(self.box_path)
        name, ext = os.path.splitext(box_path_file)
        run_name = f"{name}_{self.model_type}_{date.today().strftime('%b %d')}"
        return run_name

    def get_wings_detection_model(self):
        """ load a pretrained YOLOv8 segmentation model"""
        model = YOLO(self.wings_detection_model_path)
        model.fuse()
        return model

    def get_cropzone(self):
        return h5py.File(self.box_path, "r")["/cropzone"]

    def predict_wings_and_body_same_model(self):
        all_pnts = self.predict_wings()
        tail_points = all_pnts[:, :, [8, 18], :]
        tail_points = np.expand_dims(np.mean(tail_points, axis=2), axis=2)
        head_points = all_pnts[:, :, [9, 19], :]
        head_points = np.expand_dims(np.mean(head_points, axis=2), axis=2)
        wings_points = all_pnts[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17], :]
        wings_and_body_pnts = np.concatenate((wings_points, tail_points, head_points), axis=2)
        return wings_and_body_pnts


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

    def get_all_3D_pnts_pairs(self, points_2D, cropzone):
        points_3D_all, reprojection_errors, triangulation_errors = \
            self.triangulate.triangulate_2D_to_3D_reprojection_optimization(points_2D, cropzone)
        return points_3D_all, reprojection_errors, triangulation_errors


if __name__ == "__main__":
    config_path = r"/cs/labs/tsevi/lior.kotlar/amitai-s-thesis/from 2D to 3D/predict_2D_pytorch/predict_2D__pytorch_config.json"
    predictor = Predictor2D(config_path)
    predictor.run_predict_2D()

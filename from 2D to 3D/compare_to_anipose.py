import os.path

import matplotlib.pyplot as plt
import numpy as np
from traingulate import Triangulate
import h5py
from predict_2D_sparse_box import Predictor2D
import scipy
from scipy import signal
import json
from visualize import Visualizer
import cv2
from matplotlib.widgets import Slider


reprojected_points_2D_path = (r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled "
                              r"dataset\estimated_positions.mat")
ground_truth_2D_path = (r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled "
                        r"dataset\ground_truth_labels.mat")
configuration_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\2D_to_3D_config.json"
h5 = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\trainset_movie_1_370_520_ds_3tc_7tj.h5"
base_anipose_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\deep-lab-cut"
backbones = ['resnet_50', 'resnet_101', 'hrnet_w48']
RELEVANT_FEATURE_POINTS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17]


class CompareAnipose:
    def __init__(self, compare_to='resnet_50', filter_2D_first=True):
        self.ground_truth_2D = CompareAnipose.load_from_mat(ground_truth_2D_path, name='ground_truth')[:, :, RELEVANT_FEATURE_POINTS]
        self.num_frames = len(self.ground_truth_2D)
        self.num_joints = self.ground_truth_2D.shape[2]
        self.cropzone = h5py.File(h5, 'r')['/cropzone'][:self.num_frames]
        with open(configuration_path) as C:
            config = json.load(C)
            self.triangulate = Triangulate(config)

        self.our_method_2D = CompareAnipose.load_from_mat(reprojected_points_2D_path, name='positions',
                                                          end=self.num_frames)[:, :, RELEVANT_FEATURE_POINTS]
        self.our_method_3D = self.get_3D_points(self.our_method_2D, self.cropzone)

        self.ground_truth_3D = self.get_3D_points(self.ground_truth_2D, self.cropzone)
        self.ground_truth_3D_smoothed = Predictor2D.smooth_3D_points(self.ground_truth_3D)
        self.reprojected_smoothed_ground_truth = self.triangulate.get_reprojections(self.ground_truth_3D_smoothed, self.cropzone)

        # for filter_size in range(1, 20, 2):
        filter_size = 11  # found to be the optimal
        # load and filter the anipose points
        self.anipose_raw_2D = np.load(os.path.join(base_anipose_path, f"{compare_to}.npy"))
        self.anipose_raw_2D_filtered = CompareAnipose.filter_2d_detections(self.anipose_raw_2D,
                                                                           medfilt_size=filter_size)

        # get_all_options
        anipose_3D_raw_after_2D_raw = self.triangulate_using_anipose_method(self.anipose_raw_2D)
        anipose_3D_raw_after_2D_filter = self.triangulate_using_anipose_method(self.anipose_raw_2D_filtered)
        anipose_3D_filter_after_2D_raw = self.filter_3d_detections(self.triangulate_using_anipose_method(self.anipose_raw_2D), medfilt_size=filter_size)
        anipose_3D_filter_after_2D_filter = self.filter_3d_detections(self.triangulate_using_anipose_method(self.anipose_raw_2D_filtered), medfilt_size=filter_size)

        # get all errors
        self.anipose_3D_raw_after_2D_raw_error, _ = CompareAnipose.calculate_error(self.ground_truth_3D, anipose_3D_raw_after_2D_raw)
        self.anipose_3D_raw_after_2D_filter_error, _ = CompareAnipose.calculate_error(self.ground_truth_3D, anipose_3D_raw_after_2D_filter)
        self.anipose_3D_filter_after_2D_raw_error, std_anipose_error = CompareAnipose.calculate_error(self.ground_truth_3D, anipose_3D_filter_after_2D_raw)
        self.anipose_3D_filter_after_2D_filter_error, _ = CompareAnipose.calculate_error(self.ground_truth_3D, anipose_3D_filter_after_2D_filter)
        min_error = np.min([self.anipose_3D_raw_after_2D_raw_error, self.anipose_3D_raw_after_2D_filter_error,
                            self.anipose_3D_filter_after_2D_raw_error, self.anipose_3D_filter_after_2D_filter_error])
        print(f"Filter size: {filter_size}, min error: {min_error}")



        # my method 3D error
        self.our_method_3D_error, self.our_method_std = CompareAnipose.calculate_error(self.ground_truth_3D_smoothed, self.our_method_3D)
        self.our_error_std = np.std(np.linalg.norm(self.ground_truth_3D_smoothed - self.our_method_3D, axis=2))

        ratio_3D = min_error/self.our_method_3D_error
        ratio_std = std_anipose_error/self.our_error_std

        reprojected_anipose = self.triangulate.get_reprojections(anipose_3D_filter_after_2D_raw, self.cropzone)
        anipose_error_pixels = np.mean(np.linalg.norm(reprojected_anipose -
                                                      self.reprojected_smoothed_ground_truth, axis=3))
        our_method_error_pixels = np.mean(np.linalg.norm(self.our_method_2D -
                                                         self.reprojected_smoothed_ground_truth, axis=3))
        ratio_2D = anipose_error_pixels/our_method_error_pixels

        print(f"Anipose 3D error in [mm]: {1000 * self.anipose_3D_filter_after_2D_raw_error} +- {1000 * std_anipose_error}\n"
              f"Our method 3D error in [mm]: {1000 * self.our_method_3D_error} +- {1000 * self.our_method_std}\n"
              f"ratio 3D: {ratio_3D}\n"
              f"ratio std: {ratio_std}\n")
        # save
        np.save(f"anipose_3D.npy", anipose_3D_raw_after_2D_raw)
        np.save(f"anipose_2D.npy", reprojected_anipose)

        # plot
        # self.display_differences(reprojected_anipose, self.reprojected_smoothed_ground_truth, single_camera=-1)

        self.compare_predictions(reprojected_anipose, self.our_method_2D, self.reprojected_smoothed_ground_truth,
                                 camera=2, frame=2)
        pass

    @staticmethod
    def calculate_error(ground_truth, prediction):
        all_errors = np.linalg.norm(ground_truth - prediction, axis=2)
        mean_error = np.mean(all_errors)
        std = np.std(all_errors)
        return mean_error, std

    @staticmethod
    def visualize(points_3D):
        positions = [7, 15]
        # Create an array of NaNs with shape (N, number_of_insertions, 3)
        nan_insert = np.full((points_3D.shape[0], len(positions), points_3D.shape[2]), np.nan)
        # Sort positions to handle shifting indices correctly
        positions_sorted = sorted(positions)
        # Insert NaNs at the specified positions along axis=1
        arr_extended = np.insert(points_3D, positions_sorted, nan_insert, axis=1)
        Visualizer.show_points_in_3D(arr_extended)

    def triangulate_using_anipose_method(self, points_2D):
        points_3D_all, reprojection_errors, _ = self.triangulate.triangulate_2D_to_3D_reprojection_optimization(points_2D, self.cropzone)
        num_frames = len(points_3D_all)
        num_feature_points = points_3D_all.shape[1]
        min_indices = np.argmin(reprojection_errors, axis=2)  # Shape: (num_frames, num_feature_points)
        selected_candidates = points_3D_all[np.arange(num_frames)[:, None], np.arange(num_feature_points), min_indices]
        return selected_candidates

    def get_3D_points(self, points_2D, cropzone):
        points_3D_all_multiviews, _ = self.triangulate.triangulate_points_all_possible_views(points_2D, cropzone)
        points_3D_4_cams = points_3D_all_multiviews[:, :, -1, :]  # take the 4 views triangulation
        return points_3D_4_cams

    @staticmethod
    def load_from_mat(path, name, end=100):
        mat = scipy.io.loadmat(path)
        numpy_array = mat[name]
        numpy_array = np.transpose(numpy_array, (3, 2, 0, 1))
        return numpy_array[:end]

    @staticmethod
    def medfilt_data(values, size=5):
        padsize = size + 5
        vpad = np.pad(values, (padsize, padsize), mode='median', stat_length=5)
        vpadf = signal.medfilt(vpad, kernel_size=size)
        return vpadf[padsize:-padsize]

    @staticmethod
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    @staticmethod
    def interpolate_data(vals):
        nans, ix = CompareAnipose.nan_helper(vals)
        out = np.copy(vals)
        if np.mean(nans) > 0.85:
            return out
        out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
        return out

    @staticmethod
    def filter_2d_detections(detections, medfilt_size=5):
        """
        Filter 2D detections using median filtering and interpolation.

        Parameters:
            detections (np.array): Array of 2D detections of shape
                                   (number of frames, number of cameras, number of feature points, 2).
            medfilt_size (int): Size of the median filter kernel. Default is 15.

        Returns:
            np.array: Filtered 2D detections.
        """
        num_frames, num_cameras, num_points, _ = detections.shape
        filtered_detections = np.copy(detections)

        for cam_idx in range(num_cameras):
            for point_idx in range(num_points):
                for coord in range(2):  # x and y coordinates
                    values = detections[:, cam_idx, point_idx, coord]
                    values = CompareAnipose.interpolate_data(values)
                    values = CompareAnipose.medfilt_data(values, size=medfilt_size)
                    filtered_detections[:, cam_idx, point_idx, coord] = values

        return filtered_detections

    @staticmethod
    def filter_3d_detections(detections, medfilt_size=5):
        """
        Filter 3D detections using median filtering and interpolation.

        Parameters:
            detections (np.array): Array of 3D detections of shape
                                   (number of frames, number of feature points, 3).
            medfilt_size (int): Size of the median filter kernel. Default is 15.
            offset_threshold (float): Threshold for filtering out large offsets. Default is 100.

        Returns:
            np.array: Filtered 3D detections.
        """
        num_frames, num_points, _ = detections.shape
        filtered_detections = np.copy(detections)

        for point_idx in range(num_points):
            for coord in range(3):  # x, y, z coordinates
                values = detections[:, point_idx, coord]
                values = CompareAnipose.interpolate_data(values)
                values = CompareAnipose.medfilt_data(values, size=medfilt_size)
                filtered_detections[:, point_idx, coord] = values

        return filtered_detections

    @staticmethod
    def display_differences(predictions_2D, ground_truth, single_camera=-1):
        """
        Display differences between predictions and ground truth.
        Args:
            predictions_2D: Predictions array
            ground_truth: Ground truth array
            single_camera: If -1, shows all cameras. If 1-4, shows only that camera in full screen.
        """

        def get_distinct_colors(n):
            """Generate n distinct colors"""
            colors = plt.cm.rainbow(np.linspace(0, 1, n))
            return colors

        base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\deep-lab-cut"
        video_paths = ["labeled_cam_1.mp4", "labeled_cam_2.mp4", "labeled_cam_3.mp4", "labeled_cam_4.mp4"]

        # Validate single_camera parameter
        if single_camera != -1 and not (1 <= single_camera <= 4):
            raise ValueError("single_camera must be -1 for all cameras, or 1-4 for a specific camera")

        # Open all videos (or just the selected one)
        if single_camera != -1:
            caps = [cv2.VideoCapture(os.path.join(base_path, video_paths[single_camera - 1]))]
        else:
            caps = [cv2.VideoCapture(os.path.join(base_path, path)) for path in video_paths]

        # Get video properties
        num_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

        # Create figure and axes based on display mode
        if single_camera != -1:
            fig = plt.figure(figsize=(15, 13))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.1])
            axs = [fig.add_subplot(gs[0])]
            slider_ax = fig.add_subplot(gs[1])
        else:
            fig = plt.figure(figsize=(15, 17))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.1])
            axs = [fig.add_subplot(gs[i]) for i in range(4)]
            slider_ax = fig.add_subplot(gs[4:])

        # Get number of points per frame
        num_points = predictions_2D.shape[2]  # (frames, cameras, points, 2)
        colors = get_distinct_colors(num_points)

        # Create slider
        slider = Slider(slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
        current_frame = 0

        def update_frame(frame_idx):
            # Set frame position for all videos
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Read frames from videos
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Clear and update all plots
            for i, (frame, ax) in enumerate(zip(frames, axs)):
                ax.clear()
                ax.imshow(frame)

                # Calculate camera index based on display mode
                cam_idx = (single_camera - 1) if single_camera != -1 else i

                # Plot each point with its own color
                for j in range(num_points):
                    # Get points
                    dlc_point = predictions_2D[frame_idx][cam_idx][j]
                    gt_point = ground_truth[frame_idx][cam_idx][j]

                    # Draw yellow line connecting corresponding points
                    ax.plot([dlc_point[0], gt_point[0]],
                            [dlc_point[1], gt_point[1]],
                            color='yellow',
                            linestyle='-',
                            linewidth=1,
                            alpha=0.7)

                    # DLC predictions as 'o'
                    ax.scatter(dlc_point[0], dlc_point[1],
                               c=[colors[j]], marker='o', s=100,
                               # label=f'DLC Point {j + 1}' if i == 0 else None,
                               )

                    # Ground truth as 'x'
                    ax.scatter(gt_point[0], gt_point[1],
                               c=[colors[j]], marker='x', s=100,
                               # label=f'GT Point {j + 1}' if i == 0 else None,
                               )

                title = f'Camera {cam_idx + 1}' if single_camera == -1 else f'Camera {single_camera} (Full Screen)'
                ax.set_title(title)
                ax.axis('off')

                # Add legend only for the first subplot or single camera view
                if i == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            fig.canvas.draw_idle()

        # Callback for slider changes
        def on_slider_change(val):
            nonlocal current_frame
            current_frame = int(val)
            update_frame(current_frame)

        slider.on_changed(on_slider_change)

        # Keyboard controls
        def on_key_press(event):
            nonlocal current_frame
            if event.key == 'left':
                current_frame = max(0, current_frame - 1)
            elif event.key == 'right':
                current_frame = min(num_frames - 1, current_frame + 1)
            slider.set_val(current_frame)

        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Initial frame display
        update_frame(0)
        plt.tight_layout()
        plt.show()

        # Clean up
        for cap in caps:
            cap.release()

    @staticmethod
    def compare_predictions(predictions_2D_1, predictions_2D_2, ground_truth, camera, frame):
        """
        Compare two sets of predictions with ground truth for a specific camera and frame.

        Args:
            predictions_2D_1: First set of predictions array
            predictions_2D_2: Second set of predictions array
            ground_truth: Ground truth array
            camera: Camera number (1-4)
            frame: Frame number to display
        """

        def get_distinct_colors(n):
            """Generate n distinct colors"""
            colors = plt.cm.rainbow(np.linspace(0, 1, n))
            return colors

        # Validate camera input
        if not (1 <= camera <= 4):
            raise ValueError("camera must be between 1-4")

        # Convert camera number to index
        camera_idx = camera - 1

        # Get video frame
        base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\deep-lab-cut"
        video_path = f"labeled_cam_{camera}.mp4"
        cap = cv2.VideoCapture(os.path.join(base_path, video_path))

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame_img = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame} from camera {camera}")

        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        cap.release()

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Get number of points
        num_points = predictions_2D_1.shape[2]  # (frames, cameras, points, 2)
        colors = get_distinct_colors(num_points)

        # Plot first comparison (predictions_2D_1 vs ground truth)
        ax1.imshow(frame_img)
        ax1.set_title(f'Model 1 vs Ground Truth - Camera {camera}', color='white')

        for j in range(num_points):
            # Get points
            pred1_point = predictions_2D_1[frame][camera_idx][j]
            gt_point = ground_truth[frame][camera_idx][j]

            # Draw yellow line connecting corresponding points
            ax1.plot([pred1_point[0], gt_point[0]],
                     [pred1_point[1], gt_point[1]],
                     color='yellow',
                     linestyle='-',
                     linewidth=1,
                     alpha=0.7)

            # Predictions as 'o'
            ax1.scatter(pred1_point[0], pred1_point[1],
                        c=[colors[j]], marker='o', s=100)

            # Ground truth as 'x'
            ax1.scatter(gt_point[0], gt_point[1],
                        c=[colors[j]], marker='x', s=100)

        # Add left-aligned text annotations in top right
        ax1.text(0.75, 0.98, 'o → Anipose detections\nX → ground truth',
                 color='white',
                 transform=ax1.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=10)

        # Get image dimensions
        height, width = frame_img.shape[:2]
        center_x = width / 2
        center_y = height / 2
        zoom_factor = 1.8

        # Calculate zoom boundaries
        x_range = width / (2 * zoom_factor)
        y_range = height / (2 * zoom_factor)

        # Set zoom limits around center
        ax1.set_xlim(center_x - x_range, center_x + x_range)
        ax1.set_ylim(center_y + y_range, center_y - y_range)  # Inverted for image coordinates

        ax1.axis('off')
        # Remove legend calls completely

        # Plot second comparison (predictions_2D_2 vs ground truth)
        ax2.imshow(frame_img)
        ax2.set_title(f'Model 2 vs Ground Truth - Camera {camera}', color='white')

        for j in range(num_points):
            # Get points
            pred2_point = predictions_2D_2[frame][camera_idx][j]
            gt_point = ground_truth[frame][camera_idx][j]

            # Draw yellow line connecting corresponding points
            ax2.plot([pred2_point[0], gt_point[0]],
                     [pred2_point[1], gt_point[1]],
                     color='yellow',
                     linestyle='-',
                     linewidth=1,
                     alpha=0.7)

            # Predictions as 'o'
            ax2.scatter(pred2_point[0], pred2_point[1],
                        c=[colors[j]], marker='o', s=100)

            # Ground truth as 'x'
            ax2.scatter(gt_point[0], gt_point[1],
                        c=[colors[j]], marker='x', s=100)

        # Add left-aligned text annotations in top right
        ax2.text(0.75, 0.98, 'o → Our detections\nX → ground truth',
                 color='white',
                 transform=ax2.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=10)

        # Set the same zoom limits for consistency
        ax2.set_xlim(center_x - x_range, center_x + x_range)
        ax2.set_ylim(center_y + y_range, center_y - y_range)  # Inverted for image coordinates

        ax2.axis('off')
        # Remove legend calls completely

        plt.tight_layout()
        plt.show()
        return fig

if __name__ == '__main__':
    CompareAnipose()

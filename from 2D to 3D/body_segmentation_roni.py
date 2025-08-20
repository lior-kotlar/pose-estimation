import h5py
import numpy as np
import scipy.ndimage
import json
import skimage
from skimage import transform
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import center_of_mass, shift, binary_closing, binary_opening, binary_dilation
from scipy.interpolate import make_smoothing_spline
from skimage.morphology import convex_hull_image
import matplotlib
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
from visualize import Visualizer
from traingulate import Triangulate
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
SPAN = 36
INTERVAL = 1
CROP_SIZE = 192
ORIG_IMAGE_SHAPE = (800, 1280)
FIRST_GRID_SIZE = 0.005
FIRST_GRID_SPACING = 0.0001
SECOND_GRID_SPACING = 0.00005
SEGMENTATION_THRESHOLD = 1.5  # up is a lower threshold


class BodySegmentation:
    def __init__(self, box_path, config_path=""):
        self.box_path = box_path
        self.box = self.get_box()[:200]
        self.present = self.box[..., 1]
        self.cropzone = self.get_cropzone()
        self.num_frames = self.box.shape[0]
        self.num_cams = self.box.shape[1]
        self.image_size = self.present.shape[-1]
        with open(config_path) as C:
            config = json.load(C)
            self.triangulate = Triangulate(config)
        self.CMs_2D, self.body_segmentations = self.get_CMs_body_segmentations()
        self.x_body, self.CMs_3D, self.head_tail_points = self.get_x_body()

        # directory = os.path.dirname(box_path)
        # new_file_name = os.path.join(directory, "body_segmentations_.h5")
        # with h5py.File(new_file_name, "w") as f:
        #     ds_pos = f.create_dataset("CMs_3D", data=self.CMs_3D, compression="gzip",
        #                               compression_opts=1)
        #     ds_pos = f.create_dataset("x_body", data=self.x_body, compression="gzip",
        #                               compression_opts=1)
        #     ds_pos = f.create_dataset("head_tail_points", data=self.head_tail_points, compression="gzip",
        #                               compression_opts=1)
        #     ds_pos = f.create_dataset("body_segmentations", data=self.body_segmentations, compression="gzip",
        #                               compression_opts=1)

    def triangulate_points(self, points_2D, cropzone):
        return self.triangulate.triangulate_2D_to_3D_reprojection_optimization(points_2D, cropzone)[0]

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
        return h5py.File(self.box_path, "r")["/cropzone"][:]

    def get_CMs_body_segmentations(self):
        body_segmentations = np.zeros_like(self.present)
        CMs_estimate = np.zeros((self.num_frames, self.num_cams, 2))
        CMs = np.zeros_like(CMs_estimate)
        print("get coarse center of mass")
        for frame in tqdm(range(SPAN, self.num_frames - SPAN)):
            for cam in range(self.num_cams):
                original_image = np.zeros(ORIG_IMAGE_SHAPE)
                for frame_span in range(frame - SPAN, frame + SPAN):
                    crop_y = self.cropzone[frame_span, cam, 0] - 1
                    crop_x = self.cropzone[frame_span, cam, 1] - 1
                    fly = self.present[frame_span, cam, :, :]
                    original_image[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE] += fly
                original_image = np.where(original_image > 25, original_image, 0)
                CMs_estimate[frame, cam, :] = center_of_mass(original_image)

        for cam in range(self.num_cams):
            for axis in range(CMs.shape[-1]):
                vals = CMs_estimate[:, cam, axis]
                vals = vals[SPAN:-SPAN]
                vals_smoothed = np.zeros_like(vals)
                # window_size = 100
                # for i in range(len(vals)):
                #     # Define the start and end of the window
                #     start = max(0, i - window_size)
                #     end = min(len(vals), i + window_size)
                #
                #     # Get the window of data around the current point
                #     window = vals[start:end]
                #
                #     # Fit a 2nd degree polynomial to the window of data
                #     poly_coeffs = np.polyfit(np.arange(start, end), window, 2)
                #
                #     # Evaluate the fitted polynomial at the current point
                #     vals_smoothed[i] = np.polyval(poly_coeffs, i)

                A = np.arange(len(vals))
                spline = make_smoothing_spline(A, vals, lam=100000)
                vals_smoothed = spline(A)
                # plt.plot(vals_smoothed)
                # plt.plot(vals)
                # plt.show()
                CMs_estimate[SPAN:-SPAN, cam, axis] = vals_smoothed
        print("\nget accurate center of mass")
        for frame in tqdm(range(2*SPAN, self.num_frames - 2*SPAN)):
            for cam in range(self.num_cams):
                original_image = np.zeros(ORIG_IMAGE_SHAPE)
                cm_present = CMs_estimate[frame, cam, :]
                crop_y = self.cropzone[frame - SPAN:frame + SPAN, cam, 0] - 1
                crop_x = self.cropzone[frame - SPAN:frame + SPAN, cam, 1] - 1
                cm_span = CMs_estimate[frame - SPAN:frame + SPAN, cam, :]
                flies = self.present[frame - SPAN:frame + SPAN, cam, :, :]
                shift_to_do = cm_present - cm_span
                for i in range(flies.shape[0]):
                    fly = flies[i]
                    nz = np.nonzero(fly)
                    vals = fly[nz]
                    inds = np.column_stack((nz[0], nz[1])).astype(float)
                    # add the crop
                    inds[:, 0] += crop_y[i]
                    inds[:, 1] += crop_x[i]
                    # shift
                    inds[:, 0] += shift_to_do[i, 0]
                    inds[:, 1] += shift_to_do[i, 1]

                    (y, x) = np.round(inds)[:, 0].astype(int), np.round(inds)[:, 1].astype(int)
                    original_image[y, x] += vals

                crop_y = self.cropzone[frame, cam, 0] - 1
                crop_x = self.cropzone[frame, cam, 1] - 1
                num_channels = flies.shape[0]
                body_segmentation = np.where(original_image > num_channels / SEGMENTATION_THRESHOLD, original_image, 0)
                body_segmentation_cropped = body_segmentation[crop_y: crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE]
                CMs[frame, cam, :] = scipy.ndimage.center_of_mass(body_segmentation_cropped)
                body_segmentation_cropped = binary_opening(body_segmentation_cropped.astype(bool), iterations=1)
                body_segmentation_cropped = binary_dilation(body_segmentation_cropped, iterations=1)
                body_segmentation_cropped = np.bitwise_and(body_segmentation_cropped,
                                                           self.present[frame, cam, ...].astype(bool))
                body_segmentations[frame, cam, :, :] = body_segmentation_cropped
        box = np.concatenate((self.present[..., np.newaxis], self.present[..., np.newaxis],
                                                                    body_segmentations[..., np.newaxis]), axis=-1)
        Visualizer.display_movie_from_box(box)
        return CMs, body_segmentations

    def get_x_body(self):
        print("\ngetting x body")
        head_tail_vec_roni = np.zeros((self.num_frames, 3))
        head_tail_points = np.zeros((self.num_frames, 2, 3))
        CMs_3D = np.zeros((self.num_frames, 3))
        all_3D_CMs = np.squeeze(self.triangulate_points(self.CMs_2D[:, :, np.newaxis, :], self.cropzone))
        for frame in tqdm(range(2*SPAN, self.num_frames - 2*SPAN)):
            # print("frame ", frame)
            center_point = np.mean(all_3D_CMs[frame, :, :], axis=0)
            # create a grid:
            # Size of the grid (5 mm by 5 mm)
            grid_size_mm = FIRST_GRID_SIZE

            # Spacing between points (100 µm)
            spacing_um = FIRST_GRID_SPACING

            cube = self.create_grid(center_point, grid_size_mm, spacing_um)

            fly_points = self.segment_body_3D_points(cube, frame, 1)

            # Find the maximum and minimum values for each coordinate
            x_max, y_max, z_max = np.max(fly_points, axis=0)
            x_min, y_min, z_min = np.min(fly_points, axis=0)

            # Create a grid of 3D points within the specified limits
            spacing = SECOND_GRID_SPACING
            x_coords = np.arange(x_min, x_max + spacing, spacing)
            y_coords = np.arange(y_min, y_max + spacing, spacing)
            z_coords = np.arange(z_min, z_max + spacing, spacing)

            # Create a meshgrid from the coordinates
            X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords)
            cube = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            fly_points = self.segment_body_3D_points(cube, frame, 2)

            pca = PCA(n_components=3)
            pca.fit(fly_points)
            principal_components = pca.components_

            # Project points onto the first principal component after centering them around 0
            projected_points = np.dot(fly_points - pca.mean_, principal_components[0])

            # Determine the threshold values for selecting points
            threshold_low = np.percentile(projected_points, 10)
            threshold_high = np.percentile(projected_points, 90)

            # Select points from both ends
            head_points = fly_points[projected_points < threshold_low]
            tail_points = fly_points[projected_points > threshold_high]
            remaining_points = fly_points[(projected_points >= threshold_low) & (projected_points <= threshold_high)]

            head = np.mean(head_points, axis=0)
            tail = np.mean(tail_points, axis=0)

            head_tail_points[frame, 0, :] = tail
            head_tail_points[frame, 1, :] = head

            CM = pca.mean_
            CMs_3D[frame, :] = CM

            head_tail_vec = (head - tail)/np.linalg.norm(head - tail)
            head_tail_vec_roni[frame, :] = head_tail_vec
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # Plot the points
            # # ax.scatter(*fly_points.T, color='orange')
            #
            # ax.scatter(*remaining_points.T, color='orange')
            # ax.scatter(*head_points.T, color='blue')
            # ax.scatter(*tail_points.T, color='black')
            # plt.show()

            # np.save("fly_points.npy", fly_points)

            pass

        return head_tail_vec_roni, CMs_3D, head_tail_points

    def segment_body_3D_points(self, cube, frame, iter):
        cropzone = np.tile(self.cropzone[frame, ...], (cube.shape[0], 1, 1))
        reprojections = np.squeeze(self.triangulate.get_reprojections(cube[:, np.newaxis, :], cropzone))
        limit = np.array(self.body_segmentations.shape[-2:]) - 1
        are_inside = (reprojections >= 0) & (reprojections < limit)
        are_inside_all = np.all(are_inside, axis=(1, 2))
        reprojections = reprojections[are_inside_all]
        cube = cube[are_inside_all]
        is_inside = []
        for cam in range(self.num_cams):
            reprojections_cam = np.round(reprojections[:, cam, :]).astype(int)
            # reprojections_cam[~are_inside[:, cam, :]] = 0
            mask = self.body_segmentations[frame, cam, :, :]
            present = self.box[frame, cam, :, :, 1]
            inside = mask[reprojections_cam[:, 1], reprojections_cam[:, 0]].astype(bool)
            is_inside.append(inside)

            # outside = np.bitwise_not(inside)
            # outide_points = reprojections_cam[outside, :]
            # inside_points = reprojections_cam[inside, :]
            # plt.figure()
            # plt.imshow(present + mask, cmap='gray')  # Display the binary mask
            # plt.scatter(inside_points[:, 0], inside_points[:, 1], color='red')  # Display the points
            # # plt.scatter(outide_points[:, 0], outide_points[:, 1], color='blue')
            # plt.show()
            pass

        is_inside = np.vstack(is_inside).T
        is_inside = np.all(is_inside, axis=1)
        fly_points = cube[is_inside]

        # get accurate body masks
        # if iter == 2:
        #     cropzone = np.tile(self.cropzone[frame, ...], (fly_points.shape[0], 1, 1))
        #     fly_points_reprojected = np.squeeze(self.triangulate.get_reprojections(fly_points[:, np.newaxis, :], cropzone))
        #     for cam in range(self.num_cams):
        #         fly_reprejections_cam = np.round(fly_points_reprojected[:, cam, :]).astype(int)
        #         new_mask = np.zeros((self.body_segmentations.shape[-1], self.body_segmentations.shape[-1]))
        #         new_mask[fly_reprejections_cam[:, 1], fly_reprejections_cam[:, 0]] = 1
        #         plt.imshow(new_mask + self.present[frame, cam])
        #         plt.show()
        return fly_points

    @staticmethod
    def create_grid(center_point, grid_size_mm, spacing_um):
        # Calculate the number of points along each axis
        num_points_per_axis = int(grid_size_mm / spacing_um)
        # Create 1D arrays for each axis
        x_coords = np.linspace(center_point[0] - grid_size_mm / 2, center_point[0] + grid_size_mm / 2,
                               num_points_per_axis)
        y_coords = np.linspace(center_point[1] - grid_size_mm / 2, center_point[1] + grid_size_mm / 2,
                               num_points_per_axis)
        z_coords = np.linspace(center_point[2] - grid_size_mm / 2, center_point[2] + grid_size_mm / 2,
                               num_points_per_axis)
        # Create 3D meshgrid from the 1D arrays
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords)
        # Combine X, Y, Z coordinates to form the point cloud
        cube = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        return cube


if __name__ == '__main__':

    # h5 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov62_d\body_segmentations_.h5"
    # CMs_3D = h5py.File(h5, "r")["/CMs_3D"][:]
    # x_body = h5py.File(h5, "r")["/x_body"][:]
    # head_tail_points = h5py.File(h5, "r")["head_tail_points"][:]
    # pass
    #
    # pts_3D = np.load(r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov62_d\points_3D_ensemble.npy")
    # head_tail_points2 = pts_3D[:, [-2, -1], :]
    # cm = np.mean(pts_3D[:, [-2, -1], :], axis=1)
    #
    # plt.figure()
    # # plt.plot(CMs_3D[72:-72])
    # # plt.plot(cm[72:-72])
    #
    # plt.plot(head_tail_points[72:-72, 1])
    # plt.plot(head_tail_points2[72:-72, 1])
    # plt.show()

    config_path = "2D_to_3D_config.json"
    # movie 1
    box_path = (r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on "
                r"cluster\selected_movies\mov10_u\movie_10_130_1666_ds_3tc_7tj.h5")
    BS = BodySegmentation(box_path, config_path)
    # # # movie 2
    # box_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov11_u\movie_11_10_1143_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)
    #
    # # movie 3 a
    # box_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov61_d\mov61_d1\movie_61_10_2355_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)
    #
    # # movie 3 b
    # box_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov61_d\mov61_d2\movie_61_2356_2342_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)

    # movie 4
    # box_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies\mov62_d\movie_62_160_1888_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)


    # config_path = "2D_to_3D_config.json"
    # # movie 1
    # box_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/selected_movies/mov10_u/movie_10_130_1666_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)
    # # movie 2
    # box_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/selected_movies/mov11_u/movie_11_10_1143_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)
    #
    # # movie 3 a
    # box_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/selected_movies/mov61_d1/movie_61_10_2355_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)
    # # movie 3 b
    # box_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/selected_movies/mov61_d2/movie_61_2356_2342_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)
    #
    # # movie 4
    # box_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/selected_movies/mov62_d/movie_62_160_1888_ds_3tc_7tj.h5"
    # BS = BodySegmentation(box_path, config_path)



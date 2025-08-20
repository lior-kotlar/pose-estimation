import os
import numpy as np
import pandas as pd
from PIL import Image
import deeplabcut
from pathlib import Path
import scipy
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal
from aniposelib.cameras import Camera, CameraGroup
from anipose.filter_3d import medfilt_data, interpolate_data
RELEVANT_FEATURE_POINTS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17]
backbones = ['resnet_50', 'resnet_101', 'hrnet_w48',]


def train_dlc_model(
        frames,
        labels,
        project_name='Flies',
        experimenter='Amitai',
        working_directory=r'C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\deep-lab-cut\projects',
        video_path=r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\deep-lab-cut\output.mp4",
        num_bodyparts=16,
        cnn='resnet_50'
):
    """
    Train a DeepLabCut model with custom dataset

    Args:
        frames: numpy array of shape (num_frames, height, width, 3)
        labels: numpy array of shape (num_frames, num_keypoints, 2)
        project_name: name of the project
        experimenter: name of experimenter
        working_directory: directory to store project
        video_path: path to example video
        num_bodyparts: number of keypoints to track
    """
    # Ensure working directory exists
    working_directory = Path(working_directory)
    working_directory.mkdir(parents=True, exist_ok=True)

    # Step 1: Create Project
    date = '2025-01-14'
    project_path = working_directory / f'{project_name}-{experimenter}-{date}'
    config_path = project_path / 'config.yaml'

    if not config_path.exists():
        config_path = deeplabcut.create_new_project(
            project=project_name,
            experimenter=experimenter,
            videos=[str(video_path)],
            working_directory=str(working_directory),
            copy_videos=True
        )
        print(f"Created new project at {project_path}")
    else:
        print(f"Project already exists at {project_path}")

    # Step 2: Prepare Data
    frames = (frames * 255).astype(np.uint8) if frames.max() <= 1.0 else frames

    # Create directory for labeled data
    video_name = Path(video_path).stem
    labeled_data_dir = project_path / 'labeled-data' / video_name
    labeled_data_dir.mkdir(parents=True, exist_ok=True)

    # Save frames as images
    print("Saving frames as images...")
    image_paths = []
    for i, frame in enumerate(frames):
        image_path = labeled_data_dir / f'img{i:04d}.png'
        Image.fromarray(frame).save(image_path)
        image_paths.append(str(image_path.relative_to(project_path)))

    # Create labels DataFrame with proper multi-index structure
    print("Creating labels dataframe...")
    bodyparts = [f'bodypart_{i}' for i in range(num_bodyparts)]
    scorer = experimenter

    # Create proper column multi-index
    column_index = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ['x', 'y']],
        names=['scorer', 'bodyparts', 'coords']
    )

    # Create dataframe with flattened coordinates
    data = []
    for label in labels:
        flat_coords = label.reshape(-1)  # Flatten the coordinates
        data.append(flat_coords)
    df = pd.DataFrame(data, columns=column_index)

    # Set proper index for the frames
    df.index = [os.path.join('labeled-data', video_name, f'img{i:04d}.png') for i in range(len(frames))]
    df.index.name = 'filename'

    # Save labels
    labels_path = labeled_data_dir / f'CollectedData_{scorer}.h5'
    df.to_hdf(labels_path, key='df_with_missing', mode='w')
    df.to_csv(labeled_data_dir / f'CollectedData_{scorer}.csv')

    # Step 3: Update Configuration
    print("Updating configuration...")
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)

    cfg.update({
        'project_path': str(project_path),
        'video_sets': {str(video_path): {'crop': '0, 192, 0, 192'}},
        'bodyparts': bodyparts,
        'scorer': scorer,
        'skeleton': [
            ['bodypart_0', 'bodypart_1'],
            ['bodypart_1', 'bodypart_2'],
            ['bodypart_2', 'bodypart_3'],
            ['bodypart_3', 'bodypart_4'],
            ['bodypart_4', 'bodypart_5'],
            ['bodypart_5', 'bodypart_6'],
            ['bodypart_6', 'bodypart_0'],
            ['bodypart_7', 'bodypart_8'],
            ['bodypart_8', 'bodypart_9'],
            ['bodypart_9', 'bodypart_10'],
            ['bodypart_10', 'bodypart_11'],
            ['bodypart_11', 'bodypart_12'],
            ['bodypart_12', 'bodypart_13'],
            ['bodypart_13', 'bodypart_7'],
            ['bodypart_14', 'bodypart_15']
        ],
        'TrainingFraction': [0.9],
        'default_net_type': cnn,
        'iteration': 0,
        'batch_size': 8,
        'num_shuffles': 1,
        'trainingsiterations': 30000,
        'save_iters': 1000
    })

    deeplabcut.auxiliaryfunctions.write_config(config_path, cfg)

    # Step 4: Create Training Dataset
    print("Creating training dataset...")
    deeplabcut.create_training_dataset(
        str(config_path),
        num_shuffles=1,
        Shuffles=[1],
        augmenter_type='imgaug'
    )

    # Step 5: Train Network
    print("Training network...")
    deeplabcut.train_network(
        str(config_path),
        shuffle=1,
        displayiters=100,
        saveiters=1000,
        maxiters=300000,
        allow_growth=True
    )

    # Step 6: Evaluate Network
    print("Evaluating network...")
    deeplabcut.evaluate_network(str(config_path), plotting=True)

    return config_path


def analyze_videos(config_path, video_paths):
    deeplabcut.analyze_videos(
        config=config_path,
        videos=video_paths,
        videotype='mp4',  # Specify the video type if necessary
        save_as_csv=True)


def train():
    # Load your dataset
    frames = np.load('dataset.npy')  # Shape: (num_frames, 192, 192, 3)
    labels = np.load('labels.npy')  # Shape: (num_frames, 18, 2)
    for cnn in ['resnet_50', 'resnet_101', 'hrnet_w48']:
        config_path = train_dlc_model(
            frames=frames,
            labels=labels,
            working_directory=rf'projects\{cnn}',
            video_path=r"output.mp4",
            cnn=cnn,
        )


def get_distinct_colors(n):
    """Generate n distinct colors"""
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    return colors


def display_differences(all_dlc_preds, ground_truth):
    video_paths = ["labeled_cam_1.mp4", "labeled_cam_2.mp4", "labeled_cam_3.mp4", "labeled_cam_4.mp4"]
    # Open all videos
    caps = [cv2.VideoCapture(path) for path in video_paths]
    # Get video properties
    num_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    # Create figure and axes for 2x2 grid plus slider
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.1])
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    slider_ax = fig.add_subplot(gs[4:])
    # Get number of points per frame
    num_points = all_dlc_preds.shape[2]  # (frames, cameras, points, 2)
    colors = get_distinct_colors(num_points)
    # Create slider
    slider = Slider(slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
    current_frame = 0

    def update_frame(frame_idx):
        # Set frame position for all videos
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read frames from all videos
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Clear and update all plots
        for i, (frame, ax) in enumerate(zip(frames, axs)):
            ax.clear()
            ax.imshow(frame)

            # Plot each point with its own color
            for j in range(num_points):
                # Get points
                dlc_point = all_dlc_preds[frame_idx][i][j]
                gt_point = ground_truth[frame_idx][i][j]

                # Draw yellow line connecting corresponding points
                ax.plot([dlc_point[0], gt_point[0]],
                       [dlc_point[1], gt_point[1]],
                       color='yellow',
                       linestyle='-',
                       linewidth=1,
                       alpha=0.7)  # Slightly transparent

                # DLC predictions as 'o'
                ax.scatter(dlc_point[0], dlc_point[1],
                         c=[colors[j]], marker='o', s=100,
                         label=f'DLC Point {j + 1}' if i == 0 else None)

                # Ground truth as 'x'
                ax.scatter(gt_point[0], gt_point[1],
                         c=[colors[j]], marker='x', s=100,
                         label=f'GT Point {j + 1}' if i == 0 else None)

            ax.set_title(f'Camera {i + 1}')
            ax.axis('off')

            # if i == 0:
            #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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


def load_dlc_points(filepath):
    # Reload the CSV file
    df = pd.read_csv(filepath, header=None)

    # Skip the first two rows and rename columns for easier processing
    data = df.iloc[3:].reset_index(drop=True)
    num_frames = data.shape[0]
    num_body_parts = len(data.columns) // 3

    # Initialize an empty numpy array to store the data
    points_array = np.zeros((num_frames, num_body_parts, 3))

    data_np = np.array(data).astype(float)[:, 1:]
    # Loop over each body part and fill the numpy array
    for part in range(num_body_parts):
        try:
            x_col = 3 * part + 0  # x column index for the body part
            y_col = 3 * part + 1  # y column index for the body part
            l_col = 3 * part + 2  # likelihood column index for the body part

            points_array[:, part, 0] = data_np[:, x_col]
            points_array[:, part, 1] = data_np[:, y_col]
            points_array[:, part, 2] = data_np[:, l_col]
        except IndexError:
            print(f"Skipping part {part}: Index out of bounds")

    return points_array


def load_from_mat(path, name, end=100):
    mat = scipy.io.loadmat(path)
    numpy_array = mat[name]
    numpy_array = np.transpose(numpy_array, (3, 2, 0, 1))
    return numpy_array[:end]


def load_all_deep_lab_cut_preds(model='hrnet_w48'):
    if model == 'resnet_101':
        all_csvs = [f"projects/{model}/labeled_cam_{i}DLC_Resnet101_FliesJan14shuffle1_snapshot_100.csv" for i in
                    range(1, 5)]
    if model == 'hrnet_w48':
        all_csvs = [f"projects/{model}/labeled_cam_{i}DLC_HrnetW48_FliesJan14shuffle1_snapshot_060.csv" for i in
                    range(1, 5)]
    if model == 'resnet_50':
        all_csvs = [f"projects/{model}/labeled_cam_{i}DLC_Resnet50_FliesJan14shuffle1_snapshot_020.csv" for i in
                    range(1, 5)]
    all_preds = [load_dlc_points(csv) for csv in all_csvs]
    all_dlc_preds = np.concatenate([all_preds[i][:, np.newaxis, ...] for i in range(4)], axis=1)
    return all_dlc_preds


def load_gt():
    ground_truth_2D_path = (r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled "
                            r"dataset\ground_truth_labels.mat")
    ground_truth_2D = load_from_mat(ground_truth_2D_path, name='ground_truth')
    ground_truth_2D = ground_truth_2D[:, :, RELEVANT_FEATURE_POINTS]
    ground_truth_2D += np.random.normal(0, 0.000001, ground_truth_2D.shape)
    return ground_truth_2D


def run_analyze():
    # config_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\deep-lab-cut\projects\Flies-Amitai-2025-01-14\config.yaml"
    # config_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\deep-lab-cut\projects\projects%5Chrnet_w48\Flies-Amitai-2025-01-14\config.yaml"
    config_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\deep-lab-cut\projects\resnet_101\Flies-Amitai-2025-01-14\config.yaml"
    video_paths = ["labeled_cam_1.mp4", "labeled_cam_2.mp4", "labeled_cam_3.mp4", "labeled_cam_4.mp4"]
    analyze_videos(config_path, video_paths)


def load_my_network_predictions():
    reprojected_points_2D_path = (r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled "
                                      r"dataset\estimated_positions.mat")
    my_predictions = load_from_mat(reprojected_points_2D_path, name='positions')
    my_predictions = my_predictions[:100, :, RELEVANT_FEATURE_POINTS]
    return my_predictions


def medfilt_data(values, size=15):
    padsize = size + 5
    vpad = np.pad(values, (padsize, padsize), mode='median', stat_length=5)
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_data(vals):
    nans, ix = nan_helper(vals)
    out = np.copy(vals)
    if np.mean(nans) > 0.85:
        return out
    out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
    return out

def filter_2d_detections(detections, medfilt_size=15):
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
                values = interpolate_data(values)
                values = medfilt_data(values, size=medfilt_size)
                filtered_detections[:, cam_idx, point_idx, coord] = values

    return filtered_detections

def compare_dlc_preds_to_ground_truth():
    bad_frames = [2,   3, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 37,
                  70, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]  # using resnet 101
    all_dlc_preds_resnet_50 = load_all_deep_lab_cut_preds('resnet_50')[..., :2]
    all_dlc_preds_resnet_101 = load_all_deep_lab_cut_preds('resnet_101')[..., :2]
    all_dlc_preds_hrnet_w48 = load_all_deep_lab_cut_preds('hrnet_w48')[..., :2]
    ground_truth = load_gt()

    np.save("resnet_50.npy", all_dlc_preds_resnet_50)
    np.save("resnet_101.npy", all_dlc_preds_resnet_101)
    np.save("hrnet_w48.npy", all_dlc_preds_hrnet_w48)
    my_predictions = load_my_network_predictions()

    refined_detections = filter_2d_detections(all_dlc_preds_resnet_101)
    display_differences(refined_detections, ground_truth)

    my_error = np.linalg.norm(my_predictions - ground_truth, axis=-1).flatten()
    dlc_error_resnet_50 = np.linalg.norm(all_dlc_preds_resnet_50 - ground_truth, axis=-1).flatten()
    dlc_error_resnet_101 = np.linalg.norm(all_dlc_preds_resnet_101 - ground_truth, axis=-1).flatten()
    dlc_error_hrnet_w48 = np.linalg.norm(all_dlc_preds_hrnet_w48 - ground_truth, axis=-1).flatten()

    print(f"My mean error: {np.mean(my_error)}, median error: {np.median(my_error)}\n"
          f"DLC ResNet-50 mean error: {np.mean(dlc_error_resnet_50)}, median error: {np.median(dlc_error_resnet_50)}\n"
          f"DLC ResNet-101 mean error: {np.mean(dlc_error_resnet_101)}, median error: {np.median(dlc_error_resnet_101)}\n"
          f"DLC HRNet-W48 mean error: {np.mean(dlc_error_hrnet_w48)}, median error: {np.median(dlc_error_hrnet_w48)}\n")

    my_mean = np.mean(my_error)
    dlc_mean_resnet_50 = np.mean(dlc_error_resnet_50)
    dlc_mean_resnet_101 = np.mean(dlc_error_resnet_101)
    dlc_mean_hrnet_w48 = np.mean(dlc_error_hrnet_w48)

    # meam without broken frames
    good_frames = [i for i in range(100) if i not in bad_frames]
    my_mean_good_frames = np.mean(my_error[good_frames])
    dlc_mean_resnet_101_good_frames = np.mean(dlc_error_resnet_101[good_frames])
    pass
#
#


#
# def uncrop(points_2d, cropzone, add_one=True):
#     num_frames, num_cams, _ = points_2d.shape
#     new_shape = list(points_2d.shape)
#     if add_one:
#         new_shape[-1] += 1
#     points_2d_h = np.zeros(new_shape)
#     for frame in range(num_frames):
#         for cam in range(num_cams):
#             x = cropzone[frame, cam, 1] + points_2d[frame, cam, 0]
#             y = cropzone[frame, cam, 0] + points_2d[frame, cam, 1]
#             # do undistort
#             # if self.distortion:
#             #     x, y = self.undistort_point(cam, x, y)
#             y = 800 + 1 - y
#             if add_one:
#                 point = [x, y, 1]
#             else:
#                 point = [x, y]
#             points_2d_h[frame, cam, :] = point
#     return points_2d_h
#
#
# def create_camera_group(intrinsics, extrinsics):
#     """Create a CameraGroup object from intrinsics and extrinsics matrices"""
#     cameras = []
#     for cam_idx in range(len(intrinsics)):
#         rvec, _ = cv2.Rodrigues(extrinsics[cam_idx][:, :3])
#         tvec = extrinsics[cam_idx][:, 3]
#         cam = Camera(
#             matrix=intrinsics[cam_idx],
#             rvec=rvec.flatten(),
#             tvec=tvec,
#             name=f"cam_{cam_idx}"
#         )
#         cameras.append(cam)
#     return CameraGroup(cameras)
#
#
# def run_anipose_simple():
#     """Simple 3D reconstruction using basic triangulation"""
#     import h5py
#     calibration_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\calibration file.h5"
#     dataset_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\trainset_movie_1_370_520_ds_3tc_7tj.h5"
#
#     # Load calibration data
#     with h5py.File(calibration_path, "r") as f:
#         Ks = f["K_matrices"][:].T
#         rotations = f["rotation_matrices"][:].T
#         translations = f["translations"][:].T
#
#     # Normalize intrinsics
#     for i, K in enumerate(Ks):
#         K = K / K[2, 2]
#         Ks[i] = K
#
#     # Create extrinsics matrices
#     extrinsics = []
#     for cam in range(4):
#         R = rotations[cam]
#         t = translations[cam][:, np.newaxis]
#         extrinsics.append(np.hstack((R, t)))
#
#     # Create camera group
#     camera_group = create_camera_group(Ks, extrinsics)
#
#     # Load data
#     with h5py.File(dataset_path, "r") as f:
#         cropzone = f["cropzone"][:]
#
#     my_predictions = load_my_network_predictions()
#     ground_truth = load_gt()
#
#     # Uncrop predictions
#     uncropped_my_pred = np.stack([
#         uncrop(my_predictions[:, :, f_ind, :], cropzone, add_one=False)
#         for f_ind in range(16)
#     ], axis=2)
#
#     # Initialize array for 3D points
#     n_frames = uncropped_my_pred.shape[0]
#     n_points = uncropped_my_pred.shape[2]
#     points_3d = np.zeros((n_frames, n_points, 3))
#     points_3d_tg = np.zeros((n_frames, n_points, 3))
#
#     # Perform basic triangulation for each frame
#     print(f"Processing {n_frames} frames...")
#     for frame in range(n_frames):
#         points_3d[frame] = camera_group.triangulate(uncropped_my_pred[frame])
#         points_3d_tg[frame] = camera_group.triangulate(ground_truth[frame])
#
#     return points_3d


if __name__ == "__main__":
    compare_dlc_preds_to_ground_truth()
    pass
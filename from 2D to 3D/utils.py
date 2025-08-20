import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys
import glob
import csv
import scipy.io
import os
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit
from scipy.optimize import minimize, fsolve


haltere_root_x = haltere_root_x_image = 172
haltere_root_y = 240 - 106
haltere_root_y_image = 106

def find_starting_frame(readme_file):
    start_pattern = re.compile(r'start:\s*(\d+)')
    with open(readme_file, 'r') as file:
        for line in file:
            match = start_pattern.search(line)
            if match:
                start_number = match.group(1)
                return start_number


def get_start_frame(movie_dir_path):
    start_frame = 0
    for filename in os.listdir(movie_dir_path):
        if filename.startswith("README_mov"):
            readme_file = os.path.join(movie_dir_path, filename)
            start_frame = find_starting_frame(readme_file)
    return start_frame


def find_flip_in_files(movie_dir_path):
    # Word to search for
    word_to_search = "flip"

    # Regular expression pattern to match filenames like README_mov{some number}.txt
    pattern = re.compile(r"README_mov\d+\.txt")

    try:
        # List all files in the directory
        for filename in os.listdir(movie_dir_path):
            # Check if the filename matches the pattern
            if pattern.match(filename):
                file_path = os.path.join(movie_dir_path, filename)
                # Open the file and search for the word
                with open(file_path, 'r') as file:
                    for line in file:
                        if word_to_search in line:
                            return True
        return False
    except FileNotFoundError:
        # If the directory does not exist, return False
        return False


def get_movie_length(movie_dir_path):
    end_frame = 0
    points_path = os.path.join(movie_dir_path, "points_3D_ensemble_best.npy")
    points = np.load(points_path)
    num_frames = len(points)
    return num_frames



def clean_directory(base_path):
    # Loop through all items in the base directory
    for subdirectory in os.listdir(base_path):
        dir_path = os.path.join(base_path, subdirectory)

        # Check if the item is indeed a directory
        if os.path.isdir(dir_path):
            # Define the files to remove
            files_to_remove = ['points_3D_ensemble.npy', 'points_3D_smoothed_ensemble.npy']

            # Loop over the files to remove
            for filename in files_to_remove:
                file_path = os.path.join(dir_path, filename)
                # Check if the file exists, and if so, delete it
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted {file_path}", flush=True)

            # Check for the existence of 'smoothed_trajectory.html'
            if not os.path.exists(os.path.join(dir_path, 'smoothed_trajectory.html')):
                print(f"'smoothed_trajectory.html' does not exist in {subdirectory}", flush=True)


def get_scores_from_readme(readme_path):
    with open(readme_path, 'r') as file:
        file_content = file.read()

    # Regular expression to extract floating-point numbers in scientific notation
    scores = re.findall(r'\d+\.\d+e[-+]\d+', file_content)

    # Convert extracted strings to float for numerical operations if necessary
    first_score = float(scores[0]) if scores else None
    second_score = float(scores[1]) if len(scores) > 1 else None
    return first_score, second_score


def summarize_results(base_path):
    readme_name = "README_scores_3D_ensemble.txt"
    output_file = os.path.join(base_path, 'summary_results.csv')

    # Prepare to write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['movie_num', 'start_frame', 'length_frame', 'start_ms', 'length_ms', 'score1', 'score2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over directories in the base path
        for movie in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, movie)):
                movie_path = os.path.join(base_path, movie)
                try:
                    start_frame = int(get_start_frame(movie_path))
                    length_frame = int(get_movie_length(movie_path))
                    start_ms = int(start_frame / 16)
                    length_ms = int(length_frame / 16)
                    movie_num = int(re.findall(r'\d+', movie)[0])
                    readme_path = os.path.join(movie_path, readme_name)
                    score1, score2 = get_scores_from_readme(readme_path)

                    # Write the results to CSV
                    writer.writerow({
                        'movie_num': movie_num,
                        'start_frame': start_frame,
                        'length_frame': length_frame,
                        'start_ms': start_ms,
                        'length_ms': length_ms,
                        'score1': score1,
                        'score2': score2
                    })
                except Exception as e:
                    print(f"the exception {e} occurred in movie: {movie_path}\n")

    # Read and sort data after writing is completed
    with open(output_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    sorted_data = sorted(data, key=lambda x: float(x['length_frame']))

    # Write sorted data back to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_data)


def npy_to_mat(npy_file_path):
    # Load the .npy file
    data = np.load(npy_file_path)

    # Extract the filename without the extension
    filename = os.path.splitext(os.path.basename(npy_file_path))[0]

    # Define the output .mat file path
    mat_file_path = os.path.join(os.path.dirname(npy_file_path), f"{filename}.mat")

    # Save the data to a .mat file
    scipy.io.savemat(mat_file_path, {filename: data})

    print(f"Saved {npy_file_path} to {mat_file_path}")


def annotate_video_with_points(excel_path, video_path, output_path):
    # Load the Excel file
    df = pd.read_csv(excel_path, skiprows=2)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the corresponding coordinates for the current frame
        if frame_idx < len(df):
            x = int(np.round(df.loc[frame_idx, ('x')]))
            y = int(np.round(df.loc[frame_idx, ('z')]))


            # Draw the point on the frame
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)  # Red dot with radius 5
            cv2.circle(frame, (haltere_root_x_image, haltere_root_y_image), 1, (0, 255, 255), -1)

        # Write the frame to the output video
        out.write(frame)

        frame_idx += 1

    # Release everything when done
    cap.release()
    out.release()

    print(f"Annotated video saved at {output_path}")


def fit_ellipse(points):
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Design matrix for the general conic equation of an ellipse
    D1 = np.column_stack([x ** 2, x * y, y ** 2])
    D2 = np.column_stack([x, y, np.ones_like(x)])

    # Form the scatter matrix
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)

    # Solve for the ellipse parameters
    T = -np.linalg.inv(S3).dot(S2.T)
    M = S1 + np.dot(S2, T)

    # Solve the generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(M)

    # Find the eigenvector corresponding to the smallest eigenvalue
    cond = 4 * eigvecs[0, :] * eigvecs[2, :] - eigvecs[1, :] ** 2
    a1 = eigvecs[:, np.argmax(cond)]

    # Solve for the rest of the ellipse parameters
    a2 = np.dot(T, a1)

    # Full solution vector
    coeff = np.concatenate([a1, a2])
    return ellipse_params(coeff)


def ellipse_params(coeff):
    # Extract ellipse coefficients
    A, B, C, D, E, F = coeff

    # Calculate the center of the ellipse
    x0 = (C * D - B * E) / (B ** 2 - A * C)
    y0 = (A * E - B * D) / (B ** 2 - A * C)

    # Calculate the orientation of the ellipse
    theta = 0.5 * np.arctan(B / (A - C))

    # Calculate the semi-major and semi-minor axes
    term = np.sqrt((A - C) ** 2 + B ** 2)
    a = np.sqrt(2 * (A * x0 ** 2 + C * y0 ** 2 + B * x0 * y0 - F) / (A + C - term))
    b = np.sqrt(2 * (A * x0 ** 2 + C * y0 ** 2 + B * x0 * y0 - F) / (A + C + term))

    return (x0, y0), a, b, np.degrees(theta)


def fit_circle(points):
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Set up the system of equations
    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    B = x ** 2 + y ** 2

    # Solve for D, E, F in the equation: A * [D, E, F]^T = B
    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    D, E, F = C

    # Calculate the center (a, b) and radius r
    a = -D / 2
    b = -E / 2
    r = np.sqrt(a ** 2 + b ** 2 + F)

    return a, b, r


def fit_parametric_model(data):
    # Define the parametric model function
    def r_xz_model(t, a, b, c, R):
        x = R * (-b * np.cos(t) / np.sqrt(a**2 + b**2) - (a * c * np.sin(t)) / np.sqrt((a**2 + b**2) * c**2 + (a**2 + b**2)**2))
        z = R * ((a**2 + b**2) * np.sin(t) / np.sqrt((a**2 + b**2) * c**2 + (a**2 + b**2)**2))
        return np.array([x, z])

    # Extract x and z data from the dataset
    x_data = data[:, 0]
    z_data = data[:, 1]

    # Define a function to fit the model to the data
    def fit_function(t, a, b, c, R):
        return r_xz_model(t, a, b, c, R).ravel()

    # Initial guess for the parameters
    initial_guess = [0.82, 0.53, -0.189, 21]

    # Perform curve fitting
    t_guess = np.linspace(-np.pi/2, np.pi/2, len(x_data))  # Initial guess for t values
    params, covariance = curve_fit(fit_function, t_guess, np.hstack([x_data, z_data]), p0=initial_guess)

    a_fit, b_fit, c_fit, R_fit = params

    # Generate fitted data
    t_fit = np.linspace(-np.pi/2, np.pi/2, 100)
    fitted_xz = r_xz_model(t_fit, a_fit, b_fit, c_fit, R_fit)

    guess_vector = np.array([0.82, 0.53, -0.189])
    # guess_vector = R.from_euler('y', 45, degrees=False).apply(guess_vector)
    guess_vector = -guess_vector

    est_x, est_z = r_xz_model(t=t_guess, a=guess_vector[0], b=guess_vector[1], c=guess_vector[2], R=21)

    # Plot the original data and the fitted curve
    plt.scatter(-x_data, z_data, color='blue', label='Original Data')
    plt.plot(est_x, est_z, color='red', label='Fitted Curve')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.show()

    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}, c = {c_fit}, R = {R_fit}")

    return a_fit, b_fit, c_fit, R_fit


def compute_y(x, z, a=-0.82, b=-0.53, c=-(-0.189), R=21):
    # Compute t from the z equation
    t = np.arcsin(z * np.sqrt((a**2 + b**2) * c**2 + (a**2 + b**2)**2) / (R * (a**2 + b**2)))

    # Calculate cos(t) and sin(t) for the given t
    cos_t = np.sqrt(1 - np.sin(t)**2)
    sin_t = np.sin(t)

    # Calculate y using the y equation
    y = R * (a * cos_t / np.sqrt(a**2 + b**2)) + R * (-b * c * sin_t / np.sqrt((a**2 + b**2) * c**2 + (a**2 + b**2)**2))

    return y


def find_best_fit_parameters(x_z_data):
    """
    This function takes an array of size (N, 2) with pairs (x, z) and finds the
    best-fitting parameters alpha, beta, and R that describe the data based on
    the given parametric equations for a circle embedded in a plane.

    Parameters:
    x_z_data: numpy array of shape (N, 2) where each row is (x_i, z_i)

    Returns:
    alpha_opt: Optimal alpha (in radians)
    beta_opt: Optimal beta (in radians)
    R_opt: Optimal radius R
    """

    def x_t(t, alpha, beta, R):
        return R * (np.cos(alpha) * np.cos(beta) * np.cos(t) + np.cos(alpha) * np.sin(beta) * np.sin(t))

    def z_t(t, beta, R):
        return R * (-np.sin(beta) * np.cos(t))

    # Objective function to minimize
    def objective(params, x_data, z_data):
        alpha, beta, R = params
        error = 0
        for x_i, z_i in zip(x_data, z_data):
            # Solve for t_i using z_i
            t_i = np.arccos(-z_i / (R * np.sin(beta)))

            # Compute the error between the observed and predicted x, z
            error += (x_t(t_i, alpha, beta, R) - x_i) ** 2 + (z_t(t_i, beta, R) - z_i) ** 2
        return error

    # Extract x and z from the input data
    x_data = x_z_data[:, 0]
    z_data = x_z_data[:, 1]

    # Initial guess for the parameters [alpha, beta, R]
    initial_guess = [0.588, 1.76, 21]

    # Perform the optimization
    result = minimize(objective, initial_guess, args=(x_data, z_data), method='Nelder-Mead')
    result.message
    # Extract the optimized parameters
    alpha_opt, beta_opt, R_opt = result.x

    # Return the optimal parameters
    return alpha_opt, beta_opt, R_opt


def compute_x_z(t, alpha_opt, beta_opt, R_opt):
    x_est = R_opt * (
                np.cos(alpha_opt) * np.cos(beta_opt) * np.cos(t) + np.cos(alpha_opt) * np.sin(beta_opt) * np.sin(t))
    z_est = R_opt * (-np.sin(beta_opt) * np.cos(t))

    return x_est, z_est


def predict_3D_points_all_pairs(base_path):
    all_points_file_list = []
    points_3D_file_list = []
    dir_path = os.path.join(base_path)
    dirs = glob.glob(os.path.join(dir_path, "*"))
    for dir in dirs:
        if os.path.isdir(dir):
            all_points_file = os.path.join(dir, "points_3D_all.npy")
            points_3D_file = os.path.join(dir, "points_3D.npy")
            if os.path.isfile(all_points_file):
                all_points_file_list.append(all_points_file)
            if os.path.isfile(points_3D_file):
                points_3D_file_list.append(points_3D_file)
    all_points_arrays = [np.load(array_path) for array_path in all_points_file_list]
    points_3D_arrays = [np.load(array_path)[:, :, np.newaxis, :] for array_path in points_3D_file_list]
    big_array_all_points = np.concatenate(all_points_arrays, axis=2)
    return big_array_all_points, all_points_arrays


def add_nan_frames(original_array, N):
    nan_frames = np.full((N,) + original_array.shape[1:], np.nan)
    new_array = np.concatenate((nan_frames, original_array), axis=0)
    return new_array


def analyse_video(excel_path):
    from extract_flight_data import FlightAnalysis, extract_yaw_pitch
    df = pd.read_csv(excel_path, skiprows=2)
    last_frame = -1

    # load all points
    x = -(df['x'].values[:last_frame] - haltere_root_x)
    z = 240 - df['z'].values[:last_frame] - haltere_root_y
    # y = compute_y(x=x, z=z)
    points_2D = np.column_stack((x, z))

    # dispaly_point_cloud(np.column_stack((x, y, z)))

    alpha_opt, beta_opt, R_opt = find_best_fit_parameters(points_2D)
    alpha_opt, beta_opt, R_opt = -2.568, 0.191, 21
    t = np.linspace(-np.pi, np.pi, 200)
    x_est, z_est = compute_x_z(t, alpha_opt, beta_opt, R_opt)

    plt.scatter(x_est, z_est, color='red')
    plt.scatter(x, z, color='blue')
    plt.show()

    # get unit vectors
    vec_2D = points_2D / np.linalg.norm(points_2D, axis=-1)[:, np.newaxis]
    # get the 2D radii
    radii = np.linalg.norm(points_2D, axis=-1)

    # get gama (elevation in the xz plane)
    gama = np.degrees(np.arctan2(vec_2D[:, 1], vec_2D[:, 0]))

    # calculate phi
    radii_min_90 = radii[(gama > -91) & (gama < -89)]
    radii_0 = radii[(gama < 1) & (gama > -1)]
    phi_plane = np.degrees(np.arcsin(np.median(radii_0)/np.median(radii_min_90)))

    # according to Ipad paining
    # L = np.median(radii_min_90)
    L = np.median(radii_min_90) / np.cos(np.radians(phi_plane))  # try the original

    theta_3D = np.degrees(np.arccos(z/L))
    phi_3D = np.degrees(np.arccos((radii * np.cos(np.radians(gama))/(L * np.sin(np.radians(theta_3D))))))

    est_x = radii * np.sin(np.radians(theta_3D)) * np.cos(np.radians(phi_3D))
    est_y = radii * np.sin(np.radians(theta_3D)) * np.sin(np.radians(phi_3D))
    est_z = radii * np.cos(np.radians(theta_3D))

    est_3D = np.column_stack((est_x, est_y, est_z))
    est_3D = est_3D[~np.isnan(est_3D).any(axis=1)]

    xyz_rotated = R.from_euler('y', -45, degrees=True).apply(est_3D)
    plane, error = FlightAnalysis.fit_plane(xyz_rotated)

    dispaly_point_cloud(xyz_rotated)
    yaw, pitch = extract_yaw_pitch(plane[:-1])


    # older calculations
    # L = np.median(radii_min_90) / np.sin(np.radians(phi_plane))
    # # transfer from gama to theta in Spherical coordinate system
    # theta_2D = 90 - gama
    # theta = np.degrees(np.arccos((radii/L) * np.cos(np.radians(theta_2D))))
    # phi = np.degrees(np.arcsin(x / (L*np.sin(np.radians(theta)))))
    #
    # est_x = radii * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    # est_y = radii * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    # est_z = radii * np.cos(np.radians(theta))
    #
    # plt.plot(phi, theta, marker='o')
    # plt.axis('equal')
    # plt.show()
    #
    # # Create a rotation object for -45 degrees around the z-axis
    # xyz = np.column_stack((est_x, est_y, est_z))
    # dispaly_point_cloud(xyz)

    # xyz_rotated = R.from_euler('y', -45, degrees=True).apply(xyz)
    # dispaly_point_cloud(xyz_rotated)


def dispaly_point_cloud(xyz):
    fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='b')
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def from_numpy_to_mp4():
    path_h5 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\example datasets\mov53\movie_53_10_2398_ds_3tc_7tj.h5"
    output_path = "output.mp4"
    box = h5py.File(path_h5, 'r')['/box'][:100, 4][..., np.newaxis]
    image_arrays = np.concatenate((box, box, box), axis=-1)
    # Get the shape from the first image
    height, width = image_arrays[0].shape[:2]

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

    # Write each frame
    for img in image_arrays:
        # OpenCV expects BGR format
        # If your arrays are in RGB format, convert them
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(bgr_img)

    # Release the video writer
    out.release()


if __name__ == '__main__':
    from_numpy_to_mp4()
    excel_path = r"C:\Users\amita\Downloads\doi_10_5061_dryad_g4f4qrfnr__v20201221\MJR_JLF_HaltereKinematicsData\MagPulseData\RepresentativeTraceFigures\LEFTCAM_20190919_1623DeepCut_resnet50_UntreatedHaltTrackerNov6shuffle1_1030000.csv"
    video_path = r"C:\Users\amita\Downloads\doi_10_5061_dryad_g4f4qrfnr__v20201221\MJR_JLF_HaltereKinematicsData\MagPulseData\RepresentativeTraceFigures\LEFTCAM_20190919_1623.avi"
    output_path = r"C:\Users\amita\Downloads\doi_10_5061_dryad_g4f4qrfnr__v20201221\MJR_JLF_HaltereKinematicsData\MagPulseData\RepresentativeTraceFigures\new_video.avi"
    # analyse_video(excel_path)
    # annotate_video_with_points(excel_path, video_path, output_path)

    # npy_file_path = r"G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\points_ensemble_smoothed_reprojected.npy"
    # npy_to_mat(npy_file_path)
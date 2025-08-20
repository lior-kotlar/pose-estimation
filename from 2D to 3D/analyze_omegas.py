import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from extract_flight_data import FlightAnalysis
from visualize import Visualizer
import scipy
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
import scipy.io
import os
import h5py
import multiprocessing as mp
from scipy.spatial.transform import Rotation
import pickle

from functools import partial


# matplotlib.use('TkAgg')


def check_high_blind_axis_omegas(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["body_wx_take", "body_wy_take", "body_wz_take"])
    # Extract omega values for Mahalanobis filtering
    omega, _, _, _ = get_3D_attribute_from_df(df)

    # Compute the mean and covariance matrix of omega
    mean = np.mean(omega, axis=0)
    covariance_matrix = np.cov(omega, rowvar=False)

    # Filter omega rows based on Mahalanobis distance
    filtered_indices = []
    for i, row in enumerate(omega):
        dist = mahalanobis(row, mean, np.linalg.inv(covariance_matrix))
        if dist < 3:
            filtered_indices.append(i)

    # Filter the dataframe based on Mahalanobis distance
    filtered_df = df.iloc[filtered_indices]
    filtered_omega, _, _, _ = get_3D_attribute_from_df(filtered_df)
    vec_all, yaw_all, pitch_all, yaw_std_all, pitch_std_all = get_pca_points(filtered_omega)

    # Project omega vectors onto vec_all and calculate the distance from the origin
    projections = np.dot(filtered_omega, vec_all)
    distances = np.abs(projections)

    # Define a threshold for filtering based on distances (for example, top 20% based on distance)
    top = 20
    threshold_distance = np.percentile(distances, 100 - top)

    # Filter rows where the projection distance is within the threshold
    final_filtered_indices = [i for i, distance in enumerate(distances) if distance >= threshold_distance]
    all_indices_filtered_df = np.arange(len(distances))
    remaining_indices = list(set(all_indices_filtered_df) - set(final_filtered_indices))

    # Get the corresponding original indices
    final_filtered_df = filtered_df.iloc[final_filtered_indices]
    non_filtered_df = filtered_df.iloc[remaining_indices]

    high_omegas, _, _, _ = get_3D_attribute_from_df(final_filtered_df)
    rest_of_omegas, _, _, _ = get_3D_attribute_from_df(non_filtered_df)
    high_omegas_torques, _, _, _ = get_3D_attribute_from_df(final_filtered_df, attirbutes=["torque_body_x_take",
                                                                                           "torque_body_y_take",
                                                                                           "torque_body_z_take"])
    rest_of_torques, _, _, _ = get_3D_attribute_from_df(non_filtered_df, attirbutes=["torque_body_x_take",
                                                                                     "torque_body_y_take",
                                                                                     "torque_body_z_take"])

    dir = os.path.dirname(csv_file)
    output_file_path = os.path.join(dir, 'high 20 percent omegas.csv')
    final_filtered_df.to_csv(output_file_path, index=False)

    # Plotting
    fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(high_omegas[:, 0], high_omegas[:, 1], high_omegas[:, 2], s=1, color='blue', label='top 20')
    # ax.scatter(rest_of_omegas[:, 0], rest_of_omegas[:, 1], rest_of_omegas[:, 2], s=1, color='red', label='rest')
    ax.scatter(rest_of_torques[:, 0], rest_of_torques[:, 1], rest_of_torques[:, 2], s=5, color='red', label='rest')
    ax.set_aspect('equal')
    plt.legend()
    plt.show()


def create_rotating_frames(N, dt, omegas):
    x_body = np.array([1, 0, 0])
    y_body = np.array([0, 1, 0])
    z_body = np.array([0, 0, 1])

    # Create arrays to store the frames
    x_frames = np.zeros((N, 3))
    y_frames = np.zeros((N, 3))
    z_frames = np.zeros((N, 3))

    # Initialize the frames with the initial reference frame
    x_frames[0] = x_body
    y_frames[0] = y_body
    z_frames[0] = z_body

    for i in range(1, N):
        omega = omegas[i]
        # Construct the skew-symmetric matrix for omega
        omega_matrix = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])

        # Apply the angular velocity tensor to generate the frames
        R = np.stack((x_frames[i - 1], y_frames[i - 1], z_frames[i - 1]), axis=-1)
        dR = np.dot(omega_matrix, R) * dt
        R_new = R + dR

        # Ensure orthogonality and normalization
        U, _, Vt = np.linalg.svd(R_new, full_matrices=False)
        R_new = np.dot(U, Vt)

        x_frames[i] = R_new[:, 0]
        y_frames[i] = R_new[:, 1]
        z_frames[i] = R_new[:, 2]

    return x_frames, y_frames, z_frames


def experiment():
    # Initial reference frame
    N = 10000
    dt = 1
    t = np.linspace(0, 2 * np.pi, N)  # Generate time array
    # Generate N omegas with sine function
    omegas = np.vstack([
        0.0001 * np.sin(t),
        0.0002 * np.sin(2 * t),
        0.0001 * np.sin(3 * t)
    ]).T

    x_frames, y_frames, z_frames = create_rotating_frames(N, dt, omegas)

    omega_lab, omega_body, angular_speed_lab, angular_speed_body = FlightAnalysis.get_angular_velocities(
        x_frames, y_frames, z_frames, start_frame=0, end_frame=N, sampling_rate=1)
    omega_lab = np.radians(omega_lab)

    percentage_error = np.abs((omegas - omega_lab) / (omegas + 0.00000001)) * 100
    mean_percentage_error = percentage_error.mean(axis=0)

    plt.plot(omegas, color='r')
    plt.plot(omega_lab, color='b')
    plt.show()
    return


def create_rotating_frames_yaw_pitch_roll(N, yaw_angles, pitch_angles, roll_angles):
    x_body = np.array([1, 0, 0])
    y_body = np.array([0, 1, 0])
    z_body = np.array([0, 0, 1])

    # Create arrays to store the frames
    x_frames = np.zeros((N, 3))
    y_frames = np.zeros((N, 3))
    z_frames = np.zeros((N, 3))

    # Initialize the frames with the initial reference frame
    x_frames[0] = x_body
    y_frames[0] = y_body
    z_frames[0] = z_body

    for i in range(0, N):
        # Get the yaw, pitch, and roll angles for the current frame
        yaw_angle = yaw_angles[i]
        pitch_angle = pitch_angles[i]
        roll_angle = roll_angles[i]

        R = FlightAnalysis.euler_rotation_matrix(yaw_angle, pitch_angle, roll_angle).T

        # Apply the rotation to the initial body frame
        x_frames[i] = R @ x_body
        y_frames[i] = R @ y_body
        z_frames[i] = R @ z_body

    return x_frames, y_frames, z_frames


# Example usage
def experiment_2(what_to_enter):
    # what_to_enter: coule be either omega or yaw, pitch roll
    N = 1000
    if what_to_enter == 'omega':
        dt = 1
        # Generate N omegas with sine function
        d = 0.001
        wx = np.zeros(N)
        wy = np.zeros(N)
        wz = np.zeros(N)

        # wx = 3 * d * np.linspace(0, 10, N)
        wy = 2 * d * np.ones(N)
        wz = 10 * d * np.ones(N)
        omegas = np.vstack([
            wx,
            wy,
            wz
        ]).T

        x_frames, y_frames, z_frames = create_rotating_frames(N=N, omegas=omegas, dt=1)
        Rs = np.stack((x_frames, y_frames, z_frames), axis=-1)

        # rs = [scipy.spatial.transform.Rotation.from_matrix(Rs[i]) for i in range(N)]
        # yaw = np.array([rs[i].as_euler('zyx', degrees=False)[0] for i in range(N)])
        # pitch = np.array([rs[i].as_euler('zyx', degrees=False)[1] for i in range(N)])
        # roll = np.array([rs[i].as_euler('zyx', degrees=False)[2] for i in range(N)])

        yaw = np.unwrap(np.array([np.arctan2(r[1, 0], r[0, 0]) for r in Rs]))
        pitch = -np.unwrap(np.array([np.arcsin(-r[2, 0]) for r in Rs]), period=np.pi)
        roll = np.unwrap(np.array([np.arctan2(r[2, 1], r[2, 2]) for r in Rs]))

        yaw_mine = np.radians(FlightAnalysis.get_body_yaw(x_frames))
        pitch_mine = np.radians(FlightAnalysis.get_body_pitch(x_frames))
        roll_mine = np.radians(FlightAnalysis.get_body_roll(phi=np.degrees(yaw_mine),
                                                            theta=np.degrees(pitch_mine),
                                                            x_body=x_frames,
                                                            y_body=y_frames,
                                                            yaw=np.degrees(yaw_mine),
                                                            pitch=np.degrees(pitch_mine),
                                                            start=0,
                                                            end=N, ))

        is_close = (np.all(np.isclose(yaw_mine, yaw))
                    and np.all(np.isclose(pitch_mine, pitch))
                    and np.all(np.isclose(roll_mine, roll)))
        print(f"is mine like the other way? {is_close}")

        plt.title("yaw, pitch, roll")
        plt.plot(yaw, label='yaw', c='r')
        plt.plot(pitch, label='pitch', c='g')
        plt.plot(roll, label='roll', c='b')
        plt.plot(yaw_mine, label='yaw mine', c='r', linestyle='--')
        plt.plot(pitch_mine, label='pitch mine', c='g', linestyle='--')
        plt.plot(roll_mine, label='roll mine', c='b', linestyle='--')
        plt.legend()
        plt.show()
        pass
    else:
        yaw = np.zeros(N)
        pitch = np.zeros(N)
        roll = np.zeros(N)

        yaw = np.linspace(0, 2 * 2 * np.pi, N)  # Example yaw angles for each frame
        pitch = np.linspace(0, 2 * 2 * np.pi, N)  # Example pitch angles for each frame
        roll = np.linspace(0, 2 * 2 * np.pi, N)  # Example roll angles for each frame
        x_frames, y_frames, z_frames = create_rotating_frames_yaw_pitch_roll(N, yaw, pitch, roll)

    yaw_dot = np.gradient(yaw)
    roll_dot = np.gradient(roll)
    pitch_dot = np.gradient(pitch)

    p, q, r = FlightAnalysis.get_pqr_calculation(-np.degrees(pitch), -np.degrees(pitch_dot), np.degrees(roll),
                                                 np.degrees(roll_dot),
                                                 np.degrees(yaw_dot))
    wx_1, wy_1, wz_1 = p, q, r
    omega_lab, omega_body, _, _ = FlightAnalysis.get_angular_velocities(x_frames, y_frames, z_frames, start_frame=0,
                                                                        end_frame=N, sampling_rate=1)
    omega_lab, omega_body = np.radians(omega_lab), np.radians(omega_body)
    wx_2, wy_2, wz_2 = omega_body[:, 0], omega_body[:, 1], omega_body[:, 2]

    # plt.plot(omega_body)
    # plt.plot(omega_lab, linestyle='--')
    # plt.show()
    # Plot all values in one plot

    plt.title("yaw, pitch, roll")
    plt.plot(yaw, label='yaw')
    plt.plot(pitch, label='pitch')
    plt.plot(roll, label='roll')
    plt.legend()
    plt.show()

    plot_pqr = True
    plot_est_omega = True
    plt.figure(figsize=(12, 8))
    if plot_pqr:
        plt.plot(wx_1, label='p -> wx', c='b')
        plt.plot(wy_1, label='q -> wy', c='r')
        plt.plot(wz_1, label='r -> wz', c='g')
    if plot_est_omega:
        plt.plot(wx_2, label='est wx', linestyle='--', c='b')
        plt.plot(wy_2, label='est wy', linestyle='--', c='r')
        plt.plot(wz_2, label='est wz', linestyle='--', c='g')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocities: wx, wy, wz')
    plt.legend()
    plt.show()

    omega = np.column_stack((p, q, r)) * 50
    # omega = omega_body * 500
    Visualizer.visualize_rotating_frames(x_frames, y_frames, z_frames, omega)
    pass


def get_omegas(csv_path):
    df = pd.read_csv(csv_path)
    df_dir1, df_dir2 = filter_between_light_dark(df)
    no_dark = get_3D_attribute_from_df(df_dir1)
    with_dark = get_3D_attribute_from_df(df_dir2)
    return no_dark, with_dark


def filter_between_light_dark(df):
    df_dir1 = df[df['wingbit'].str.startswith('dir1')]
    df_dir2 = df[df['wingbit'].str.startswith('dir2')]
    return df_dir1, df_dir2


def get_3D_attribute_from_df(df, attirbutes=["body_wx_take", "body_wy_take", "body_wz_take"]):
    wx = df[attirbutes[0]].values
    wx = wx[~np.isnan(wx)]
    wy = df[attirbutes[1]].values
    wy = wy[~np.isnan(wy)]
    wz = df[attirbutes[2]].values
    wz = wz[~np.isnan(wz)]
    omega = np.column_stack((wx, wy, wz))
    return omega, wx, wy, wz


def extract_yaw_pitch(vector):
    # Extract components
    v_x, v_y, v_z = vector

    # Calculate yaw angle
    yaw = np.arctan2(v_y, v_x)

    # Calculate pitch angle
    pitch = np.arcsin(v_z / np.linalg.norm(vector))

    # Convert from radians to degrees
    yaw_degrees = np.degrees(yaw)
    pitch_degrees = np.degrees(pitch)

    return yaw_degrees, pitch_degrees


def reconstruct_vector(yaw_degrees, pitch_degrees):
    # Convert angles from degrees to radians
    yaw = np.radians(yaw_degrees)
    pitch = np.radians(pitch_degrees)

    # Create rotation for yaw around z-axis
    r_yaw = R.from_euler('z', yaw, degrees=False)

    # Create rotation for pitch around y-axis
    r_pitch = R.from_euler('y', pitch, degrees=False)

    # Apply rotations to the original unit vector (1, 0, 0)
    initial_vector = np.array([1, 0, 0])
    rotated_vector = r_pitch.apply(r_yaw.apply(initial_vector))

    return rotated_vector


def scratch():
    vector = np.array([0.81, -0.53, -0.2])
    vector /= np.linalg.norm(vector)
    # Convert angles from degrees to radians

    yaw = np.arctan2(vector[1], vector[0])
    pitch = np.arctan2(vector[2], np.sqrt(vector[0] ** 2 + vector[1] ** 2))
    pitch_ = np.arcsin(vector[2])

    # vector = np.array([1,0,0])
    # yaw = -np.radians(30)
    # pitch = -np.radians(18.6)

    # Reconstruct the original vector from yaw and pitch
    reconstructed_vector = np.array([
        np.cos(-pitch) * np.cos(yaw),
        np.cos(-pitch) * np.sin(yaw),
        -np.sin(-pitch)
    ])

    yaw_degrees, pitch_degrees = extract_yaw_pitch(vector)


def compute_yaw_pitch(vec_bad):
    if vec_bad[0] < 0:
        vec_bad *= -1
    only_xy = vec_bad[[0, 1]] / np.linalg.norm(vec_bad[[0, 1]])
    yaw = np.rad2deg(np.arctan2(only_xy[1], only_xy[0]))
    pitch = np.rad2deg(np.arcsin(vec_bad[2]))

    # print(f"yaw: {yaw}, pitch: {pitch}")
    return yaw, pitch


def display_good_vs_bad_haltere(good_haltere, bad_haltere, use_both_light_and_dark=True, rotate=False, use_plotly=True):
    no_dark, with_dark = get_omegas(bad_haltere)
    omega_light, _, _, _ = no_dark
    omega_dark, _, _, _ = with_dark

    omega_good, wx_good, wy_good, wz_good = get_3D_attribute_from_df(pd.read_csv(good_haltere))
    omega_bad, wx_bad, wy_bad, wz_bad = get_3D_attribute_from_df(pd.read_csv(bad_haltere))

    # let us take only the ωₕ of the cut fly without dark
    omega_bad = omega_light

    if use_both_light_and_dark:
        omega_bad = np.concatenate((omega_light, omega_dark), axis=0)

    mahal_dist_bad = calculate_mahalanobis_distance(omega_bad)
    omega_bad = omega_bad[mahal_dist_bad < 3]

    norm_bad = np.linalg.norm(omega_bad, axis=1).mean()

    mahal_dist_good = calculate_mahalanobis_distance(omega_good)
    omega_good = omega_good[mahal_dist_good < 3]

    norm_good = np.linalg.norm(omega_good, axis=1).mean()

    vec_good, yaw_good, pitch_good, yaw_std_good, pitch_std_good = get_pca_points(omega_good)
    vec_bad, yaw_bad, pitch_bad, yaw_std_bad, pitch_std_bad = get_pca_points(omega_bad)

    print(f"yaw angle of bad haltere: {yaw_bad}, and the std is {yaw_std_bad}, "
          f"the pitch of the bad haltere is {pitch_bad} and the std is {pitch_std_bad}")

    # Calculate variance explained by first principal component
    variance_good = pca_variance_explained(omega_good)
    variance_bad = pca_variance_explained(omega_bad)

    p1_good, p2_good = omega_good.mean(axis=0) + 10000 * vec_good, omega_good.mean(axis=0) - 10000 * vec_good
    p1_bad, p2_bad = omega_bad.mean(axis=0) + 10000 * vec_bad, omega_bad.mean(axis=0) - 10000 * vec_bad

    # Rotate body axis quivers
    size = 5000
    quivers = [
        {'x': [0, size], 'y': [0, 0], 'z': [0, 0], 'color': 'red', 'name': r'$x_{body}$'},
        {'x': [0, 0], 'y': [0, size], 'z': [0, 0], 'color': 'green', 'name': r'$y_{body}$'},
        {'x': [0, 0], 'y': [0, 0], 'z': [0, size], 'color': 'orange', 'name': r'$z_{body}$'}
    ]

    if rotate:
        theta = np.radians(-45)
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        omega_good = omega_good @ rotation_matrix.T
        omega_bad = omega_bad @ rotation_matrix.T
        p1_good = p1_good @ rotation_matrix.T
        p2_good = p2_good @ rotation_matrix.T
        p1_bad = p1_bad @ rotation_matrix.T
        p2_bad = p2_bad @ rotation_matrix.T

        for q in quivers:
            rotated_points = np.array([q['x'], q['y'], q['z']]).T @ rotation_matrix.T
            q['x'], q['y'], q['z'] = rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2]

    if use_plotly:
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=omega_good[:, 0], y=omega_good[:, 1], z=omega_good[:, 2],
            mode='markers',
            marker=dict(size=1, color='red'),
            name=r'$\omega_{body}$ of intact flies'
        ))

        fig.add_trace(go.Scatter3d(
            x=omega_bad[:, 0], y=omega_bad[:, 1], z=omega_bad[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue'),
            name=r'$\omega_{body}$ of severed flies'
        ))

        fig.add_trace(go.Scatter3d(
            x=[p1_bad[0], p2_bad[0]], y=[p1_bad[1], p2_bad[1]], z=[p1_bad[2], p2_bad[2]],
            mode='lines',
            line=dict(color='blue', width=2),
            name=r'Dominant $\omega_{body}$ axis'
        ))

        for q in quivers:
            fig.add_trace(go.Scatter3d(
                x=q['x'], y=q['y'], z=q['z'],
                mode='lines+text',
                line=dict(color=q['color'], width=5),
                name=q['name']
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title=dict(
                    text=r'$\omega_x$ (deg/s)',
                    font=dict(size=24)
                ),
                yaxis_title=dict(
                    text=r'$\omega_y$ (deg/s)',
                    font=dict(size=24)
                ),
                zaxis_title=dict(
                    text=r'$\omega_z$ (deg/s)',
                    font=dict(size=24)
                ),
                xaxis=dict(tickfont=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
                zaxis=dict(tickfont=dict(size=14)),
                aspectmode='data'
            ),
            title=dict(
                text=r"Comparison of the $\omega_{body}$ distribution between severed and intact halteres" +
                     f"\n{'(Free flight + Dark experiments)' if use_both_light_and_dark else '(Free flight only)'}",
                font=dict(size=28)
            ),
            legend=dict(
                itemsizing='constant',
                font=dict(size=22),
                x=-0.3,  # Move legend to the left
                xanchor='left'
            )
        )

        fig.write_html("fly_omegas_display.html")

    else:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 14,  # Reduced tick label size
            "axes.titlesize": 24,  # Title size
            "axes.labelsize": 22,  # Label size
            "legend.fontsize": 20,  # Legend font size
            "xtick.labelsize": 14,  # Explicit tick label size
            "ytick.labelsize": 14,  # Explicit tick label size
        })

        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(omega_good[:, 0], omega_good[:, 1], omega_good[:, 2],
                   s=1, color='red', label=r'$\omega_{body}$ of intact flies')
        ax.scatter(omega_bad[:, 0], omega_bad[:, 1], omega_bad[:, 2],
                   s=2, color='blue', label=r'$\omega_{body}$ of severed flies')

        ax.plot([p1_bad[0], p2_bad[0]], [p1_bad[1], p2_bad[1]], [p1_bad[2], p2_bad[2]],
                color='blue', linewidth=2, label=r'Dominant $\omega_{body}$ axis')

        for q in quivers:
            ax.quiver(0, 0, 0,
                      q['x'][1], q['y'][1], q['z'][1],
                      color=q['color'], label=q['name'])

        ax.set_xlabel(r'$\omega_x$ (deg/s)', fontsize=22, labelpad=15)
        ax.set_ylabel(r'$\omega_y$ (deg/s)', fontsize=22, labelpad=15)
        ax.set_zlabel(r'$\omega_z$ (deg/s)', fontsize=22, labelpad=15)
        ax.set_aspect('equal')

        title = ax.set_title(
            r"Comparison of the $\omega_{body}$ distribution" +
            " between severed and intact halteres" +
            f"\n{'(Free flight + Dark experiments)' if use_both_light_and_dark else '(Free flight only)'}",
            pad=20,
            fontsize=24
        )

        # Adjusted legend position to the left
        ax.legend(fontsize=20, bbox_to_anchor=(-0.3, 1.0))

        plt.tight_layout()
        plt.show()


def display_omegas_plt(omega_bad, omega_good, p1_bad, p2_bad, pitch_bad, yaw_bad):
    fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(omega_good[:, 0], omega_good[:, 1], omega_good[:, 2], s=1, color='red')
    ax.scatter(omega_bad[:, 0], omega_bad[:, 1], omega_bad[:, 2], s=1, color='blue')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("wx")
    ax.set_ylabel("wy")
    ax.set_zlabel("wz")
    # plt.plot([p1_good[0], p2_good[0]], [p1_good[1], p2_good[1]], [p1_good[2], p2_good[2]], color='red')
    plt.plot([p1_bad[0], p2_bad[0]], [p1_bad[1], p2_bad[1]], [p1_bad[2], p2_bad[2]], color='blue')
    p1_body_axis = 5000 * np.array([1, 0, 0])
    p2_body_axis = 5000 * np.array([-1, 0, 0])
    # plt.plot([p1_body_axis[0], p2_body_axis[0]], [p1_body_axis[1], p2_body_axis[1]], [p1_body_axis[2], p2_body_axis[2]], color='black')
    size = 5000
    ax.quiver(0, 0, 0, size, 0, 0, color='r', label='xbody')
    ax.quiver(0, 0, 0, 0, size, 0, color='g', label='ybody')
    ax.quiver(0, 0, 0, 0, 0, size, color='orange', label='zbody')
    ax.legend()
    ax.title.set_text(f'Yaw of bad axis: {yaw_bad} and pitch is {pitch_bad}')
    ax.set_aspect('equal')
    plt.show()


def estimate_bootstrap_error(omegas):
    n_points = omegas.shape[0]
    yaw_samples = []
    pitch_samples = []
    num_bootstrap = 1000
    for _ in range(num_bootstrap):
        # Resample the point cloud with replacement
        resampled_points = omegas[np.random.choice(n_points, n_points, replace=True)]

        # Compute the principal component
        principal_component = get_first_component(resampled_points)

        # Normalize the principal component
        principal_component = principal_component / np.linalg.norm(principal_component)

        # Compute yaw and pitch
        yaw, pitch = compute_yaw_pitch(principal_component)
        yaw_samples.append(yaw)
        pitch_samples.append(pitch)
    yaw_samples, pitch_samples = np.array(yaw_samples), np.array(pitch_samples)
    pitch_std = np.std(pitch_samples)
    yaw_std = np.std(yaw_samples)
    return yaw_std, pitch_std


def estimate_monte_carlo_error(omegas):
    num_samples = 1000
    sigma_matrix = np.array([[80, 80, 80]])  # sigma found
    yaw_samples, pitch_samples = [], []
    for _ in range(num_samples):
        mean_points = omegas
        mean_shape = mean_points.shape
        random_samples = np.random.randn(*mean_shape)  # Match the shape of mean_points
        new_omegas = mean_points + random_samples * sigma_matrix
        principal_component = get_first_component(new_omegas)
        yaw, pitch = compute_yaw_pitch(principal_component)
        yaw_samples.append(yaw)
        pitch_samples.append(pitch)
    yaw_std = np.std(yaw_samples)
    pitch_std = np.std(pitch_samples)
    return yaw_std, pitch_std


def get_pca_points(omegas):
    first_component = get_first_component(omegas)
    yaw, pitch = compute_yaw_pitch(first_component)
    mean = np.mean(omegas, axis=0)
    yaw_std, pitch_std = estimate_bootstrap_error(omegas)
    # yaw_std, pitch_std = estimate_monte_carlo_error(omegas)
    return first_component, yaw, pitch, yaw_std, pitch_std


def pca_variance_explained(omegas):
    pca = PCA(n_components=3)
    pca.fit(omegas)
    return pca.explained_variance_ratio_ * 100


def get_first_component(omega):
    pca = PCA(n_components=3)
    pca.fit(omega)
    first_component = pca.components_[0]
    return first_component


def calculate_mahalanobis_distance(data):
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = np.array([mahalanobis(point, mean, inv_cov_matrix) for point in data])
    return distances


def display_omegas_dark_vs_light(csv_file):
    no_dark, with_dark = get_omegas(csv_file)
    omega_dark, wx_dark, wy_dark, wz_dark = with_dark
    omega_light, wx_light, wy_light, wz_light = no_dark
    all_omegas = np.concatenate((omega_dark, omega_light), axis=0)

    # remove outliers using mahalanobis
    mahal_dist_dark = calculate_mahalanobis_distance(omega_dark)
    omega_dark = omega_dark[mahal_dist_dark < 4]
    mahal_dist_light = calculate_mahalanobis_distance(omega_light)
    omega_light = omega_light[mahal_dist_light < 4]
    mahal_dist_all = calculate_mahalanobis_distance(all_omegas)
    all_omegas = all_omegas[mahal_dist_all < 3]

    vec_dark, yaw_dark, pitch_dark, yaw_std_dark, pitch_std_dark = get_pca_points(omega_dark)
    vec_light, yaw_light, pitch_light, yaw_std_light, pitch_std_light = get_pca_points(omega_light)
    vec_all, yaw_all, pitch_all, yaw_std_all, pitch_std_all = get_pca_points(all_omegas)

    pca = PCA(n_components=3)
    pca.fit(all_omegas)
    first_component = pca.components_[0]
    mean = pca.mean_
    p1 = mean + 10000 * first_component
    p2 = mean - 10000 * first_component

    # r = R.from_euler('y', -45, degrees=True)
    # Rot = np.array(r.as_matrix())
    # omega_dark = (Rot @ omega_dark.T).T
    # omega_light = (Rot @ omega_light.T).T

    omega_dist = np.array([all_omegas[i] @ first_component for i in range(len(all_omegas))])
    omega_light_dist = np.array([omega_light[i] @ first_component for i in range(len(omega_light))])
    # plt.hist(omega_dist, bins=100)
    # plt.hist(omega_light_dist, bins=100)
    # plt.show()

    fig = plt.figure(figsize=(20, 20))  # Adjust the figure size here
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(all_omegas[:, 0], all_omegas[:, 1], all_omegas[:, 2], s=1, color='blue')

    ax.scatter(omega_light[:, 0], omega_light[:, 1], omega_light[:, 2], s=1, color='red')
    ax.scatter(omega_dark[:, 0], omega_dark[:, 1], omega_dark[:, 2], s=1, color='blue')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("wx")
    ax.set_ylabel("wy")
    ax.set_zlabel("wz")
    # plt.plot([p1_good[0], p2_good[0]], [p1_good[1], p2_good[1]], [p1_good[2], p2_good[2]], color='red')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='blue')
    p1_body_axis = 5000 * np.array([1, 0, 0])
    p2_body_axis = 5000 * np.array([-1, 0, 0])
    plt.plot([p1_body_axis[0], p2_body_axis[0]], [p1_body_axis[1], p2_body_axis[1]], [p1_body_axis[2], p2_body_axis[2]],
             color='black')
    size = 5000
    ax.quiver(0, 0, 0, size, 0, 0, color='r', label='xbody')
    ax.quiver(0, 0, 0, 0, size, 0, color='g', label='ybody')
    ax.quiver(0, 0, 0, 0, 0, size, color='orange', label='zbody')
    ax.legend()
    ax.set_aspect('equal')
    plt.show()


def display_mosquito_omegas(intact_path, cut_paths):
    # Get omega data for intact and cut mosquitos
    omega_intact = get_all_rotations([intact_path], sample=31 * 2)
    omega_cut = get_all_rotations(cut_paths, sample=31 * 2)

    # Calculate PCA and vectors for both sets
    vec_intact, yaw_intact, pitch_intact, yaw_std_intact, pitch_std_intact = get_pca_points(omega_intact)
    vec_cut, yaw_cut, pitch_cut, yaw_std_cut, pitch_std_cut = get_pca_points(omega_cut)

    # Calculate variance explained by first principal component
    variance_intact = pca_variance_explained(omega_intact)
    variance_cut = pca_variance_explained(omega_cut)

    line_size = 3000
    # Calculate points for the principal axes
    p1_intact, p2_intact = omega_intact.mean(axis=0) + line_size * vec_intact, omega_intact.mean(
        axis=0) - line_size * vec_intact
    p1_cut, p2_cut = omega_cut.mean(axis=0) + line_size * vec_cut, omega_cut.mean(axis=0) - line_size * vec_cut

    # Define rotation matrix for 45-degree rotation around y-axis
    theta = np.radians(-45)
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # Apply rotation to all points
    omega_intact = omega_intact @ rotation_matrix.T
    omega_cut = omega_cut @ rotation_matrix.T
    p1_intact = p1_intact @ rotation_matrix.T
    p2_intact = p2_intact @ rotation_matrix.T
    p1_cut = p1_cut @ rotation_matrix.T
    p2_cut = p2_cut @ rotation_matrix.T

    # Rotate body axis quivers
    size = line_size
    quivers = [
        {'x': [0, size], 'y': [0, 0], 'z': [0, 0], 'color': 'red', 'name': 'xbody'},
        {'x': [0, 0], 'y': [0, size], 'z': [0, 0], 'color': 'green', 'name': 'ybody'},
        {'x': [0, 0], 'y': [0, 0], 'z': [0, size], 'color': 'orange', 'name': 'zbody'}
    ]

    for q in quivers:
        rotated_points = np.array([q['x'], q['y'], q['z']]).T @ rotation_matrix.T
        q['x'], q['y'], q['z'] = rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2]

    # Create the 3D plot
    fig = go.Figure()

    # Add intact mosquito data
    fig.add_trace(go.Scatter3d(
        x=omega_intact[:, 0], y=omega_intact[:, 1], z=omega_intact[:, 2],
        mode='markers',
        marker=dict(size=1, color='red'),
        name='Intact Mosquito'
    ))

    # Add cut mosquito data
    fig.add_trace(go.Scatter3d(
        x=omega_cut[:, 0], y=omega_cut[:, 1], z=omega_cut[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Cut Mosquito'
    ))

    # Add line for the cut axis
    fig.add_trace(go.Scatter3d(
        x=[p1_cut[0], p2_cut[0]], y=[p1_cut[1], p2_cut[1]], z=[p1_cut[2], p2_cut[2]],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Cut Axis'
    ))

    # Add body axis quivers
    for q in quivers:
        fig.add_trace(go.Scatter3d(
            x=q['x'], y=q['y'], z=q['z'],
            mode='lines+text',
            line=dict(color=q['color'], width=5),
            name=q['name']
        ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='wx',
            yaxis_title='wy',
            zaxis_title='wz',
            aspectmode='cube'
        ),
        title=f'Mosquito Angular Velocities - Intact vs Cut<br>Cut Mosquito - Yaw: '
              f'{yaw_cut:.2f}° (±{yaw_std_cut:.2f}°), Pitch: {pitch_cut:.2f}° '
              f'(±{pitch_std_cut:.2f}°)<br>Variance explained - Intact: PC1 {variance_intact[0]:.1f}%, '
              f'PC2 {variance_intact[1]:.1f}%, PC3 {variance_intact[2]:.1f}%<br>Variance explained - Cut: '
              f'PC1 {variance_cut[0]:.1f}%, PC2 {variance_cut[1]:.1f}%, PC3 {variance_cut[2]:.1f}%',
        legend=dict(itemsizing='constant')
    )
    # Save the figure to an HTML file
    fig.write_html("mosquito_omega_plot.html")


def calculate_single_autocorrelation(xyz_data):
    x_body, y_body, z_body = xyz_data
    return FlightAnalysis.get_auto_correlation_axis_angle(
        x_body, y_body, z_body,
        start_frame=0,
        end_frame=len(x_body)
    )


def display_mosquitoes_auto_correlation(intact_path, cut_paths):
    # Get all trajectory data
    all_intact_xyz_body = get_all_mosquitoes_xyz([intact_path])[:2]
    all_cut_xyz_body = get_all_mosquitoes_xyz(cut_paths)[:2]

    # Create a pool of workers
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)

    try:
        # Calculate autocorrelations in parallel
        all_AC_intact = pool.map(calculate_single_autocorrelation, all_intact_xyz_body)
        all_AC_cut = pool.map(calculate_single_autocorrelation, all_cut_xyz_body)
    finally:
        # Make sure to close the pool
        pool.close()
        pool.join()

    # Save the autocorrelations to an h5 file
    with h5py.File('mosquito_autocorrelations.h5', 'w') as f:
        # Create groups for intact and cut data
        intact_group = f.create_group('intact')
        cut_group = f.create_group('cut')

        # Save each intact autocorrelation as a dataset
        for i, ac in enumerate(all_AC_intact):
            intact_group.create_dataset(f'ac_{i}', data=ac)

        # Save each cut autocorrelation as a dataset
        for i, ac in enumerate(all_AC_cut):
            cut_group.create_dataset(f'ac_{i}', data=ac)

        # Add metadata about the number of samples
        f.attrs['n_intact_samples'] = len(all_AC_intact)
        f.attrs['n_cut_samples'] = len(all_AC_cut)

    # Create visualization plots
    Visualizer.create_autocorrelation_plot(
        all_AC_cut, all_AC_intact,
        "mosquitoes XYZ Autocorrelations",
        "mosquitoes XYZ_autocorrelations.html",
        cut=2000
    )
    Visualizer.plot_mean_std(
        all_AC_cut, all_AC_intact,
        "mosquitoes XYZ Autocorrelations",
        "mosquitoes_XYZ_mean_std_autocorrelations.html",
        cut=2000
    )


def get_all_mosquitoes_center_of_mass(path_dir):
    all_COM = []
    for directory in path_dir:
        for mat_file in os.listdir(directory):
            full_path = os.path.join(directory, mat_file)
            data = scipy.io.loadmat(full_path)
            COM = data['CoM']
            all_COM.append(COM)
    return all_COM


def display_mosquitoes_speed_distribution(intact_path, cut_paths):
    all_intact_COM = get_all_mosquitoes_center_of_mass([intact_path])
    all_cut_COM = get_all_mosquitoes_center_of_mass(cut_paths)

    all_cut_speeds = []
    all_intact_speeds = []
    for movie_COM in all_cut_COM:
        cut_speeds = FlightAnalysis.get_speed(movie_COM, sampling_rate=20000)
        all_cut_speeds.append(cut_speeds)

    for movie_COM in all_intact_COM:
        intact_speeds = FlightAnalysis.get_speed(movie_COM, sampling_rate=20000)
        all_intact_speeds.append(intact_speeds)

    Visualizer.display_speeds(all_speeds_cut=all_cut_speeds, all_speeds_intact=all_intact_speeds)


def get_all_rotations(path_dir, sample=62, filter_outliers=True):
    all_omegas = []
    if not isinstance(path_dir, list):
        path_dir = [path_dir]
    frame_rate = 20000

    all_xyz_body = get_all_mosquitoes_xyz(path_dir)

    for xyz in all_xyz_body:
        x_body, y_body, z_body = xyz
        _, omega_body, _, _ = FlightAnalysis.get_angular_velocities(x_body, y_body, z_body,
                                                                    start_frame=0,
                                                                    end_frame=len(x_body) - 1,
                                                                    sampling_rate=frame_rate)

        mask_not_nans = np.all(~np.isnan(omega_body), axis=1)
        omega_body = omega_body[mask_not_nans]
        omega_sampled = omega_body[::sample]
        all_omegas.append(omega_sampled)

    all_omegas = np.concatenate(all_omegas, axis=0)

    if filter_outliers:
        # Remove outliers using Mahalanobis distance
        mahal_dist = calculate_mahalanobis_distance(all_omegas)
        all_omegas = all_omegas[mahal_dist < 3]

    return all_omegas


def get_all_mosquitoes_xyz(path_dir):
    all_xyz_body = []
    for directory in path_dir:
        for mat_file in os.listdir(directory):
            full_path = os.path.join(directory, mat_file)
            data = scipy.io.loadmat(full_path)
            rotations = data['rotmats']
            # frame_rate = data['framerate'].squeeze()
            rotations = np.transpose(rotations, (2, 0, 1))

            x_body = rotations[:, :, 0]
            y_body = rotations[:, :, 1]
            z_body = rotations[:, :, 2]
            all_xyz_body.append((x_body, y_body, z_body))
    return all_xyz_body


def analyze_mosquitos_omega():
    path_intact_mosquitos = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_03_18_clean_mosquito"
    path_cut_mosquitos_1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_03_21_Mosquito_cut1"
    path_cut_mosquitos_2 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_07_08_Mosquito_cut2"

    display_mosquito_omegas(
        intact_path=path_intact_mosquitos,
        cut_paths=[path_cut_mosquitos_1, path_cut_mosquitos_2]
    )


def display_omegas_fly():
    bad_haltere = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data\all_wingbits_attributes_severed_haltere.csv"
    good_haltere = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data\all_wingbits_attributes_good_haltere.csv"
    display_good_vs_bad_haltere(good_haltere, bad_haltere, use_plotly=False, use_both_light_and_dark=False)


def analyze_mosquitoes_auto_correlation(cluster=False):
    if cluster:
        path_intact_mosquitos = r"omega analysis/mosquitos data/2021_03_18_clean_mosquito"
        path_cut_mosquitos_1 = r"omega analysis/mosquitos data/2021_03_21_Mosquito_cut1"
        path_cut_mosquitos_2 = r"omega analysis/mosquitos data/2021_07_08_Mosquito_cut2"
    else:
        path_intact_mosquitos = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_03_18_clean_mosquito"
        path_cut_mosquitos_1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_03_21_Mosquito_cut1"
        path_cut_mosquitos_2 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_07_08_Mosquito_cut2"

    display_mosquitoes_auto_correlation(
        intact_path=path_intact_mosquitos,
        cut_paths=[path_cut_mosquitos_1, path_cut_mosquitos_2]
    )


def analyze_mosquitoes_speeds_distribution():
    path_intact_mosquitos = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_03_18_clean_mosquito"
    path_cut_mosquitos_1 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_03_21_Mosquito_cut1"
    path_cut_mosquitos_2 = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\2021_07_08_Mosquito_cut2"

    display_mosquitoes_speed_distribution(
        intact_path=path_intact_mosquitos,
        cut_paths=[path_cut_mosquitos_1, path_cut_mosquitos_2]
    )


def get_mosquitoes_autocorrelations_pval():
    # load all mosquitos data from h5 file
    h5_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\visualizations\omega analysis\mosquitos data\mosquito_autocorrelations.h5"
    with h5py.File(h5_path, 'r') as f:
        n_intact_samples = f.attrs['n_intact_samples']
        n_cut_samples = f.attrs['n_cut_samples']
        all_AC_intact = [f['intact'][f'ac_{i}'][:] for i in range(n_intact_samples)]
        all_AC_cut = [f['cut'][f'ac_{i}'][:] for i in range(n_cut_samples)]
    # for each autocorrelation value (not movie but the index itself), calculate the p-value across the two groups
    # use the longest autocorrelation array to determine the number of values to calculate the p-value for
    pvals = []
    max_T = max(len(ac) for ac in all_AC_intact + all_AC_cut)
    for T in range(max_T // 4):
        # get the values for each group
        values_intact = [ac[T] for ac in all_AC_intact if T < len(ac)]
        values_cut = [ac[T] for ac in all_AC_cut if T < len(ac)]
        # calculate the p-value
        _, pval = scipy.stats.ttest_ind(values_intact, values_cut, equal_var=False)
        pvals.append(pval)
    plt.plot(pvals)
    # plt.yscale('log')
    plt.show()


def calculate_rotation_angles(xb, yb, zb, look_back):
    """Calculate rotation angles for a single trajectory."""
    not_nans = ~np.isnan(yb).any(axis=1)
    xb, yb, zb = xb[not_nans], yb[not_nans], zb[not_nans]
    Rs = np.stack([xb, yb, zb], axis=-1)
    all_deviations = []

    for j in range(look_back, Rs.shape[0] - 1):
        R_current = Rs[j]
        R_past = Rs[j - look_back]
        relative_rotation = R_current @ R_past.T
        r = Rotation.from_matrix(relative_rotation)
        axis_angle = r.as_rotvec()
        deviation_rad = np.linalg.norm(axis_angle)
        deviation = np.degrees(deviation_rad)
        all_deviations.append(deviation)

    return all_deviations


def process_path(path, look_back):
    """Process a single path and return all deviations."""
    x_bodies = Visualizer.load_all_attributes_from_h5(path, attribute='x_body')
    y_bodies = Visualizer.load_all_attributes_from_h5(path, attribute='y_body')
    z_bodies = Visualizer.load_all_attributes_from_h5(path, attribute='z_body')

    all_deviations = []
    for xb, yb, zb in zip(x_bodies, y_bodies, z_bodies):
        deviations = calculate_rotation_angles(xb, yb, zb, look_back)
        all_deviations.extend(deviations)

    return all_deviations


def plot_single_analysis(deviations_by_path, look_back):
    """Plot histograms for single look-back analysis."""
    all_deviations_global = [dev for devs, _ in deviations_by_path for dev in devs]
    global_min = min(all_deviations_global)
    global_max = max(all_deviations_global)
    bins = np.linspace(global_min, global_max, 101)

    for all_deviations, i in deviations_by_path:
        mean_dev = np.mean(all_deviations)
        median = np.median(all_deviations)
        std_dev = np.std(all_deviations)

        plt.hist(
            all_deviations,
            bins=bins,
            alpha=0.5,
            density=True,
            label=f"{'Cut' if i == 0 else 'Not Cut'} -> Mean: {mean_dev:.2f}, Std: {std_dev:.2f}, median: {median:.2f} [deg]"
        )

    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.title(f"Distribution of Rotation Angles Relative to {look_back} Frames Earlier")
    plt.legend()
    plt.show()


def plot_multiple_analysis(results, look_back_values):
    """Plot results for multiple look-back analysis with standard error bands."""
    plt.rcParams.update({'font.size': 14})  # Increase base font size

    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert frames to milliseconds
    ms_values = [frame / 16 for frame in look_back_values]  # 16000 fps = 16 frames/ms

    # Plot means with standard error bands
    for condition, color in [('cut', 'red'), ('not_cut', 'blue')]:  # Changed colors here
        means = results[condition]['means']
        stderr = results[condition]['standard errors']

        ax.plot(ms_values, means, label='Flies with severed halteres' if condition == 'cut' else 'Intact flies',
                color=color)
        ax.fill_between(ms_values,
                        [m - se for m, se in zip(means, stderr)],
                        [m + se for m, se in zip(means, stderr)],
                        alpha=0.2, color=color)

    ax.set_xlabel('Time [ms]', fontsize=16)
    ax.set_ylabel('Mean Rotation Angle [degrees]', fontsize=16)
    ax.set_title('Mean Rotation Angle vs Time Step', fontsize=18)
    ax.legend(loc='upper left', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    ax.margins(x=0)
    plt.tight_layout()
    plt.show()

def save_analysis_results(results, look_back_values, filename='analysis_results.pkl'):
    """Save analysis results and look_back_values to a pickle file."""
    data = {
        'results': results,
        'look_back_values': look_back_values
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Results saved to {filename}")

def load_analysis_results(filename='analysis_results.pkl'):
    """Load analysis results and look_back_values from a pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['look_back_values']

def visualize_saved_results(filename='analysis_results.pkl'):
    """Load and visualize previously saved analysis results."""
    results, look_back_values = load_analysis_results(filename)
    plot_multiple_analysis(results, look_back_values)

def analyze_flight_freakness(analysis_type='single', look_back_range=None, save_results=True):
    """Analyze flight patterns from video data with option to save results."""
    path_cut = (r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024"
                r" undisturbed\moved from cluster\free 24-1 movies")
    path_not_cut = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight"
    paths = [path_cut, path_not_cut]

    if analysis_type == 'single':
        look_back = 280 if look_back_range is None else look_back_range[0]
        deviations_by_path = []

        for i, path in enumerate(paths):
            all_deviations = process_path(path, look_back)
            deviations_by_path.append((all_deviations, i))

        plot_single_analysis(deviations_by_path, look_back)

    elif analysis_type == 'multiple':
        if look_back_range is None:
            look_back_range = (1, 301)  # Default range from 1 to 300

        results = {
            'cut': {'means': [], 'standard errors': []},
            'not_cut': {'means': [], 'standard errors': []}
        }
        look_back_values = range(look_back_range[0], look_back_range[1])

        for look_back in look_back_values:
            for i, path in enumerate(paths):
                all_deviations = process_path(path, look_back)

                mean_dev = np.mean(all_deviations)
                std_dev = np.std(all_deviations)
                standard_error = std_dev  # / np.sqrt(len(all_deviations))

                if i == 0:  # Cut case
                    results['cut']['means'].append(mean_dev)
                    results['cut']['standard errors'].append(standard_error)
                else:  # Not cut case
                    results['not_cut']['means'].append(mean_dev)
                    results['not_cut']['standard errors'].append(standard_error)

        if save_results:
            save_analysis_results(results, look_back_values)

        plot_multiple_analysis(results, look_back_values)


def display_all_wingbit_frequencies_per_movie():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import h5py
    import os
    import plotly.express as px
    import colorsys

    def generate_distinct_colors(n):
        """Generate n distinct colors using HSV color space"""
        colors = []
        for i in range(n):
            # Use golden ratio to space hues evenly
            hue = i * 0.618033988749895 % 1
            # Vary saturation and value to make colors more distinct
            saturation = 0.6 + (i % 3) * 0.1  # Vary between 0.6 and 0.8
            value = 0.9 - (i % 3) * 0.1  # Vary between 0.7 and 0.9

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to hex color
            color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(color)
        return colors

    path_not_cut = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight"
    movie_frequencies = {}  # {movie_name: [frequencies]}

    # Collect data
    attribute = 'right_full_wingbits'
    print("Processing Not Cut data")
    for dirpath, dirnames, _ in os.walk(path_not_cut):
        for dirname in dirnames:
            if dirname.startswith('mov'):
                h5_path = os.path.join(dirpath, dirname, f"{dirname}_analysis_smoothed.h5")
                if os.path.isfile(h5_path):
                    print(f"Processing file: {h5_path}")
                    with h5py.File(h5_path, "r") as h5_file:
                        if attribute in h5_file:
                            frequencies_this_movie = []
                            for group_name in h5_file[attribute]:
                                group = h5_file[attribute][group_name]
                                if "start" in group and "end" in group:
                                    start = group["start"][()]
                                    end = group["end"][()]
                                    length = end - start
                                    if length > 0:
                                        frequency = 16000 / length
                                        if frequency < 300:
                                            frequencies_this_movie.append(frequency)
                                        else:
                                            print(f"Skipped frequency > 300: {frequency}")
                                    else:
                                        print(f"Invalid length: {length}")
                            if frequencies_this_movie:
                                movie_frequencies[dirname] = frequencies_this_movie
                        else:
                            print(f"Attribute '{attribute}' not found in {h5_path}")

    # Setup bins
    bins = np.linspace(180, 290, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # Calculate histogram data for each movie
    fig = go.Figure()

    # Generate unique colors for each movie
    num_movies = len(movie_frequencies)
    colors = generate_distinct_colors(num_movies)

    # Calculate total frequencies for sorting
    movie_totals = {movie: len(freqs) for movie, freqs in movie_frequencies.items()}
    sorted_movies = sorted(movie_totals.items(), key=lambda x: x[1], reverse=True)

    # Initialize bottom values for stacking
    bottom = np.zeros(len(bins) - 1)

    # Create stacked bars for each movie
    for (movie, _), color in zip(sorted_movies, colors):
        freqs = movie_frequencies[movie]
        hist, _ = np.histogram(freqs, bins=bins)

        # Convert to density
        hist = hist / (len(freqs) * bin_width)

        # Extract movie number for clearer labeling
        movie_num = int(movie.replace('mov', ''))

        fig.add_trace(go.Bar(
            name=f'Movie {movie_num}',
            x=bin_centers,
            y=hist,
            offset=bottom,
            width=bin_width,
            marker_color=color,
            hovertemplate="<br>".join([
                "Movie %{fullData.name}",
                "Frequency: %{x:.1f} Hz",
                "Density: %{y:.3f}",
                "Contribution: %{customdata:.1f}%",
                "<extra></extra>"
            ]),
            # Add percentage contribution to hover data
            customdata=(hist / sum(hist) * 100 if sum(hist) > 0 else np.zeros_like(hist))
        ))

        bottom += hist

    # Update layout
    fig.update_layout(
        title={
            'text': "Wingbit Frequencies Distribution (Not Cut)",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Frequency (Hz)",
        yaxis_title="Density",
        barmode='stack',
        showlegend=True,
        legend_title="Movies",
        hovermode='closest',
        # Move legend to the right of the plot
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent white background
            bordercolor='rgba(0, 0, 0, 0.3)',  # Light border
            borderwidth=1
        ),
        # Add some margin on the right for the legend
        margin=dict(r=150, t=100, b=50, l=50)
    )

    # Add mean and std annotations for total distribution
    all_freqs = [freq for freqs in movie_frequencies.values() for freq in freqs]
    mean_freq = np.mean(all_freqs)
    std_freq = np.std(all_freqs)

    fig.add_annotation(
        text=f"Mean: {mean_freq:.2f} Hz<br>Std: {std_freq:.2f} Hz",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.3)',
        borderwidth=1
    )

    fig.show()
    # Save as HTML file
    output_path = "wingbit_frequencies_by_movie.html"
    fig.write_html(output_path)
    print(f"Figure saved as: {output_path}")


def display_all_wingbit_frequencies():
    path_cut = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies"
    path_not_cut = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight"
    paths = [path_cut, path_not_cut]
    labels = ["Cut", "Not Cut"]

    all_frequencies_data = []

    for path, label in zip(paths, labels):
        all_frequencies = []
        attribute = 'right_full_wingbits'
        print(f"Processing: {label}")
        for dirpath, dirnames, _ in os.walk(path):
            for dirname in dirnames:
                if dirname.startswith('mov'):
                    h5_path = os.path.join(dirpath, dirname, f"{dirname}_analysis_smoothed.h5")
                    if os.path.isfile(h5_path):
                        print(f"Processing file: {h5_path}")
                        with h5py.File(h5_path, "r") as h5_file:
                            if attribute in h5_file:
                                for group_name in h5_file[attribute]:
                                    group = h5_file[attribute][group_name]
                                    if "start" in group and "end" in group:
                                        start = group["start"][()]
                                        end = group["end"][()]
                                        length = end - start
                                        if length > 0:  # Avoid division by zero
                                            frequency = 16000 / length
                                            if frequency < 300:  # Frequency threshold
                                                all_frequencies.append(frequency)
                                            else:
                                                print(f"Skipped frequency > 300: {frequency}")
                                        else:
                                            print(f"Invalid length: {length}")
                            else:
                                print(f"Attribute '{attribute}' not found in {h5_path}")
        if all_frequencies:
            print(f"Found {len(all_frequencies)} frequencies for {label}")
        else:
            print(f"No frequencies found for {label}")
        all_frequencies_data.append(all_frequencies)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(180, 290, 20)
    for frequencies, label in zip(all_frequencies_data, labels):
        plt.hist(frequencies, bins=bins, density=True, alpha=0.6, label=label)
    stats = [
        (label, np.mean(freqs) if freqs else 0, np.std(freqs) if freqs else 0)
        for freqs, label in zip(all_frequencies_data, labels)
    ]
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Density")
    title = (
        f"Density Histogram of Wingbit Frequencies\n"
        f"Cut (Mean: {stats[0][1]:.2f} Hz, Std: {stats[0][2]:.2f} Hz), "
        f"Not Cut (Mean: {stats[1][1]:.2f} Hz, Std: {stats[1][2]:.2f} Hz)"
    )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_wingbit_vs_speed():
    path_cut = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\moved from cluster\free 24-1 movies"
    path_not_cut = r"G:\My Drive\Amitai\one halter experiments\sagiv free flight"
    paths = [path_cut, path_not_cut]
    labels = ["Cut", "Not Cut"]

    speeds_data = []
    frequencies_data = []
    labels_data = []

    for path, label in zip(paths, labels):
        print(f"Processing: {label}")
        for dirpath, dirnames, _ in os.walk(path):
            for dirname in dirnames:
                if dirname.startswith('mov'):
                    h5_path = os.path.join(dirpath, dirname, f"{dirname}_analysis_smoothed.h5")
                    if os.path.isfile(h5_path):
                        print(f"Processing file: {h5_path}")
                        with h5py.File(h5_path, "r") as h5_file:
                            # Get mean speed
                            if 'CM_speed' in h5_file:
                                mean_speed = np.nanmean(h5_file['CM_speed'][:])
                            else:
                                print(f"CM_speed not found in {h5_path}")
                                continue

                            # Get mean frequency
                            if 'right_full_wingbits' in h5_file:
                                movie_frequencies = []
                                for group_name in h5_file['right_full_wingbits']:
                                    group = h5_file['right_full_wingbits'][group_name]
                                    if "start" in group and "end" in group:
                                        start = group["start"][()]
                                        end = group["end"][()]
                                        length = end - start
                                        if length > 0:
                                            frequency = 16000 / length
                                            if frequency < 300:
                                                movie_frequencies.append(frequency)

                                if movie_frequencies:
                                    mean_frequency = np.mean(movie_frequencies)
                                    speeds_data.append(mean_speed)
                                    frequencies_data.append(mean_frequency)
                                    labels_data.append(label)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    colors = {'Cut': 'red', 'Not Cut': 'blue'}
    for label in colors:
        mask = [l == label for l in labels_data]
        plt.scatter(
            [speeds_data[i] for i in range(len(mask)) if mask[i]],
            [frequencies_data[i] for i in range(len(mask)) if mask[i]],
            c=colors[label],
            label=label,
            alpha=0.6
        )

    plt.xlabel("Ground Speed (m/s)")
    plt.ylabel("Wing-bit Frequency (Hz)")
    plt.title("Wing-bit Frequency vs Ground Speed")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print statistics
    for label in ['Cut', 'Not Cut']:
        mask = [l == label for l in labels_data]
        speeds = [speeds_data[i] for i in range(len(mask)) if mask[i]]
        freqs = [frequencies_data[i] for i in range(len(mask)) if mask[i]]
        print(f"\n{label} statistics:")
        print(f"Number of movies: {len(speeds)}")
        print(f"Mean speed: {np.mean(speeds):.2f} ± {np.std(speeds):.2f}")
        print(f"Mean frequency: {np.mean(freqs):.2f} ± {np.std(freqs):.2f} Hz")


if __name__ == '__main__':
    # plot_wingbit_vs_speed()
    # display_all_wingbit_frequencies()
    # analyze_flight_freakness(analysis_type='multiple', look_back_range=(1, 501))
    # get_mosquitoes_autocorrelations_pval()
    # analyze_mosquitoes_auto_correlation(cluster=False)
    # analyze_mosquitos_omega()
    display_omegas_fly()

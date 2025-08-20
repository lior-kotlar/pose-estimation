
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from extract_flight_data import FlightAnalysis
from analyze_omegas import get_3D_attribute_from_df

SAMPLING_RATE = 16000
dt = 1 / 16000
LEFT = 0
RIGHT = 1
NUM_TIPS_EACH_SIZE_Y_BODY = 10
WINGS_JOINTS_INDS = [7, 15]
WING_TIP_IND = 2
UPPER_PLANE_POINTS = [0, 1, 2, 3]
LOWER_PLANE_POINTS = [3, 4, 5, 6]


def test_forces():
    phi = 90
    theta = 0
    psi = 0
    phi_dot = 0
    theta_dot = -100
    psi_dot = 0
    yaw = 0
    pitch = 45
    roll = 0
    cm_dot = np.array([0, 0, 0])
    omega_body = np.array([0, 0, 0])

    angles_dot = np.radians([psi_dot, -theta_dot, phi_dot])
    # left
    f_body_aero_left, f_lab_aero_left, t_body_left, r_wing2sp_left, r_sp2body_left, r_body2lab_left = FlightAnalysis.exctract_forces(
        phi=phi, phi_dot=phi_dot, pitch=-pitch, psi=psi, psi_dot=psi_dot, roll=roll, theta=-theta, theta_dot=-theta_dot,
        yaw=yaw, center_mass_dot=cm_dot, omega_body=omega_body)

    Rot_mat_wing2labL = r_body2lab_left @ r_sp2body_left @ r_wing2sp_left

    # right
    f_body_aero_right, f_lab_aero_right, t_body_right, r_wing2sp_right, r_sp2body_right, r_body2lab_right = FlightAnalysis.exctract_forces(
                                                                     -phi, -phi_dot, -pitch, 180 - psi, -psi_dot, roll, -theta,
                                                                     -theta_dot, yaw , cm_dot, omega_body)
    Rot_mat_wing2labR = r_body2lab_right @ r_sp2body_right @ r_wing2sp_right

    dispaly_coordinate_systems(Rot_mat_wing2labL, Rot_mat_wing2labR, r_body2lab_left, f_lab_aero_left, f_lab_aero_right)

    torque_total = (t_body_left + t_body_right) / np.linalg.norm(t_body_left + t_body_right)
    force_body_total = (f_body_aero_left + f_body_aero_right) / np.linalg.norm(f_body_aero_left + f_body_aero_right)


def dispaly_coordinate_systems(Rot_mat_wing2labL, Rot_mat_wing2labR, r_body2lab_left, f_body_left, f_body_right, ax=None):
    amplify = 1e5
    f_body_left, f_body_right = amplify*f_body_left, amplify*f_body_right
    # Wing axes
    wing_Xax = np.dot(Rot_mat_wing2labL, np.array([1, 0, 0]))
    wing_Yax = np.dot(Rot_mat_wing2labL, np.array([0, 1, 0]))
    wing_Zax = np.dot(Rot_mat_wing2labL, np.array([0, 0, 1]))
    # Right wing axes
    wing_Xaxr = np.dot(Rot_mat_wing2labR, np.array([1, 0, 0]))
    wing_Yaxr = np.dot(Rot_mat_wing2labR, np.array([0, 1, 0]))
    wing_Zaxr = np.dot(Rot_mat_wing2labR, np.array([0, 0, 1]))
    # Body axes
    body_Xax = np.dot(r_body2lab_left, np.array([1, 0, 0]))
    body_Yax = np.dot(r_body2lab_left, np.array([0, 1, 0]))
    body_Zax = np.dot(r_body2lab_left, np.array([0, 0, 1]))
    # Plotting
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    wing_ax = r_body2lab_left @ np.array([0, 1, 0])
    # Quivers for wing axes
    ax.quiver(wing_ax[0], wing_ax[1], wing_ax[2], wing_Xax[0], wing_Xax[1], wing_Xax[2], color='k')
    ax.quiver(wing_ax[0], wing_ax[1], wing_ax[2], wing_Yax[0], wing_Yax[1], wing_Yax[2], color='r')
    ax.quiver(wing_ax[0], wing_ax[1], wing_ax[2], wing_Zax[0], wing_Zax[1], wing_Zax[2], color='b')
    # Quiver for left force
    ax.quiver(wing_ax[0], wing_ax[1], wing_ax[2], f_body_left[0], f_body_left[1], f_body_left[2], color='g')

    wing_ax = r_body2lab_left @ np.array([0, -1, 0])
    # Quivers for right wing axes
    ax.quiver(wing_ax[0], wing_ax[1], wing_ax[2], wing_Xaxr[0], wing_Xaxr[1], wing_Xaxr[2], color='k')
    ax.quiver(wing_ax[0], wing_ax[1], wing_ax[2], wing_Yaxr[0], wing_Yaxr[1], wing_Yaxr[2], color='r')
    ax.quiver(wing_ax[0], wing_ax[1], wing_ax[2], wing_Zaxr[0], wing_Zaxr[1], wing_Zaxr[2], color='b')
    # Quiver for left force
    ax.quiver(wing_ax[0], wing_ax[1], wing_ax[2], f_body_right[0], f_body_right[1], f_body_right[2], color='g')
    # Quivers for body axes
    ax.quiver(0, 0, 0, body_Xax[0], body_Xax[1], body_Xax[2], color='k')
    ax.quiver(0, 0, 0, body_Yax[0], body_Yax[1], body_Yax[2], color='r')
    ax.quiver(0, 0, 0, body_Zax[0], body_Zax[1], body_Zax[2], color='b')

    # Set axis properties
    # Set boundaries
    ax.set_xlim([-2, 2])  # Set x-axis boundaries
    ax.set_ylim([-2, 2])  # Set y-axis boundaries
    ax.set_zlim([-2, 2])  # Set z-axis boundaries
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')  # Equal aspect ratio
    plt.show()


def analyze_torque():
    csv_path_severed = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data\all_wingbits_attributes_severed_haltere.csv"
    csv_path_intact = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\wingbits data\all_wingbits_attributes_good_haltere.csv"
    df_severed = pd.read_csv(csv_path_severed)
    df_intact = pd.read_csv(csv_path_intact)
    omegas_severed, _, _, _ = get_3D_attribute_from_df(df_severed)
    torques_severed, _, _, _ = get_3D_attribute_from_df(df_severed, attirbutes=["torque_body_x_take", "torque_body_y_take", "torque_body_z_take"])
    omegas_intact, _, _, _ = get_3D_attribute_from_df(df_intact)
    torques_intact, _, _, _ = get_3D_attribute_from_df(df_intact, attirbutes=["torque_body_x_take", "torque_body_y_take", "torque_body_z_take"])
    # omegas_norm = omegas_severed / np.linalg.norm(omegas_severed, axis=1)[:, np.newaxis]
    # torques_norm = torques_severed / np.linalg.norm(torques_severed, axis=1)[:, np.newaxis]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(omegas_severed[:, 0], omegas_severed[:, 1], omegas_severed[:, 2], s=2, c="blue")
    torques_severed *= 1000000000
    ax.scatter(torques_severed[:, 0], torques_severed[:, 1], torques_severed[:, 2], s=2, c="red")


    torques_intact *= 1000000000
    ax.scatter(torques_intact[:, 0], torques_intact[:, 1], torques_intact[:, 2], s=2, c="blue")

    ax.set_aspect('equal')
    plt.show()
    pass
# import tensorflow as tf
import torch
import numpy as np


def torch_find_peaks_argmax(x):
    """ Finds the maximum value in each channel and returns the location and value.
    Args:
        x: rank-4 tensor (samples, height, width, channels)

    Returns:
        peaks: rank-3 tensor (samples, [x, y, val], channels)
    """

    x = torch.from_numpy(x)
    # formatting the input to match the channel ordering
    x = x.to(memory_format=torch.channels_last)

    in_shape = x.size()

    image_size = int(in_shape[1])

    # Flatten height/width dims
    flattened = torch.reshape(x, [in_shape[0], -1, in_shape[-1]])

    # Find peaks in linear indices
    idx = torch.argmax(flattened, axis=1)

    # Convert linear indices to subscripts
    rows = torch.floor_divide(idx.type(torch.int32), in_shape[1])
    cols = torch.floor_divide(idx.type(torch.int32), in_shape[1])
    # rows = torch.floor_divide(tf.cast(idx, tf.int32), in_shape[1])
    # cols = torch.floor_divide(tf.cast(idx, tf.int32), in_shape[1])

    # Dumb way to get actual values without indexing
    vals = torch.amax(flattened, 1)
    vals = vals.type(torch.float32)

    pred = torch.stack([cols.type(torch.float32), rows.type(torch.float32), vals], 1)
    # Return N x 3 x C tensor

    pred = np.transpose(pred, (0, 2, 1))
    pred = pred[..., :2]
    # pred = pred / image_size  # normalize points

    return pred



# def tf_find_peaks_argmax(x):
#     """ Finds the maximum value in each channel and returns the location and value.
#     Args:
#         x: rank-4 tensor (samples, height, width, channels)
#
#     Returns:
#         peaks: rank-3 tensor (samples, [x, y, val], channels)
#     """
#
#     # Store input shape
#     in_shape = tf.shape(x)
#
#     image_size = int(in_shape[1])
#
#     # Flatten height/width dims
#     flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])
#
#     # Find peaks in linear indices
#     idx = tf.argmax(flattened, axis=1)
#
#     # Convert linear indices to subscripts
#     rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
#     cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])
#
#     # Dumb way to get actual values without indexing
#     vals = tf.math.reduce_max(flattened, axis=1)
#     vals = tf.cast(vals, tf.float32)
#     # Return N x 3 x C tensor
#     pred = tf.stack([
#         tf.cast(cols, tf.float32),
#         tf.cast(rows, tf.float32),
#         vals
#     ], axis=1)
#
#     pred = np.transpose(pred, (0, 2, 1))
#     # pred = pred[..., :2]
#     # pred = pred / image_size  # normalize points
#
#     return pred


def find_peaks_soft_argmax(x):

    heatmap = torch.from_numpy(x).float()

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

    # Clamp centroid coordinates to ensure they are within image boundaries
    centroid_x = torch.clamp(centroid_x, 0, width - 1)
    centroid_y = torch.clamp(centroid_y, 0, height - 1)

    # Combine the coordinates
    centroids = torch.stack([centroid_x, centroid_y], dim=-1)

    return np.array(centroids)



def RQ3(A):
    """
    RQ decomposition of 3x3 matrix

    :param A: 3x3 matrix
    :returns: R - Upper triangular 3x3 matrix, Q - 3x3 orthonormal rotation matrix
    """
    if A.shape != (3, 3):
        raise ValueError('A must be a 3x3 matrix')

    eps = 1e-10

    # Find rotation Qx to set A[2,1] to 0
    A[2, 2] = A[2, 2] + eps
    c = -A[2, 2] / np.sqrt(A[2, 2] ** 2 + A[2, 1] ** 2)
    s = A[2, 1] / np.sqrt(A[2, 2] ** 2 + A[2, 1] ** 2)
    Qx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    R = A @ Qx

    # Find rotation Qy to set A[2,0] to 0
    R[2, 2] = R[2, 2] + eps
    c = R[2, 2] / np.sqrt(R[2, 2] ** 2 + R[2, 0] ** 2)
    s = R[2, 0] / np.sqrt(R[2, 2] ** 2 + R[2, 0] ** 2)
    Qy = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    R = R @ Qy

    # Find rotation Qz to set A[1,0] to 0
    R[1, 1] = R[1, 1] + eps
    c = -R[1, 1] / np.sqrt(R[1, 1] ** 2 + R[1, 0] ** 2)
    s = R[1, 0] / np.sqrt(R[1, 1] ** 2 + R[1, 0] ** 2)
    Qz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = R @ Qz

    Q = Qz.T @ Qy.T @ Qx.T

    # Adjust R and Q so that the diagonal elements of R are +ve
    for n in range(3):
        if R[n, n] < 0:
            R[:, n] = -R[:, n]
            Q[n, :] = -Q[n, :]

    return R, Q


def DecomposeCamera(P):
    """
    Decomposes a camera projection matrix

    :param P: 3x4 camera projection matrix
    :returns: K, Rc_w, Pc, pp, pv
    """
    p1 = P[:, 0]
    p2 = P[:, 1]
    p3 = P[:, 2]
    p4 = P[:, 3]

    M = P[:, :3]
    m3 = M[2, :].T

    # Camera centre, analytic solution
    X = np.linalg.det(np.column_stack((p2, p3, p4)))
    Y = -np.linalg.det(np.column_stack((p1, p3, p4)))
    Z = np.linalg.det(np.column_stack((p1, p2, p4)))
    T = -np.linalg.det(M)

    Pc = np.array([X, Y, Z, T])
    Pc = Pc / Pc[3]
    Pc = Pc[:3]  # Make inhomogeneous

    # Principal point
    pp = M @ m3
    pp = pp / pp[2]
    pp = pp[:2]  # Make inhomogeneous

    # Principal vector pointing out of camera
    pv = np.linalg.det(M) * m3
    pv = pv / np.linalg.norm(pv)

    # Perform RQ decomposition of M matrix
    K, Rc_w = RQ3(M)

    # Check if Rc_w is right-handed
    if np.dot(np.cross(Rc_w[:, 0], Rc_w[:, 1]), Rc_w[:, 2]) < 0:
        print('Note that rotation matrix is left handed')

    return K, Rc_w, Pc, pp, pv

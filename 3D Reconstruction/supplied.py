import numpy as np
from scipy.linalg import expm


def triangulation_test_case(noise):
    """
    Generates a synthetic triangulation problem with noisy 2D observations.

    noise: Standard deviation of the Gaussian noise added to the image points

    Returns:
    - Ps: List of two 3x4 projection matrices
    - us: 2x2 numpy array of noisy 2D image points (one column per camera)
    - U_true: 3x1 numpy array representing the true 3D point
    """
    
    # Ground truth 3D point
    U_true = np.random.rand(3, 1)

    # Camera centers
    C1 = np.random.rand(3, 1) + np.array([[0], [0], [-4]])
    C2 = np.random.rand(3, 1) + np.array([[0], [0], [-4]])

    # Intrinsic matrix
    K = np.diag([2000, 2000, 1])

    # Projection matrices
    P1 = K @ small_rotation(0.1) @ np.hstack((np.eye(3), -C1))
    P2 = K @ small_rotation(0.1) @ np.hstack((np.eye(3), -C2))

    Ps = [P1, P2]

    # Project U_true into image coordinates
    us1 = P1 @ np.vstack((U_true, [[1]]))
    us2 = P2 @ np.vstack((U_true, [[1]]))

    # Normalize and add noise
    us1 = us1 / us1[2] + noise * np.random.randn(3, 1)
    us2 = us2 / us2[2] + noise * np.random.randn(3, 1)

    # Remove the third row (homogeneous coordinate)
    us = np.hstack((us1[:2], us2[:2]))

    return Ps, us, U_true



def small_rotation(max_angle):
    """
    Generates a small random 3D rotation matrix.

    max_angle: Maximum rotation angle in radians

    Returns:
    - R: 3x3 rotation matrix representing a random rotation
    """
    
    # Generate a random axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    # Random angle scaled by max_angle
    angle = max_angle * np.random.rand()

    # Axis-angle vector
    s = axis * angle

    # Skew-symmetric matrix
    skew_s = np.array([
        [0, -s[2], s[1]],
        [s[2], 0, -s[0]],
        [-s[1], s[0], 0]
    ])

    # Rotation matrix
    R = expm(skew_s)

    return R


def clean_for_plot(Us):
    """
    Removes the 1% most extreme values in each dimension.
    
    Us: 3xN numpy array with the points to clean
    Returns:
    - Us_clean: Cleaned points, 3xM array with M < N
    - removed_indices: Indices of the removed points, boolean array
    """
    # Compute the 1st and 99th percentiles along each dimension (row)
    minvals = np.percentile(Us, 1, axis=1)
    maxvals = np.percentile(Us, 99, axis=1)

    # Find indices of points outside the defined percentiles
    removed_indices = (Us[0, :] > maxvals[0]) | (Us[0, :] < minvals[0])
    for kk in range(1, 3):
        removed_indices |= (Us[kk, :] > maxvals[kk]) | (Us[kk, :] < minvals[kk])

    # Filter the points to keep only the ones not removed
    Us_clean = Us[:, ~removed_indices]

    return Us_clean, removed_indices


def equal_aspect_ratio(ax):
    """
    Sets equal aspect ratio in a 3D matplotlib plot by adjusting axis limits.

    ax: A matplotlib 3D axis (Axes3D)

    Returns:
    - None (modifies the axis limits in-place)
    """

    # Note: ax.set_box_aspect([1, 1, 1]) does not actually adjust axis *limits*,
    # so the plot may still look distorted if x/y/z ranges differ.
    # This function ensures a cubic data volume by explicitly setting equal limits.
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim([z_middle - max_range/2, z_middle + max_range/2])
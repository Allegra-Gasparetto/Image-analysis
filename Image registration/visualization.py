import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import *

def plot_matches(
        source: Union[cv2.Mat, np.ndarray],
        target: Union[cv2.Mat, np.ndarray], 
        pts_src: Sequence[cv2.KeyPoint], 
        pts_tgt: Sequence[cv2.KeyPoint], 
        matches: Sequence[Sequence[cv2.DMatch]], 
        max_ratio: float = 0.8,
        ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the matches between two images.
    
    Parameters:
    -----------
    source (cv2.Mat | np.ndarray): The source image.
    target (cv2.Mat | np.ndarray): The target image.
    pts_src (Sequence[cv2.KeyPoint]): Keypoints from the source image.
    pts_tgt (Sequence[cv2.KeyPoint]): Keypoints from the target image.
    matches (Sequence[Sequence[cv2.DMatch]]): All matches between the keypoints.
    max_ratio (float): The maximum ratio for Lowe's ratio test.

    Returns:
    --------
    tuple: A tuple containing the figure and axes of the plot.
    
    """

    match_mask = [[0, 0] for i in range(len(matches))]
    for i, (m,n) in enumerate(matches):
        if m.distance < max_ratio * n.distance:
            match_mask[i] = [1, 0]
    
    plot_kwargs = {
        'matchColor': (0, 255, 0, 0.2),
        'singlePointColor': (255, 0, 0),
        'matchesMask': match_mask,
        'flags': cv2.DrawMatchesFlags_DEFAULT
    }
    img_to_plot = cv2.drawMatchesKnn(source, pts_src, target, pts_tgt, matches, None, **plot_kwargs)

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_to_plot)
    ax.axis('off')
    ax.set_title('Matches')

    return f, ax

def plot_affine_test_case_outlier(
        pts: np.ndarray,
        pts_tilde: np.ndarray,
        image_width: int,
        image_height: int,
        outlier_idxs: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the affine test case with outliers.
    
    Parameters:
    -----------
    pts (np.ndarray): The source points.
    pts_tilde (np.ndarray): The target points.
    outlier_idxs (np.ndarray): The indices of the outliers.

    Returns:
    --------
    tuple: A tuple containing the figure and axes of the plot.
    
    """
    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Plot the original points
    inlier_pts = pts[:, ~outlier_idxs]
    ax.scatter(inlier_pts[0,:], inlier_pts[1,:], label='Original inliers', color='g', marker='o', facecolor='none')

    outlier_pts = pts[:, outlier_idxs]
    ax.scatter(outlier_pts[0,:], outlier_pts[1,:], label='Original outliers', color='r', marker='o', facecolor='none')

    # Plot the transformed points
    inlier_pts_tilde = pts_tilde[:, ~outlier_idxs]
    outlier_pts_tilde = pts_tilde[:, outlier_idxs]
    ax.scatter(inlier_pts_tilde[0,:], inlier_pts_tilde[1,:], label='Transformed inliers', color='g')
    ax.scatter(outlier_pts_tilde[0,:], outlier_pts_tilde[1,:], label='Transformed outliers', color='r')

    # Plot line between inlier and transformed points
    for i in range(pts.shape[1]):

        # Check if the point is an outlier
        outlier = outlier_idxs[i]
        if outlier:

            ax.quiver(pts[0,i], pts[1,i], 
                    pts_tilde[0,i] - pts[0,i], 
                    pts_tilde[1,i] - pts[1,i], 
                    angles='xy', 
                    scale_units='xy', 
                    scale=1, 
                    color='r', 
                    alpha=0.8, 
                    )
        else:

            ax.quiver(pts[0,i], pts[1,i], 
                    pts_tilde[0,i] - pts[0,i], 
                    pts_tilde[1,i] - pts[1,i], 
                    angles='xy', 
                    scale_units='xy', 
                    scale=1, 
                    color='g', 
                    alpha=0.8, 
                    )

    # Make a rectangle the size of the original image
    rect = plt.Rectangle((0, 0), image_width, image_height, linewidth=1, edgecolor='g', facecolor='none')

    # Add the rectangle to the plot
    ax.add_patch(rect)

    # Set the limits of the plot to match the transformed points
    offset = 50
    ax.set_xlim(min(pts_tilde[0,:].min(), 0)-offset, max(pts_tilde[0,:].max(), image_width)+offset)
    ax.set_ylim(min(pts_tilde[1,:].min(),0)-offset, max(pts_tilde[1,:].max(), image_height)+offset)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Points in an 'image'")

    # Add a legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    
    return f, ax

def plot_affine_test_case_outlier_with_est(
        pts: np.ndarray,
        pts_tilde: np.ndarray,
        pts_est: np.ndarray,
        image_width: int,
        image_height: int,
        outlier_idxs: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the affine test case with outliers and estimated points.
    
    Parameters:
    -----------
    pts (np.ndarray): The source points.
    pts_tilde (np.ndarray): The target points.
    pts_est (np.ndarray): The estimated points.
    outlier_idxs (np.ndarray): The indices of the outliers.

    Returns:
    --------
    tuple: A tuple containing the figure and axes of the plot.
    
    """
    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Plot the original points
    inlier_pts = np.delete(pts, outlier_idxs, axis=1)
    ax.scatter(inlier_pts[0,:], inlier_pts[1,:], label='Original inliers', color='g', marker='o', facecolor='none')

    outlier_pts = pts[:, outlier_idxs]
    ax.scatter(outlier_pts[0,:], outlier_pts[1,:], label='Original outliers', color='r', marker='o', facecolor='none')

    # Plot the transformed points
    inlier_pts_tilde = np.delete(pts_tilde, outlier_idxs, axis=1)
    outlier_pts_tilde = pts_tilde[:, outlier_idxs]
    ax.scatter(inlier_pts_tilde[0,:], inlier_pts_tilde[1,:], label='Transformed inliers', color='g')
    ax.scatter(outlier_pts_tilde[0,:], outlier_pts_tilde[1,:], label='Transformed outliers', color='r')

    # Plot the estimated inlier points
    inlier_pts_est = np.delete(pts_est, outlier_idxs, axis=1)
    ax.scatter(inlier_pts_est[0,:], inlier_pts_est[1,:], label='Estimated transform on inliers', color='lightgreen', marker='x')

    # Plot the estimated inlier points
    outlier_pts_est = pts_est[:, outlier_idxs]
    ax.scatter(outlier_pts_est[0,:], outlier_pts_est[1,:], label='Estimated transform on outliers', color='orange', marker='x')


    # Plot line between inlier and transformed points
    for i in range(pts.shape[1]):

        # Check if the point is an outlier
        outlier = outlier_idxs[i]
        if outlier:

            ax.quiver(pts[0,i], pts[1,i], 
                    pts_tilde[0,i] - pts[0,i], 
                    pts_tilde[1,i] - pts[1,i], 
                    angles='xy', 
                    scale_units='xy', 
                    scale=1, 
                    color='r', 
                    alpha=0.8, 
                    )
        else:

            ax.quiver(pts[0,i], pts[1,i], 
                    pts_tilde[0,i] - pts[0,i], 
                    pts_tilde[1,i] - pts[1,i], 
                    angles='xy', 
                    scale_units='xy', 
                    scale=1, 
                    color='g', 
                    alpha=0.8, 
                    )

    # Make a rectangle the size of the original image
    rect = plt.Rectangle((0, 0), image_width, image_height, linewidth=1, edgecolor='g', facecolor='none')

    # Add the rectangle to the plot
    ax.add_patch(rect)

    # Set the limits of the plot to match the transformed points
    offset = 50
    ax.set_xlim(min(pts_tilde[0,:].min(), 0)-offset, max(pts_tilde[0,:].max(), image_width)+offset)
    ax.set_ylim(min(pts_tilde[1,:].min(),0)-offset, max(pts_tilde[1,:].max(), image_height)+offset)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Points in an 'image'")

    # Add a legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    
    return f, ax
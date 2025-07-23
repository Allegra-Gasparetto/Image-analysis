import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import *

from supplied import extract_sift_features, match_descriptors

def affine_test_case(N: int):
    """
    Generates a test case for estimating an affine transformation.

    Parameters:
        N (int): Number of 2D points.

    Returns:
        pts (np.ndarray): 2xN original points.
        pts_tilde (np.ndarray): 2xN transformed points.
        A_true (np.ndarray): 2x2 affine transformation matrix.
        t_true (np.ndarray): 2x1 translation vector.
    """
 
    # Generate random points
    pts = np.random.rand(2, N)

    # Generate a random affine transformation
    A_true = np.random.randn(2, 2)
    t_true = np.random.randn(2, 1)

    # Transformed points 
    pts_tilde = A_true @ pts + t_true

    return pts, pts_tilde, A_true, t_true

def estimate_affine(pts, pts_tilde):
    """
    Estimate affine transformation mapping `pts` to `pts_tilde`.

    Parameters:
        pts (np.ndarray): 2xK original points.
        pts_tilde (np.ndarray): 2xK transformed points.

    Returns:
        A (np.ndarray): 2x2 affine transformation matrix.
        t (np.ndarray): 2x1 translation vector.
    """
    K = pts.shape[1]
    X = np.zeros((2*K, 6))
    p = np.zeros((2*K, 1))

    for k in range(K):
        x, y = pts[:, k]
        x_tilde, y_tilde = pts_tilde[:, k]
        p[2*k] = x_tilde
        p[2*k + 1] = y_tilde
        X[2*k] = [x, y, 0, 0, 1, 0]
        X[2*k + 1] = [0, 0, x, y, 0, 1]
    
    # Solve using least squares
    params, _, _, _ = np.linalg.lstsq(X, p, rcond=None)
    A = params[:4].reshape(2, 2)
    t = params[4:].reshape(2, 1)

    return A, t

def residual_lgths(A, t, pts, pts_tilde):
    """
    Compute the lengths of the 2D residual vectors.

    Inputs:
        A (matrix): 2x2 affine transformation matrix.
        t (vector): 2x1 translation vector.
        pts (matrix): 2xN original points.
        pts_tilde (matrix): 2xN transformed points.

    Returns:
        lgths (array): N residual vector lengths.
    """

    # Residuals
    residuals = pts_tilde - (A @ pts + t.reshape(2, 1))

    # Column-wise sum of the squared elements
    lgths = np.sum(np.pow(residuals, 2), axis=0)

    return lgths

def affine_test_case_outlier(outlier_rate, n_samples, image_height, image_width):
    """
    Generates a test case for estimating an affine transformation and produces a percentage of outliers among the output points.
    
    Parameters:
        outlier_rate (float): percentage of outliers.
        n_samples (int): number of sample points to generate.
        image_height (int): height of the image.
        image_width (int): width of the image.
    
    Returns:
        pts (array): 2xN original points.
        pts_tilde (array): 2xN transformed points.
        A_true (matrix): 2x2 affine transformation matrix.
        t_true (vector): 2x1 translation vector.
        outlier_idxs (boolean array): indices of outliers.
    """
  
    # Generate random points
    pts = np.random.rand(2, n_samples)
    pts[0] = pts[0] * image_width   # Scale x-coordinates by image width
    pts[1] = pts[1] * image_height  # Scale y-coordinates by image height

    # Generate a random affine transformation
    A_true = np.random.rand(2, 2)  
    t_true = np.random.rand(2, 1)  
    
    # Transformed points 
    pts_tilde = A_true @ pts + t_true
    
    # Produce outliers
    n_outliers = int(outlier_rate * n_samples)
    outlier_idxs = np.zeros(n_samples, dtype=bool)
    outlier_idxs[np.random.choice(n_samples, n_outliers, replace=False)] = True
    outlier_pts = np.random.rand(2, n_samples)
    outlier_pts[0] = outlier_pts[0] * image_width   # Scale x-coordinates by image width
    outlier_pts[1] = outlier_pts[1] * image_height  # Scale y-coordinates by image height
    
    # Randomly spread outliers over the image
    pts_tilde[:, outlier_idxs] = outlier_pts[:, outlier_idxs]
    
    return pts, pts_tilde, A_true, t_true, outlier_idxs

def ransac_fit_affine(pts: np.ndarray, pts_tilde: np.ndarray, threshold: float, n_iter: int = 10000, max_inliers: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate an affine transformation between two sets of points using RANSAC.
    
    Parameters:
        pts (np.ndarray): 2xN original points.
        pts_tilde (np.ndarray): 2xN array of observed points.
        threshold (float): maximum residual to consider an inlier.
        n_iter (int): number of RANSAC iterations.
        max_inliers (int): early stop if this number is reached.
    
    Returns:
        A (np.ndarray): 2x2 affine transformation matrix.
        t (np.ndarray): 2x1 translation vector.
    """
    
    M = pts.shape[1]
    best_in = 0
    A = None
    t = None

    for _ in range(n_iter):
        # Estimate affine transformation 
        idx = np.random.choice(M, 3, replace=False)
        pts_sample = pts[:, idx]
        pts_tilde_sample = pts_tilde[:, idx]
        A_current, t_current = estimate_affine(pts_sample, pts_tilde_sample)
        
        # Compute the residuals
        errors = residual_lgths(A_current, t_current, pts, pts_tilde)

        # Maximize the number of inliers
        in_current = np.sum(errors < threshold)
        if in_current > best_in:
            best_in = in_current
            A = A_current
            t = t_current

    return A, t

def align_images(source: np.ndarray, target: np.ndarray, threshold: float = 10, plot: bool = True):
    """
    Aligns the source image to the target image using affine transformation.

    Parameters:
        source (np.ndarray): grayscale source image.
        target (np.ndarray): grayscale target image.
        thresh (float):  maximum residual to consider an inlier.
        plot (bool): True to plot the matched keypoints.

    Returns:
        warped (np.ndarray): warped image.
    """

    # Find good points and match them between the images
    kp_source, desc_source = extract_sift_features(source)
    kp_target, desc_target = extract_sift_features(target)
    good_matches, _ = match_descriptors(desc_source, desc_target)

    # Find coordinates of the matching keypoints
    from supplied import extract_keypoint_matches
    points_source, points_target = extract_keypoint_matches(kp_source, kp_target, good_matches)
 
    # Estimate affine transform
    A, t = ransac_fit_affine(points_target, points_source, threshold)

    # Warp source image using affine transformation
    from supplied import affine_warp
    affine_matrix = np.hstack([A, t.reshape(2, 1)])
    affine_matrix = np.vstack([affine_matrix, [0, 0, 1]])
    warped = affine_warp(source, affine_matrix, target.shape)

    # Plot matches
    if plot:
        from visualization import plot_matches
        matched_img = cv2.drawMatchesKnn(source, kp_source, target, kp_target, good_matches, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(matched_img, cmap='gray')
        plt.title("SIFT Feature Matches")
        plt.axis('off')
        plt.show()
    
    return warped

def ransac_fit_affine_ls(pts: np.ndarray, pts_tilde: np.ndarray, threshold: float, n_iter: int = 10000, max_inliers: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate an affine transformation between two sets of points using RANSAC.
    
    Parameters:
        pts (np.ndarray): 2xN original points.
        pts_tilde (np.ndarray): 2xN array of observed points.
        threshold (float): maximum residual to consider an inlier.
        n_iter (int): number of RANSAC iterations.
        max_inliers (int): early stop if this number is reached.
    
    Returns:
        A (np.ndarray): 2x2 affine transformation matrix.
        t (np.ndarray): 2x1 translation vector.
    """
    
    M = pts.shape[1]
    best_in = 0
    best_in_index = []
    best_A = None
    best_t = None

    for _ in range(n_iter):
        # Estimate affine transformation 
        idx = np.random.choice(M, 3, replace=False)
        pts_sample = pts[:, idx]
        pts_tilde_sample = pts_tilde[:, idx]
        A_current, t_current = estimate_affine(pts_sample, pts_tilde_sample)
        
        # Compute the residuals
        errors = residual_lgths(A_current, t_current, pts, pts_tilde)

        # Maximize the number of inliers
        in_current = np.sum(errors < threshold)
        in_current_index = np.where(errors < threshold)[0]
        if in_current > best_in:
            best_in = in_current
            best_in_index = in_current_index
            best_A = A_current
            best_t = t_current

    # Correction using all estimated inliers
    A, t = estimate_affine(pts[:, best_in_index], pts_tilde[:, best_in_index])

    return A, t

def align_images_inlier_ls(source: np.ndarray, target: np.ndarray, threshold: float = 10, plot: bool = False) -> np.ndarray:    
    """
    Aligns the source image to the target image using affine transformation.

    Parameters:
        source (np.ndarray): grayscale source image.
        target (np.ndarray): grayscale target image.
        thresh (float):  maximum residual to consider an inlier.
        plot (bool): True to plot the matched keypoints.

    Returns:
        warped (np.ndarray): warped image.
    """

    # Find good points and match them between the images
    kp_source, desc_source = extract_sift_features(source)
    kp_target, desc_target = extract_sift_features(target)
    good_matches, _ = match_descriptors(desc_source, desc_target)

    # Find coordinates of the matching keypoints
    from supplied import extract_keypoint_matches
    points_source, points_target = extract_keypoint_matches(kp_source, kp_target, good_matches)

    # Estimate affine transform
    A, t = ransac_fit_affine_ls(points_target, points_source, threshold) 

    # Warp source image using affine transformation
    from supplied import affine_warp
    affine_matrix = np.hstack([A, t.reshape(2, 1)])
    affine_matrix = np.vstack([affine_matrix, [0, 0, 1]])
    target_shape = target.shape
    warped = affine_warp(source, affine_matrix, target_shape)

    # Plot matches
    if plot:
        from visualization import plot_matches
        matched_img = cv2.drawMatchesKnn(source, kp_source, target, kp_target, good_matches, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(matched_img, cmap='gray')
        plt.title("SIFT Feature Matches")
        plt.axis('off')
        plt.show()
    
    return warped

def sample_image_at(image, position):
    """
    Gives the pixel value at position using neareast neighbours.

    Parameters:
        image (np.ndarray): 2D image array.
        position (tuple): (x, y) coordinates where the image should be sampled.

    Returns:
        int: pixel value at the nearest neighbor to the position, or 255 if out of bounds.
    """

    # Pixel coordinate
    x, y = position
    x_rounded = int(round(x))
    y_rounded = int(round(y))
    
    # Pixel value
    height = image.shape[0]
    width = image.shape[1]    
    if (x_rounded >= 0) and (x_rounded < width) and (y_rounded >= 0) and (y_rounded < height):
        return image[y_rounded, x_rounded]
    else:
        return 255

def warp_16x16(source):
    """
    Warps a 16 x 16 image source according to the function transform_coordinates and forms an output 16 x 16 image warped.

    Parameters:
        source (np.ndarray): grayscale source image.
    
    Returns:
        warped (np.ndarray): warped image.
    """
    
    warped = np.zeros((16, 16), dtype=np.uint8)
    
    # Loop over all target pixels
    from supplied import transform_coordinates
    for y_target in range(16):
        for x_target in range(16):
            pos_target = (x_target, y_target)
            pos_source = transform_coordinates(pos_target)
            pixel_value = sample_image_at(source, pos_source)
            warped[y_target, x_target] = pixel_value

    return warped
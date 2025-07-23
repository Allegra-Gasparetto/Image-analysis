import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import cv2

from PIL import Image

from typing import *

def transform_coordinates(pos):
    """"
    Transform the coordinates of a point from the source image to the target image."
    
    Parameters:
    -----------
    pos (list): The coordinates of the point in the source image.

    Returns:
    --------
    list: The coordinates of the point in the target image.
    """
    

    dx, dy = get_delta(pos[0], pos[1])
    pos_tilde = [pos[0] + dx, pos[1] + dy]
    return pos_tilde

def get_delta(x, y):
    """
    Get the transformation of a point from the source image to the target image.

    Parameters:
    -----------
    x (int): The x-coordinate of the point in the source image.
    y (int): The y-coordinate of the point in the source image.

    Returns:
    --------
    tuple: The transformation of the point.
    """

    x = min(max(x, 0), 15)
    y = min(max(y, 0), 15)

    rowd = np.array([
        [0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0],
        [-2, 0,  0,  0,  0,  1,  1,  1,  1, -1, -1, 0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],
        [0,  0,  0,  0,  0,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  0, -1, 0],
        [0,  0, -2, -2, 0,  -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, 0],
        [0,  0,  0,  0,  0,  0, -2, 0,  0, -2, 0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, 0,  0,  0,  0],
        [0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, 0,  0,  0,  0]
    ])

    cold = np.array([
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0, -1, 0,  0,  0,  0,  0,  0,  0],
        [0,  0,  1, -1, -2, 0,  0,  0, -1, 0,  0,  0, -1, -1, 0,  0],
        [0,  2,  0, -1, 0,  0,  0,  0, -1, -2, 0,  0, -1, 0,  -2, 0],
        [0,  2,  0,  1,  0, -3, 2,  1,  0, -2, -3, 0, -1, 0,  -2, 0],
        [3,  3,  2,  1, -2, -3, 2,  1,  0,  0, -3, -4, -5, -6, -7, -3],
        [3,  3,  2,  1, -2, -3, 2,  1,  0,  0, -3, -4, -1, 0,  -7, -3.1],
        [3,  0,  2,  1, -2, -3, 2,  1,  0,  0, -3, 0,  -1, 0,  0,  -3],
        [3,  0,  0,  1,  0,  0,  0,  1,  0, -2, -3, 0,  -1, 0,  0,  -3],
        [3,  0,  0,  1,  0,  0,  0,  0, -1, -2, -3, -4, 1,  0,  -7, -3],
        [3,  0,  2,  1, -1, 0,  0,  0,  0,  0,  0,  0, -5, -6, -7, -3],
        [0,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1, -2, 0],
        [0,  0,  1, -1, 0,  0,  0,  0,  0,  0,  0,  0, -1, -1, 0,  0],
        [0,  0,  1, -1, 0,  0,  0,  0,  0,  0,  0,  0, -1, -1, 0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ])

    dy = rowd[y, x]
    dx = cold[y, x]
    return dx, dy

def read_as_grayscale(image_path: str) -> np.ndarray:

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    return image


def extract_sift_features(image: np.ndarray) -> Tuple[Sequence[cv2.KeyPoint], cv2.UMat]:
    """
    Extracts SIFT descriptors from an image.

    Parameters:
    -----------
    image (numpy.ndarray): The input image.

    Returns:
    --------
    Tuple[Sequence[cv2.KeyPoint], cv2.UMat]: A tuple containing the keypoints and descriptors.
    """
    
    # Create a SIFT detector
    sift = cv2.SIFT.create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors

def match_descriptors(
        descriptors_src: cv2.UMat, 
        descriptors_tgt: cv2.UMat, 
        max_ratio: float = 0.8) -> Tuple[Sequence[Sequence[cv2.DMatch]], Sequence[Sequence[cv2.DMatch]]]:
    """
    Matches descriptors from the test image to the training descriptors using brute-force
    matches and Lowe's ratio test.

    Returns the good matches and all matches.

    Parameters:
    -----------
    descriptors_src (cv2.UMat): The source image descriptors.
    descriptors_tgt (cv2.UMat): The target image descriptors.
    max_ratio (float): The maximum ratio for Lowe's ratio test.

    Returns:
    --------
    Tuple[Sequence[Sequence[cv2.DMatch]], Sequence[Sequence[cv2.DMatch]]]: A tuple containing the good matches and all matches.

    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors_src, descriptors_tgt, k=2)

    # Apply Lowe's ratio test
    good_matches = []

    # Iterate through the matches and apply the ratio test
    # selecting only the good matches
    for i, (m,n) in enumerate(matches):
        if m.distance < max_ratio * n.distance:
            good_matches.append([m])

    return good_matches, matches

def extract_keypoint_matches(
        pts_src: Sequence[cv2.KeyPoint], 
        pts_tgt: Sequence[cv2.KeyPoint], 
        matches: Sequence[Sequence[cv2.DMatch]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the matched keypoints from the source and target images.

    Parameters:
    -----------
    keypoints_src (Sequence[cv2.KeyPoint]): The keypoints from the source image.
    keypoints_tgt (Sequence[cv2.KeyPoint]): The keypoints from the target image.
    matches (Sequence[Sequence[cv2.DMatch]]): The matches between the keypoints.

    Returns:
    --------
    Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the matched keypoints from the source and target images.
    """
    
    pts_src_to_use = [pts_src[m[0].queryIdx].pt for m in matches]
    pts_tgt_to_use = [pts_tgt[m[0].trainIdx].pt for m in matches]
    pts_src_to_use = np.array(pts_src_to_use).astype(np.float32).T
    pts_tgt_to_use = np.array(pts_tgt_to_use).astype(np.float32).T

    return pts_src_to_use, pts_tgt_to_use


def affine_warp(source: np.ndarray, affine_matrix: np.ndarray, target_shape: np.ndarray):
    """
    Warp the source image to the target image using an affine transformation.
    
    Parameters:
    -----------
    source (numpy.ndarray): The source image to be warped.
    affine_matrix (numpy.ndarray): The affine transformation matrix.
    
    Returns:
    --------
    numpy.ndarray: The warped image.
    """

    # Use OpenCV to warp the image
    warped = cv2.warpPerspective(source, affine_matrix, (target_shape[1], target_shape[0]), flags=cv2.WARP_INVERSE_MAP)  # <-- flag tells OpenCV that we are giving a target → source mapping, i.e. backward warping

    return warped

def read_image(image_path):

    # Load the image
    image = Image.open(image_path)

    # Convert the image to a numpy array
    image = np.array(image)

    # Convert image to 0-1 range
    image = image / 255

    return image
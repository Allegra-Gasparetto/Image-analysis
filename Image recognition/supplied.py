import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob

def plot_overlay(image, binary_mask, maxima_coords=None):
    """
    Creates and displays an overlay of the binary mask on the original image.
    
    Parameters:
    image (numpy array): Grayscale image normalized to [0,1].
    binary_mask (numpy array): Binary mask with detected regions.
    maxima_coords (numpy array, optional): 2×n array with local maxima coordinates
    """

    if image.ndim == 3:
        image = np.mean(image, axis=2)
    
    overlay = np.zeros((*image.shape, 3), dtype=np.float32)
    overlay[..., 1] = binary_mask  # Green channel
    blended = (0.7 * image[..., np.newaxis]) + (0.3 * overlay)  # Blend grayscale with overlay

    blended = np.clip(blended, 0, 1)
    f, ax = plt.subplots()
    ax.imshow(blended)
    if maxima_coords is not None:
        ax.scatter(maxima_coords[0], maxima_coords[1], c='red', s=10, label='Local Maxima')
    ax.axis('off')
    plt.show()

def extract_sift_features(image_path):
    """Extracts SIFT descriptors from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'Error: Could not load image {image_path}')
        return None, None
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_descriptors(descriptors, training_data):
    """
    Matches descriptors from the test image to the training descriptors using Lowe's ratio test.
    Returns the good matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors.astype(np.float32), training_data['descriptors'].astype(np.float32), k=2)

    # Apply Lowe's ratio test
    ratio_thresh = 0.8  # Lowe's ratio test threshold
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    return good_matches

def create_training_data(training_folder):
    """Extracts SIFT descriptors from all training images and returns the training data."""
    descriptors_list = []
    labels_list = []
    class_names = sorted([d for d in os.listdir(training_folder) if os.path.isdir(os.path.join(training_folder, d))])

    for label, class_name in enumerate(class_names):
        image_paths = glob(os.path.join(training_folder, class_name, '*.jpg'))
        for image_path in image_paths:
            _, descriptors = extract_sift_features(image_path)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                labels_list.extend([label] * descriptors.shape[0])
    
    descriptors_array = np.vstack(descriptors_list) if descriptors_list else np.array([])
    labels_array = np.array(labels_list, dtype=np.int32)
    
    training_data = {
        'descriptors': descriptors_array,
        'labels': labels_array,
        'names': class_names
    }

    return training_data
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
from PIL import Image
from scipy.ndimage import rank_filter
from supplied import extract_sift_features, match_descriptors

def read_image(image_path):
    """
    Takes the file path to an image and returns an image array
    
    Parameters:
    image_path (str): File path to an image ['lab_N/data/name.png']

    Returns:
    image (numpy array): The RGB image.
    """

    # Load the image
    image = Image.open(image_path)
    # Convert the image to a numpy array
    image = np.array(image)
    # Convert image to 0-1 range
    image = image / np.max(image)

    return image

def read_as_grayscale(image_path):
    """
    Takes the file path to an image and return a grayscale image array
    
    Parameters:
    image_path (str): File path to an image ['lab_N/data/name.png']

    Returns:
    image_grayscale (numpy array): The grayscale image.
    """
    
    # Load the image
    image = Image.open(image_path)
    # Convert the image to a numpy array
    image = np.array(image)
    # Convert image to 0-1 range
    image = image / np.max(image)
    if image.ndim == 3:
        # RGB image to grayscale image
        image_grayscale = np.mean(image, axis=2)
    else:
        image_grayscale = image

    return image_grayscale

def create_covariance_filter(pos_patches):
    """
    Creates a covariance filter from a set of positive patches.
    
    Parameters:
    pos_patches (numpy array): Array of positive patches.
    
    Returns:
    w (numpy array): The computed covariance filter.
    """
    
    nbr_pos = len(pos_patches)
    w = np.zeros_like(pos_patches[0])

    # Compute the number of pixel
    if len(pos_patches[0].shape) == 3: 
        m, n, c = pos_patches[0].shape
        N = m*n*c
    else:
        m, n = pos_patches[0].shape
        N = m*n
    # Compute the average cell
    for pos_patch in pos_patches:
        w = w + pos_patch
    w = w / nbr_pos
    # Compute the mean pixel value of average cell from the positive examples
    mu = np.mean(np.array(w))
    # Compute the covariance filter
    w = (w - mu) / N   
    
    return w

def classification(pos_patches, neg_patches, w, tau):
    """
    Classify
    
    Parameters:
    pos_patches (numpy array): Foreground patches.
    neg_patches (numpy array): Background patches.
    w (numpy array): The covariance filter.
    tau (numpy array): Threshold range.
    
    Returns:
    tp (integer): Number of true positives.
    fn (integer): Number of false negatives.
    fp (integer): Number of false positives.
    tn (integer): Number of true negatives.
    true_pos (numpy array): True positives.
    false_neg (numpy array): False negatives.
    true_neg (numpy array): True negatives.
    false_pos (numpy array): False positives. 
    """   

    # Compute the covariance for positive and negative examples
    pos_cov = []
    neg_cov = []
    for pos_patch in pos_patches:
        pos_cov.append(np.sum(w * pos_patch))
    for neg_patch in neg_patches:
        neg_cov.append(np.sum(w * neg_patch))
    # Compute the correctly and wrongly classified examples
    true_pos = []; false_neg = []; true_neg = []; false_pos = []
    tp = 0; fn = 0; tn = 0; fp = 0
    for i,cov in enumerate(pos_cov):
        if cov <= tau:
            false_neg.append(i)
            fn = fn + 1
        else:
            true_pos.append(i)
            tp = tp + 1
    for i,cov in enumerate(neg_cov):
        if cov <= tau:
            true_neg.append(i)
            tn = tn + 1
        else:
            false_pos.append(i) 
            fp = fp + 1

    return tp, fn, fp, tn, true_pos, false_neg, true_neg, false_pos
    
def compute_threshold(pos_patches, neg_patches, w):
    """
    Computes the optimal threshold that minimizes misclassification.
    
    Parameters:
    pos_patches (numpy array): Foreground patches.
    neg_patches (numpy array): Background patches.
    w (numpy array): The covariance filter.
    
    Returns:
    thr (float): Optimal threshold for classification.
    precision (float): Precision of the classifier.
    recall (float): Recall of the classifier.
    conf_matrix (numpy array): The confusion matrix.
    """   
    # Compute the covariance for positive and negative examples
    pos_cov = []
    neg_cov = []
    for pos_patch in pos_patches:
        pos_cov.append(np.sum(w * pos_patch))
    for neg_patch in neg_patches:
        neg_cov.append(np.sum(w * neg_patch))
    # Compute the optimal threshold
    tau_min = min(np.min(pos_cov), np.min(neg_cov))
    tau_max = max(np.max(pos_cov), np.max(neg_cov))
    tau_range = np.linspace(tau_min,tau_max,100)
    min_misclassified = len(pos_patches) + len(neg_patches) # all the images
    for tau in tau_range:
        # Compute the correctly and wrongly classified examples
        true_pos = []; false_neg = []; true_neg = []; false_pos = []
        tp = 0; fn = 0; tn = 0; fp = 0
        for i,cov in enumerate(pos_cov):
            if cov < tau:
                false_neg.append(i)
                fn = fn + 1
            else:
                true_pos.append(i)
                tp = tp + 1
        for i,cov in enumerate(neg_cov):
            if cov < tau:
                true_neg.append(i)
                tn = tn + 1
            else:
                false_pos.append(i) 
                fp = fp + 1
        # Compute the number of misclassified examples
        total_misclassified = fn + fp
        # Compute the optimal threshold, the precision and the recall, the confusion matrix
        if total_misclassified < min_misclassified:
            min_misclassified = total_misclassified
            thr = tau
            precision = tp / (tp+fp)
            recall = tp / (tp+fn)
            conf_matrix = np.matrix([[tp, fn], [fp, tn]])
    
    return thr, precision, recall, conf_matrix

def strict_local_maxima(response, threshold):
    """
    Computes the coordinates of all strict local maxima in the response image.
    
    Parameters:
    response (numpy array): Input response image.
    threshold (float): Threshold for classification.
    
    Returns:
    (col_coords, row_coords) (numpy array): 2 x n array 
                                            with column coordinates in the first row
                                            and row coordinates in the second row.
    """
   
    nhood_size = (3,3)
    next_best = rank_filter(response, -2, size=nhood_size) # Selecting the second highest pixel value from the neighborhood of each pixel.

    # Your code here

    # Compute the strict local maxima pixel values
    slm = response > next_best
    slm &= response > threshold
    # Compute the coordinates of the strict local maxima
    row_coords, col_coords = np.where(slm)

    return (col_coords, row_coords)

def detector(image, w, thr):
    """
    Detects cell centers in an image (grayscale or RGB) using a linear classifier and non-maximum suppression.
    
    Parameters:
    image (numpy array): Input image (grayscale or RGB)
    w (numpy array): The covariance filter
    thr (float): Threshold for classification

    Returns:
    centers (numpy array): Cell centers.
    thresholded_response (numpy array): Thresholded response image.
    """

    # Compute the linear classification with sliding dot product
    if len(image.shape) == 3: 
        response = cv2.filter2D(image[:, :, 0], -1, w[:, :, 0]) + cv2.filter2D(image[:, :, 1], -1, w[:, :, 1]) + cv2.filter2D(image[:, :, 2], -1, w[:, :, 2])
    else:
        response = cv2.filter2D(image, -1, w)
    thresholded_response = response > thr
    # Compute the non-Maximum  classification
    nhood_size = (3, 3)
    next_best = rank_filter(response, -2, size=nhood_size)
    nms_mask = response > next_best
    nms_mask &= response > thr
    # Compute the coordinates of detected centers
    row_coords, col_coords = np.where(nms_mask)
    centers = (col_coords, row_coords)

    return centers, thresholded_response

def classify_church(image_path, training_data):
    """
    Classifies a given image by matching its SIFT descriptors to stored training descriptors 
    and using a voting system to determine the most likely church.

    Parameters:
    image_path (str): Path to input image to be classified.
    training_data (dict): Dictionary containing:
        - 'descriptors' (numpy array): Stored SIFT descriptors (128 × N).
        - 'labels' (numpy array): 1D array mapping each descriptor to a church label.
        - 'names' (list of str): List of church names corresponding to labels.

    Returns:
    label (int): Predicted label corresponding to the most likely church.
    """
    
    # Your code here.

    # Extract SIFT descriptors from all training images 
    keypoints, descriptors = extract_sift_features(image_path)
    # Match descriptors from the test image to the training descriptors
    good_matches = match_descriptors(descriptors, training_data)
    # Compute the vote: each match votes for its class
    num_classes = len(training_data['names'])  # unique church classes
    votes = np.zeros(num_classes)
    for match in good_matches:
        matched_class_label = training_data['labels'][match.trainIdx]
        votes[matched_class_label] = votes[matched_class_label] + 1
    # Classify based on the class with the highest number of votes
    label = np.argmax(votes)

    return label
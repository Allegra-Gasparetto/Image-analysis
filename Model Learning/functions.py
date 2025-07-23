import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def partial_gradient(w, w0, example_train, label_train):
    """
    Computes the derivatives of the partial loss Li with respect to each of the classifier parameters.

    Parameters:
    - w: weight vector (same shape as example_train)
    - w0: bias term (scalar)
    - example_train: input image / example (same shape as w)
    - label_train: 0 or 1 (negative or positive example)

    Returns:
    - wgrad: gradient with respect to w
    - w0grad: gradient with respect to w0
    """

    # Write your code here

    y = np.vdot(example_train, w) + w0
    p = np.exp(y) / (1 + np.exp(y))
    
    wgrad = (label_train * (p - 1) * example_train) + ((1 - label_train) * p* example_train)
    w0grad = (label_train * (p - 1) + (1 - label_train) * p)

    return wgrad, w0grad

def process_epoch(w, w0, lrate, examples_train, labels_train, random_order=True):
    """
    Performs one epoch of stochastic gradient descent.

    Parameters:
    - w: weight array (same shape as examples)
    - w0: bias term (scalar)
    - lrate: learning rate (scalar)
    - examples_train: list or array of training examples (e.g., shape (N, 35, 35))
    - labels_train: array of labels (shape (N,))
    
    Returns:
    - Updated w and w0 after one epoch
    """

    # Write your code here

    if random_order:
        # Random permutation of the patches
        idx_permutation = np.random.permutation(len(labels_train))
        examples_train = examples_train[idx_permutation]
        labels_train = labels_train[idx_permutation]

    for i in range(len(labels_train)):
        # Compute gradients of the partial loss
        wgrad, w0grad = partial_gradient(w, w0, examples_train[i], labels_train[i])
        # Update filter and bias
        w = w - (lrate * wgrad)
        w0 = w0 - (lrate * w0grad)
        
    return w, w0

def classify(examples_val, w, w0):
    """
    Applies a classifier to the example data.
    
    Parameters:
    - examples_val: List of validation examples (each example is a 1D array)
    - w: weight array (same shape as each example in examples_val)
    - w0: bias term (scalar)
    
    Returns:
    - predicted_labels: Array of predicted labels (0 or 1) for each example
    """

    # Write your code here

    # Compute positive and negative examples
    predicted_labels = np.zeros(len(examples_val))
    for i, example_val in enumerate(examples_val):
        # Compute the response and add the bias
        y = np.vdot(example_val, w) + w0
        # Assign label 1 to positive examples
        if y>0:
            predicted_labels[i] = 1 

    return predicted_labels

def augment_data(examples_train, labels_train, M):
    """
    Data augmentation: Takes each sample of the original training data and 
    applies M random rotations, which result in M new examples.
    
    Parameters:
    - examples_train: List of training examples (each example is a 2D array)
    - labels_train: Array of labels corresponding to the examples
    - M: Number of random rotations to apply to each training example
    
    Returns:
    - examples_train_aug: Augmented examples after rotations
    - labels_train_aug: Corresponding labels for augmented examples
    """
    # Write your code here
    # Compute the rotated examples for random angles between 0 and 360 degrees
    examples_train_aug = []
    labels_train_aug = []
    for example, label in zip(examples_train, labels_train):
        for m in range(M):
            angle = np.random.uniform(0, 360)  
            rotated = rotate(example, angle, reshape=False, mode='nearest')
            examples_train_aug.append(rotated)
            labels_train_aug.append(label)
    
    examples_train_aug = np.array(examples_train_aug)
    labels_train_aug = np.array(labels_train_aug)

    return examples_train_aug, labels_train_aug
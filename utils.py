import random
import colorsys

import numpy as np

def read_classes(classes_path):
    """Reads classes from file.
    
    Args:
        classes_path (str):
            Path to file containing names of all classes.
    
    Returns:
        list: List containing names of all classes.
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    """Reads anchor values from file.
    
    Args:
        anchors_path (str):
            Path to file containing anchor values for YOLO model.
    
    Returns:
        numpy.ndarray: Array containing all anchor values.
    """
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    """Generates different colours for all classes.

    Args:
        class_names (list of `str`):
            List containing names of all classes.

    Returns:
        list: List containing colours for corresponding classes.
    """
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors
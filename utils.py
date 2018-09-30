import random
import colorsys

import cv2
import numpy as np

from keras import backend as K

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

def preprocess_image_cv2(image, dims):
    """Preprocesses and normalizes an image using openCV.

    Args:
        image (numpy.ndarray):
            Image to preprocess.
        dims (tuple of `int`):
            Desired dimensions of the image.

    Returns:
        numpy.ndarray: Preprocessed image data.
    """
    resized_image = cv2.resize(image, dims, interpolation=cv2.INTER_CUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

def scale_boxes(boxes, image_shape):
    """Scales the predicted boxes in order to be drawable on the image
    
    Args:
        boxes (tf.Tensor):
            Corner values of bounding boxes.
        image_shape (tuple of `int`):
            Shape of the original image.

    Returns:
        tf.Tensor: Scaled corner values for bounding boxes.
    """
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def draw_boxes_cv2(image, scores, boxes, classes, class_names, colors):
    """Draws bounding boxes on frame using openCV.

    Args:
        image (numpy.ndarray):
            Image on which to draw bounding boxes.
        scores (numpy.ndarray):
            Scores for each bounding box.
        classes (numpy.ndarray):
            Classes associated with each bounding box.
        class_names (list of `str`):
            List containing names of all classes.
        colors (list of `tuple` of `int`):
            List containing RGB values for all classes to use when drawing boxes.
    """
    image_shape = list(reversed(image.shape[:-1]))
    for i, c in reversed(list(enumerate(classes))):
        predicted_class = class_names[c]
        box = boxes[i]
        score = scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)
        # label = label.upper()

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image_shape[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
    
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)

        cv2.rectangle(image, (left, top), (right, bottom), colors[c], 2)
        cv2.rectangle(image, (left, top - text_size[0][1] - 10), 
                    (left + text_size[0][0] + 10, top), colors[c], cv2.FILLED)
        cv2.putText(image, label, (left + 5, top - 5), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
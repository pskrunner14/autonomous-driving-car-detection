import os
import time
import click
import logging

import cv2
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
import scipy.misc as misc

from multiprocessing.dummy import Pool
from PIL import Image, ImageDraw, ImageFont

# local imports
from utils import (
    read_classes, 
    read_anchors, 
    generate_colors, 
    preprocess_image, 
    draw_boxes, 
    scale_boxes
)

# imports from `yad2k` project
from yad2k.models.keras_yolo import (
    yolo_head, 
    yolo_boxes_to_corners, 
    preprocess_true_boxes, 
    yolo_loss, 
    yolo_body
)

@click.command()
@click.option(
    '-ip',
    '--image-path', 
    default='images/test/cars.jpg',
    type=click.Path(exists=True),
    help='Path for test image to detect objects in.'
)
@click.option(
    '-r',
    '--realtime',
    is_flag=True,
    help='Flag for real-time object detection.'
)
def main(image_path, realtime):
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level='INFO')

    anchors = read_anchors('model_data/yolo_anchors.txt')
    class_names = read_classes('model_data/coco_classes.txt')

    if realtime:
        logging.info('testing webcam frame dimensions')
        cv2.namedWindow('test-shape')
        vc = cv2.VideoCapture(0)
        while vc.isOpened():
            _, frame = vc.read()
            image_shape = tuple([float(x) for x in frame.shape[:-1]])
            break
        del vc
        cv2.destroyWindow('test-shape')
    else:
        image_shape = tuple([float(x) for x in misc.imread(image_path).shape[:-1]])

    logging.info('loading YOLOv2 Darknet19 model')
    yolo = YOLO(
        model_path='model_data/yolo_model.h5',
        dims=image_shape,
        anchors=anchors,
        class_names=class_names
    )

    if realtime:
        """ Use YOLO for making real-time detections in OpenCV """
        logging.info('starting webcam for real-time detections')
        webcam_realtime_object_detector(yolo)
    else:
        """ Use YOLO to detect objects in imags and save the results """
        logging.info('detecting objects in `{}`'.format(image_path))
        yolo.detect_image(image_path)

def webcam_realtime_object_detector(yolo=None):
    if yolo is None:
        raise UserWarning('YOLO model not found.')

    logging.info('Press ESC to exit')
    cv2.namedWindow('detector')
    vc = cv2.VideoCapture(0)
    
    while vc.isOpened():
        _, frame = vc.read()
        image = frame
        image  = yolo.run_yolo_detection(image)
        key = cv2.waitKey(10)
        cv2.imshow('detector', image)
        if key == 27: # exit on ESC
            logging.info('exiting')
            break
    logging.info('destroying detector window')
    cv2.destroyWindow('detector')

def draw_boxes_cv2(image, scores, boxes, classes, class_names):
    for i, c in reversed(list(enumerate(classes))):
        predicted_class = class_names[c]
        box = boxes[i]
        score = scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, label, 
                (left, top - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

class YOLO():
    """YOLOv2 real-time object detection using pre-trained model.
    For obtaining the pre-trained model using YOLOv2 weights, see
    YAD2K project: https://github.com/allanzelener/YAD2K.

    Args:
        model_path (str):
            Path to pre-trained model.
        dims (tuple of `float`):
            Dimensions of the frame to detect objects in.
        anchors (numpy.ndarray):
            YOLO anchor values.
        class_names (list of `str`):
            List containing names of all classes.
    """
    def __init__(self, model_path=None, dims=None, anchors=None, class_names=None):
        if model_path is None or dims is None or len(dims) != 2 or anchors is None or class_names is None:
            raise UserWarning('arguments do not match the spec!')
        self._model = keras.models.load_model(model_path, compile=False)
        self._class_names = class_names
        self._anchors = anchors
        self._dims = dims
        self._sess = K.get_session()
        self._model_input_dims = (608, 608)
        self._construct_graph()

    @staticmethod
    def _filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
        """Filter out bounding boxes that have highest scores.

        Args:
            box_confidence (tf.Tensor):
                Sigmoid confidence value for potential 
                bounding boxes.
            boxes (tf.Tensor):
                Tensor containing potential bounding 
                boxes' corners.
            box_class_probs (tf.Tensor):
                Softmax probabilities for potential 
                bounding boxes.
            threshold (float, optional):
                Threshold value for minimum score for 
                a bounding box.

        Returns:
            tf.Tensor:
                Filtered box scores.
            tf.Tensor:
                Filtered box corners.
            tf.Tensor:
                Filtered box classes.
        """
        box_scores = box_confidence * box_class_probs   # Compute box scores
        # Find box_classes thanks to max box_scores 
        # and keep track of the corresponding score
        box_classes = K.argmax(box_scores, axis=-1)   # index of max score
        box_class_scores = K.max(box_scores, axis=-1)   # actual max score
        # Create a filtering mask based on 'box_class_scores' 
        # by using 'threshold'. The mask should have the same 
        # dimension as box_class_scores, and be True for the
        # boxes we want to keep (with probability >= threshold)
        filtering_mask = box_class_scores >= threshold
        # Apply the mask to scores, boxes and classes
        scores = tf.boolean_mask(box_class_scores, filtering_mask)
        boxes = tf.boolean_mask(boxes, filtering_mask)
        classes = tf.boolean_mask(box_classes, filtering_mask)
        
        return scores, boxes, classes

    @staticmethod
    def _non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
        """Applies non-max suppression to bounding boxes.

        Args:
            scores (tf.Tensor):
                Scores of bounding boxes after filtering.
            boxes (tf.Tensor):
                Corner values of bounding boxes after filtering.
            classes (tf.Tensor):
                Classes for bounding boxes after filtering.
             max_boxes (int, optional):
                Max. number of bounding boxes for non-max 
                suppression.
            iou_threshold (float, optional):
                Intersection over union threshold for non-max 
                suppression.

        Returns:
            tf.Tensor:
                Non-max suppressed box scores.
            tf.Tensor:
                Non-max suppressed box corners.
            tf.Tensor:
                Non-max suppressed box classes.
        """
        max_boxes_tensor = K.variable(max_boxes, dtype='int32')             # tensor to be used in `tf.image.non_max_suppression`
        K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
        # To get the list of indices corresponding to boxes you keep
        nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=iou_threshold)
        # To select only nms_indices from scores, boxes and classes
        scores = K.gather(scores, nms_indices)
        boxes = K.gather(boxes, nms_indices)
        classes = K.gather(classes, nms_indices)
        
        return scores, boxes, classes

    def _construct_graph(self, max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
        """Constructs graph and instantiates operations on default graph.

        Args:
            max_boxes (int, optional):
                Max. number of bounding boxes for 
                non-max suppression.
            score_threshold (float, optional):
                Threshold value for min. score for 
                a bounding box for score-filtering.
            iou_threshold (float, optional):
                Intersection over union threshold
                for non-max suppression.
        """
        yolo_outputs = yolo_head(self._model.output, self._anchors, len(self._class_names))
        box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
        boxes = yolo_boxes_to_corners(box_xy, box_wh)   # Convert boxes to be ready for filtering functions
        scores, boxes, classes = self._filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
        boxes = scale_boxes(boxes, self._dims)   # Scale boxes back to original image shape.
        scores, boxes, classes = self._non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
        # Save tensors for later evaluation
        self._scores = scores
        self._boxes = boxes
        self._classes = classes

    def detect_image(self, image_path):
        """Detects objects in an image using YOLOv2.
        
        Args:
            image_path (str):
                Path to image for detection.

        Returns:
            numpy.ndarray:
                Output image data after detection
                and drawing bounding boxes over it.
        """
        image, image_data = preprocess_image(image_path, self._model_input_dims)
        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # Need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = self._sess.run([self._scores, self._boxes, self._classes], 
                                                    feed_dict={self._model.input: image_data, 
                                                                K.learning_phase(): 0})
        image_name = os.path.split(image_path)[-1]
        logging.info('Found {} objects belonging to known classes'.format(len(out_boxes)))
        
        colors = generate_colors(self._class_names)
        draw_boxes(image, out_scores, out_boxes, out_classes, self._class_names, colors)
        image.save(os.path.join('images/out', image_name), quality=90)
        
        return image

    def run_yolo_detection(self, frame):
        """Detects objects in real-time using YOLOv2.
        
        Args:
            frame (numpy.ndarray):
                Single frame from the webcam feed 
                to run YOLO detection on.

        Returns:
            numpy.ndarray:
                Output frame data after detection
                and drawing bounding boxes over it.
        """
        resized_image = cv2.resize(frame, self._model_input_dims, interpolation=cv2.INTER_CUBIC)
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # Need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = self._sess.run([self._scores, self._boxes, self._classes], 
                                                    feed_dict={self._model.input: image_data, 
                                                                K.learning_phase(): 0})
        logging.info('Found {} objects belonging to known classes'.format(len(out_boxes)))
        draw_boxes_cv2(frame, out_scores, out_boxes, out_classes, self._class_names)
        
        return frame

    def __del__(self):
        self._sess.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('EXIT')
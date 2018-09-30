import os
import logging

import keras
import keras.backend as K
import tensorflow as tf
import imageio as io

# local imports
from utils import (
    read_classes, 
    read_anchors, 
    generate_colors,
    preprocess_image_cv2,
    scale_boxes,
    draw_boxes_cv2
)

# imports from `yad2k` project
from yad2k.models.keras_yolo import (
    yolo_head, 
    yolo_boxes_to_corners
)

class YOLO():
    """YOLOv2 real-time object detection using pre-trained model.
    For obtaining the pre-trained model using YOLOv2 weights, see
    YAD2K project: https://github.com/allanzelener/YAD2K.

    Args:
        model_path (str):
            Path to pre-trained model.
        anchors_path (str):
            Path to file conataining YOLO anchor values.
        classes_path (str):
            Path to file containing names of all classes.
        dims (tuple of `float`):
            Dimensions of the frame to detect objects in.

    Raises:
        UserWarning: If any arg is missing or length of dims is not 2.
    """
    def __init__(self, model_path=None, anchors_path=None, classes_path=None, dims=None):
        if model_path is None or anchors_path is None or classes_path is None or dims is None or len(dims) != 2:
            raise ValueError('Arguments do not match the specification.')
        self._model = keras.models.load_model(model_path, compile=False)
        self._anchors = read_anchors(anchors_path)
        self._class_names = read_classes(classes_path)
        self._dims = dims
        self._model_input_dims = (608, 608)
        self._colors = generate_colors(self._class_names)
        self._sess = K.get_session()
        self._construct_graph()

    @staticmethod
    def _filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
        """Filter out bounding boxes that have highest scores.

        Args:
            box_confidence (tf.Tensor):
                Sigmoid confidence value for potential bounding boxes.
            boxes (tf.Tensor):
                Tensor containing potential bounding boxes' corners.
            box_class_probs (tf.Tensor):
                Softmax probabilities for potential bounding boxes.
            threshold (float, optional):
                Threshold value for minimum score for a bounding box.

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
                Max. number of bounding boxes for non-max suppression.
            iou_threshold (float, optional):
                Intersection over union threshold for non-max suppression.

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
        """Creates operations and instantiates them on default graph.

        Args:
            max_boxes (int, optional):
                Max. number of bounding boxes for non-max suppression.
            score_threshold (float, optional):
                Threshold value for min. score for a bounding box for score-filtering.
            iou_threshold (float, optional):
                Intersection over union threshold for non-max suppression.
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
        """
        image = io.imread(image_path)
        image_data = preprocess_image_cv2(image, self._model_input_dims)
        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # Need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = self._sess.run([self._scores, self._boxes, self._classes], 
                                                            feed_dict={self._model.input: image_data, 
                                                                        K.learning_phase(): 0})
        image_name = os.path.split(image_path)[-1]
        logging.info('found {} objects belonging to known classes'.format(len(out_boxes)))
        draw_boxes_cv2(image, out_scores, out_boxes, out_classes, self._class_names, self._colors)
        logging.info('saving result in `images/out/{}`'.format(image_name))
        io.imsave(os.path.join('images/out', image_name), image)

    def detect_realtime(self, frame):
        """Detects objects in real-time using YOLOv2.
        
        Args:
            frame (numpy.ndarray):
                Single frame from the webcam feed to run YOLO detection on.

        Returns:
            numpy.ndarray:
                Output frame data after detection and drawing bounding boxes over it.
        """
        image_data = preprocess_image_cv2(frame, self._model_input_dims)
        out_scores, out_boxes, out_classes = self._sess.run([self._scores, self._boxes, self._classes], 
                                                            feed_dict={self._model.input: image_data, 
                                                                        K.learning_phase(): 0})
        draw_boxes_cv2(frame, out_scores, out_boxes, out_classes, self._class_names, self._colors)
        return frame

    def __del__(self):
        self._sess.close()
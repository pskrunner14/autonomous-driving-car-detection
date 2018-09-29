import os
import click
import logging

import keras
import keras.backend as K
import tensorflow as tf
import scipy.misc as misc

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

    if realtime:
        logging.info('This project currently only supports detection on images.')

    image_shape = tuple([float(x) for x in misc.imread(image_path).shape[:-1]])
    anchors = read_anchors('model_data/yolo_anchors.txt')
    class_names = read_classes('model_data/coco_classes.txt')

    yolo = YOLO(
        model_path='model_data/yolo_model.h5',
        dims=image_shape,
        anchors=anchors,
        class_names=class_names
    )
    _, out_scores, out_boxes, out_classes = yolo.detect_image(image_path)
    # print(out_scores)
    # print(out_boxes)
    # print(out_classes)
    # print(class_names[14])
    """ Use the outputs of YOLO for make real-time detections in OpenCV """

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
            threshold (float):
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
             max_boxes (int):
                Max. number of bounding boxes for non-max 
                suppression.
            iou_threshold (float):
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
            max_boxes (int):
                Max. number of bounding boxes for 
                non-max suppression.
            score_threshold (float):
                Threshold value for min. score for 
                a bounding box for score-filtering.
            iou_threshold (float):
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
        """Detects objects in image using YOLOv2.
        
        Args:
            image_path (str):
                Path to image for detection.

        Returns:
            numpy.ndarray:
                Output image data after detection
                and drawing bounding boxes over it.
            numpy.ndarray:
                Output bounding boxes' scores.
            numpy.ndarray:
                Output bounding boxes' corner values.
            numpy.ndarray:
                Output bounding boxes' class probabalities.
        """
        image, image_data = preprocess_image(image_path, model_image_size = (608, 608))
        sess = K.get_session()
        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # Need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = sess.run([self._scores, self._boxes, self._classes], 
                                                    feed_dict={self._model.input: image_data, 
                                                                K.learning_phase(): 0})
        image_name = os.path.split(image_path)[-1]
        logging.info('Found {} objects belonging to known classes'.format(len(out_boxes), image_name))
        
        colors = generate_colors(self._class_names)
        draw_boxes(image, out_scores, out_boxes, out_classes, self._class_names, colors)
        image.save(os.path.join('images/out', image_name), quality=90)
        
        return image, out_scores, out_boxes, out_classes

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('EXIT')
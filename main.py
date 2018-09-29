import os
import click

import keras
import keras.backend as K
import tensorflow as tf
import scipy.misc as misc

from matplotlib.pyplot import imshow

# local imports
from .utils import (
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
@click.argument('image_path', type=click.Path(exists=True))
def main(image_path):

    sess = K.get_session()

    class_names = read_classes('model_data/coco_classes.txt')
    anchors = read_anchors('model_data/yolo_anchors.txt')
    image_shape = (720., 1280.)

    model = keras.models.load_model('model_data/yolo.h5')

    yolo_outputs = yolo_head(model.output, anchors, len(class_names))
    scores, boxes, classes = YOLO.eval(yolo_outputs, image_shape)
    out_scores, out_boxes, out_classes = YOLO.predict(sess, image_path)

class YOLO():

    @staticmethod
    def filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
        
        box_scores = box_confidence * box_class_probs           # Compute box scores
        
        # Find the box_classes thanks to the max box_scores, keep track of the corresponding score
        box_classes = K.argmax(box_scores, axis=-1)             # index of max score
        box_class_scores = K.max(box_scores, axis=-1)           # actual max score
        
        # Create a filtering mask based on 'box_class_scores' 
        # by using 'threshold'. The mask should have the same 
        # dimension as box_class_scores, and be True for the s
        # boxes you want to keep (with probability >= threshold)
        filtering_mask = box_class_scores >= threshold
        
        # Apply the mask to scores, boxes and classes
        scores = tf.boolean_mask(box_class_scores, filtering_mask)
        boxes = tf.boolean_mask(boxes, filtering_mask)
        classes = tf.boolean_mask(box_classes, filtering_mask)
        
        return scores, boxes, classes

    @staticmethod
    def non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):

        max_boxes_tensor = K.variable(max_boxes, dtype='int32')             # tensor to be used in tf.image.non_max_suppression()
        K.get_session().run(tf.variables_initializer([max_boxes_tensor]))   # initialize variable max_boxes_tensor
        
        # To get the list of indices corresponding to boxes you keep
        nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=iou_threshold)
        
        # To select only nms_indices from scores, boxes and classes
        scores = K.gather(scores, nms_indices)
        boxes = K.gather(boxes, nms_indices)
        classes = K.gather(classes, nms_indices)
        
        return scores, boxes, classes

    @staticmethod
    def eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):

        # Retrieve outputs of the YOLO model
        box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

        # Convert boxes to be ready for filtering functions 
        boxes = yolo_boxes_to_corners(box_xy, box_wh)

        # Perform Score-filtering with a threshold of score_threshold
        scores, boxes, classes = YOLO.filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
        
        # Scale boxes back to original image shape.
        boxes = scale_boxes(boxes, image_shape)

        # Perform Non-max suppression with a threshold of iou_threshold
        scores, boxes, classes = YOLO.non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
        
        return scores, boxes, classes

    @staticmethod
    def detect(sess, image_path):
        """Detect objects in image using YOLO.
        
        Args:
            sess (tf.Session):
                Session for Keras `backend`
            image_path (str):
                Path to image for detection.
        """
        # Preprocess your image
        image, image_data = preprocess_image('images/test/' + image_path, model_image_size = (608, 608))

        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # Need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

        # Print predictions info
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))
        # Generate colors for drawing bounding boxes.
        colors = generate_colors(class_names)
        # Draw bounding boxes on the image file
        draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
        # Save the predicted bounding box on the image
        image.save(os.path.join('images/out', image_file), quality=90)
        # Display the results in the notebook
        output_image = misc.imread(os.path.join('images/out', image_file))
        imshow(output_image)
        
        return out_scores, out_boxes, out_classes

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('EXIT')
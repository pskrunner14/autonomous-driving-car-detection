import click
import logging

import cv2
import imageio as io

from yolo import YOLO

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
        logging.info('testing webcam frame dimensions')
        image_shape = get_cam_dims()
    else:
        image_shape = tuple([float(x) for x in io.imread(image_path).shape[:-1]])

    logging.info('loading YOLOv2 Darknet19 model')
    yolo = YOLO(
        model_path='model_data/yolo_model.h5',
        anchors_path='model_data/yolo_anchors.txt',
        classes_path='model_data/coco_classes.txt',
        dims=image_shape
    )

    if realtime:
        """ Making real-time detections in OpenCV """
        logging.info('starting webcam for real-time detections')
        realtime_object_detector(yolo)
    else:
        """ Detect objects in image and save the results """
        logging.info('detecting objects in `{}`'.format(image_path))
        yolo.detect_image(image_path)

def realtime_object_detector(yolo=None):
    """Makes real-time object detections using cam feed.

    Args:
        yolo (YOLO):
            YOLO object to use for detection.

    Raises:
        ValueError: If `yolo` is None.
    """
    if yolo is None:
        raise ValueError('YOLO object not found.')

    logging.info('Press ESC to exit')
    cv2.namedWindow('detector')
    vc = cv2.VideoCapture(0)
    
    while vc.isOpened():
        _, frame = vc.read()
        frame = yolo.detect_realtime(frame)
        cv2.imshow('detector', frame)

        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            logging.info('exiting')
            break

    logging.info('destroying detector window')
    cv2.destroyWindow('detector')

def get_cam_dims():
    """Returns the webcam frame dimensions.
    
    Returns:
        tuple of `float`:
            Frame dimensions of webcam using OpenCV.
    """
    cv2.namedWindow('test-shape')
    vc = cv2.VideoCapture(0)
    while vc.isOpened():
        _, frame = vc.read()
        image_shape = tuple([float(x) for x in frame.shape[:-1]])
        break
    del vc
    cv2.destroyWindow('test-shape')
    return image_shape

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('EXIT')
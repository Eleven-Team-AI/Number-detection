import torch
import logging
import video_processing
from datetime import datetime
import pytz
import numpy as np
import cv2
import easyocr
import re
import gc
from IPython.core.display import clear_output
from PIL import Image

log = logging.getLogger('enter_system')

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
plat_yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/plat_model.pt', force_reload=True)


# TODO check parameter for load in function?
def car_detection(img):
    results = yolo_model(img)
    labels = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    names = yolo_model.names
    detected = '-'
    for label in labels:
        detected = names[label]
    return detected


def plat_detection(img):
    results = plat_yolo_model(img)
    labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    names = plat_yolo_model.names
    detected = '-'
    for label in labels:
        detected = names[label]
        if detected == 'plat-nomor':
            return cord_thres
    return None


def process_image(path: str) -> np.array:
    """
    Function for preprocessing image for OCR recognition
    :param path: path to image
    :return: numpy array
    """
    image = cv2.imread(path)
    image = cv2.resize(image, (0, 0), fx=3, fy=3)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 50, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    return invert


def ocr_recognition(path: str) -> str:
    """
    Function for OCR text from picture
    :param path: image path
    :return: recognized text
    """
    data = easyocr.Reader(['ru'], gpu=False)
    text = data.readtext(process_image(path))
    text = re.sub(r"[' :;~=+)()-]", '', text[0][1]).lower()

    return text


def masking_video(path_video: str,
                  path_masked_video: str,
                  frame_size: tuple,
                  frame_rate: int,
                  coord: list) -> None:
    """
    Function for masking and saving video
    :param path_video: path to video
    :param path_masked_video: out path
    :param frame_size: frame size
    :param frame_rate: frame rate
    :param coord: coordinates of mask [(x1, y1), (x2, y2)...]
    :return: None
    """
    mask = video_processing.create_mask(frame_size, coord)
    frames = video_processing.video_to_array_optimizing(path_video, frame_size, 3)
    mask_frames = video_processing.apply_mask(frames, mask)

    video_processing.array_to_video(mask_frames,
                                    path_masked_video,
                                    'mp4v',
                                    frame_rate,
                                    frame_size)
    gc.collect()
    clear_output()


def crop_image(image: np.array, coord: list) -> np.array:
    """
    Function for crop the image
    :param image: image as np.array
    :param coord: list
    :return: np.array
    """
    img = Image.fromarray(image)
    img_crop = img.crop(coord[0], coord[1], coord[2], coord[3])

    return np.array(img_crop)


def main_loop(video):
    # mask_video = 
    # TODO: add mask on video 
    detection = car_detection(video)
    if detection == 'car' or detection == 'truck':
        moscow_time = datetime.now(pytz.timezone('Europe/Moscow'))
        log.info(f'{detection.upper()} detected, start recording')
        # TODO start saving video

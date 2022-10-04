import torch
import logging
import video_processing
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pytz
import torch
import yaml
from IPython.core.display import clear_output
from PIL import Image

import video_processing
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import check_file

log = logging.getLogger('enter_system')

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
plat_yolo_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                 path=os.path.join(Path(__file__).parents[0],
                                                   'models', 'plat_model'),
                                 force_reload=True)


def car_detection(frame):
    results = yolo_model(frame)
    detections = results.pred[0][:, 5].unique()
    labels = detections.cpu().numpy()
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


def worker():
    with open(os.path.join(Path(__file__).parents[0], 'config', 'model.yaml')) as file:
        config = yaml.load(file, Loader=yaml.Loader)
    # masked_frames = masking_video(path_video=config['constants']['path_video'],
    #                               frame_size=tuple(config['constants']['frame_size']),
    #                               coord=config['constants']['coord'])
    source = check_file('/second_4tb/kuchuganova/other/Number-detection/Enter_system/src/record.mp4')
    masked_frames = LoadImages(source, img_size=(640, 640), stride=yolo_model.stride, auto=yolo_model.pt, vid_stride=1)

    for path, im, im0s, vid_cap, s in masked_frames:
        detection = car_detection(im)
        if detection == 'car' or detection == 'truck':
            print('detect car')
            moscow_time = datetime.now(pytz.timezone('Europe/Moscow'))
            # TODO красивый вывод в лог
            log.info(f'{detection.upper()} detected, start recording')
            # TODO start saving video
            # делать через out и просто write фрэймов
            coord = plat_detection(im)
            # TODO: coord crop
            recognized_code = ocr_recognition()
            log.info(f'Car - {recognized_code[:5]}')
            with open(config['constant']['path_enter_car']) as file:
                cars = file.readlines()
            if recognized_code not in cars:
                print('ALARM: CHECK CAR!!!!!!')
            else:
                log.info(f'Car - {recognized_code[:5]} - in passed')


if __name__ == '__main__':
    worker()

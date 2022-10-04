import torch
import logging
from .video_processing import apply_mask 
from datetime import datetime
import pytz   

log = logging.getLogger('enter_system')

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
plat_yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/plat_model.pt', force_reload=True)

# TODO check parametr for load in function?
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

def main_loop(video):
    # mask_video = 
    # TODO: add mask on video 
    detection = car_detection(video)
    if detection == 'car' or detection == 'truck':
        moscow_time = datetime.now(pytz.timezone('Europe/Moscow'))
        log.info(f'{detection.upper()} detected, start recording')
        # TODO start saving video 
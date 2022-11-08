import cv2
import numpy as np
import time
from PIL import Image
from PIL import ImageDraw
from typing import MutableSequence

def video_to_array_optimizing(path_video: str, frame_size: tuple, k: int) -> MutableSequence:
    """
    Function for grab video and making frames array
    :param frame_size: size of frames in array
    :param k: compress ratio
    :param path_video: path of video file
    :return: array of frames
    """
    start = time.monotonic()

    video_capture = cv2.VideoCapture()
    video_capture.open(path_video)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))

    out = []

    for i in range(0, int(frames)):
        frame_id = int(video_capture.get(1))
        ret, frame = video_capture.read()
        if frame_id % k == 0:
            b = cv2.resize(frame, (frame_size), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.append(b)

    out = np.array(out)
    print(f'Shape of frames array:{out.shape}')

    finish = time.monotonic() - start
    print("Program time video_to_array: {:>.3f}".format(finish) + " seconds.")
    return out


def create_mask(shape_image: tuple, coord_point_list: list) -> np.array:
    """
    Function for creating mask array for frame
    :param shape_image: size of frame
    :param coord_point_list: list tuples of coord. Ex: [(x1,y1),(x2,y2)...]
    :return: mask array
    """
    coord_point_list = [tuple(elem) for elem in coord_point_list]
    img = Image.new('L', shape_image, color=255)
    transparent_a = (0, 0, shape_image[0], shape_image[1])
    draw = ImageDraw.Draw(img, mode='L')

    draw.rectangle(transparent_a, fill=1)
    draw.polygon(xy=(coord_point_list), fill='Black')

    img = img.convert('RGB')
    return np.asarray(img)


def apply_mask(x: MutableSequence, mask: MutableSequence) -> np.array:
    """
    This function apply mask array to frame's
    :param x: array of frames
    :param mask: mask array
    :return: masked array
    """
    start = time.monotonic()
    mask_frames = x * mask
    finish = time.monotonic() - start
    print("Program time apply_mask: {:>.3f}".format(finish) + " seconds.")
    return mask_frames
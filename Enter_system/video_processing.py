import cv2
import glob
import numpy as np
import os.path
import time
from PIL import Image
from PIL import ImageDraw
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from typing import MutableSequence


def video_to_frames(path_video: str, path_frames: str) -> None:
    """
    Function for grab video, making and save frames
    :param path_video: path of video file
    :param path_frames: path for save frames
    :return: None
    """
    start = time.monotonic()

    video_capture = cv2.VideoCapture()
    video_capture.open(path_video)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))

    for i in range(int(frames)):
        ret, frame = video_сapture.read()
        cv2.imwrite(os.path.join(path_frames, 'frames00%d.jpg' % (i), frame))

    finish = time.monotonic() - start
    print("Program time video_to_frames: {:>.3f}".format(finish) + " seconds.")


def frames_to_video(path: str,
                    name: str,
                    codec: str,
                    frame_rate: int,
                    frame_size: tuple) -> None:
    """
    Function for make a video file from frames
    :param path: path with frames
    :param name: name of new video file
    :param codec: name of codecs (https://www.fourcc.org/codecs/)
    :param frame_rate: fps of video file
    :param frame_size: (width, height)
    :return: None
    """
    start = time.monotonic()

    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*codec), frame_rate, frame_size)

    for filename in sorted(glob.glob(path + '*.jpg'), key=os.path.getmtime):
        img = cv2.imread(filename)
        img = cv2.resize(img, dsize=frame_size)
        out.write(img)

    out.release()
    cv2.destroyAllWindows()

    finish = time.monotonic() - start
    print("Program time frames_to_video: {:>.3f}".format(finish) + " seconds.")


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


def video_to_array_markup(path_video: str, frame_size: tuple, begin: int, end: int) -> np.array:
    """
    Function for grab video and making frames array from
    desired range of frames
    :param begin: number of frame where starting action
    :param end: number of frame where ending action
    :param frame_size: size of frames in array
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
        if frame_id in range(begin, end):
            b = cv2.resize(frame, (frame_size), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.append(b)

    out = np.array(out)
    print(f'Shape of frames array:{out.shape}')

    finish = time.monotonic() - start
    print("Program time video_to_array: {:>.3f}".format(finish) + " seconds.")
    return out


def array_compression(x: MutableSequence, k: int) -> MutableSequence:
    """
    Function for compress array. It takes every k element of array
    and make new array from these elements
    :param x: array for compress
    :param k: compress ratio
    :return: compressed array
    """
    start = time.monotonic()

    out = np.array([x[i] for i in range(0, x.shape[0], k)])
    print(f'New array dimension: {out.shape}')

    finish = time.monotonic() - start
    print("Program time array_compression: {:>.3f}".format(finish) + " seconds.")
    return out


def array_to_video(x: MutableSequence,
                   name: str,
                   codec: str,
                   frame_rate: int,
                   frame_size: tuple) -> None:
    """
    Function for make a video file from array
    WARNING: If you used array compression function,
    use corresponding frame rate, for saving video speed. For example:
    original frame rate was: 25, compression:2, use frame rate:12
    :param x: frames array
    :param name: name of new video file
    :param codec: name of codecs (https://www.fourcc.org/codecs/)
    :param frame_rate: fps of video file
    :param frame_size: (width, height)
    :return: None
    """
    start = time.monotonic()
    out = cv2.VideoWriter(name,
                          cv2.VideoWriter_fourcc(*codec),
                          frame_rate, frame_size)

    for element in x:
        img = cv2.resize(element, dsize=frame_size)
        out.write(img)

    out.release()
    cv2.destroyAllWindows()

    finish = time.monotonic() - start
    print("Program time array_to_video: {:>.3f}".format(finish) + " seconds.")


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


def cut_video(path_video: str,
              start_time: int,
              end_time: int,
              path_out: str) -> None:
    """
    Function for cutting video file.
    NOTE: Need to install moviepy: pip3 install moviepy
    :param path_video: path + filename video file
    :param start_time: start time in second
    :param end_time: end time in second
    :param path_out: path + filename of new video file
    :return: None
    """
    ffmpeg_extract_subclip(path_video, start_time, end_time, targetname=path_out)

import cv2
import os
import numpy as np
import ffmpeg
from ..video.Image import Image


def process_ffmpeg(frame, saving_file_name, fps=30):
    i_height, i_width, _ = frame.shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(i_width, i_height))
        .output(saving_file_name, pix_fmt='yuv420p', vcodec='libx264', r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    return process


class VideoProcessor:
    def __init__(self) -> None:
        pass

    def generate_video_from_frames(self, frame_folder: str, saved_path: str, flags=cv2.IMREAD_COLOR, fps=30):
        """
        Parameters
        ----------
        frame_folder : str
            Path to the folder containing frames
        saved_path : str
            Path to the saved video
        """
        images = [img for img in os.listdir(frame_folder) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(frame_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(saved_path, 0, 30, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(frame_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        return

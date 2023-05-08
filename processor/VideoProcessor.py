import cv2
import os
import numpy as np
import ffmpeg
from ..video.Video import Video
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

    def generate_video_from_frames(self, frame_folder: str, saved_path: str, fps=30):
        """
        Parameters
        ----------
        frame_folder : str
            Path to the folder containing frames
        saved_path : str
            Path to the saved video
        """
        current_path = os.getcwd()
        os.chdir(frame_folder)
        next_path = os.getcwd()
        frames = [Image.load_image(path_img).img for path_img in os.listdir(next_path)
              if path_img.endswith(".jpg") or
              path_img.endswith(".jpeg") or
              path_img.endswith(".png")]
        os.chdir(current_path)
        
        cap = cv2.VideoCapture()
        process = process_ffmpeg(frames[0], saved_path, fps)

        for frame in frames:
            process.stdin.write(
                frame.astype(np.uint8)
                    .tobytes()
            )

        process.stdin.close()
        process.wait()
        cap.release()
        return
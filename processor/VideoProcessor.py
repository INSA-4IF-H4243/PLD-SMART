import cv2
import os
import numpy as np
import ffmpeg
#from ..video.Video import Video
#from ..video.Image import Image
from video.Video import Video
from video.Image import Image

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

    def generate_crop_video(self, path_video: str, output_path: str,
                                     start_x: int, end_x: int,
                                        start_y: int, end_y: int, 
                       start_frame: int = 0, end_frame: int = 30):
        """
        Parameters
        ----------
        path_video : str
            Path to the video
        output_path : str
            Path to the output video
        start_x : int
            Starting x coordinate
        end_x : int
            Ending x coordinate
        start_y : int
            Starting y coordinate
        end_y : int
            Ending y coordinate
        start_frame : int, optional
            Starting frame, by default 0
        end_frame : int, optional
            Ending frame, by default 30
        """
        if not os.path.exists(path_video):
            raise FileNotFoundError("The file {} does not exist".format(path_video))
        vid = Video.read_video(path_video)
        frames = vid.frames[start_frame:end_frame]
        new_frames = []
        for i in range(len(frames)):
            new_frames.append(frames[i][start_y:end_y, start_x:end_x])
        new_frames = np.array(new_frames)
        vid.save_video(new_frames, output_path)
        return

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
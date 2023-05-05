import os

from processor import ImageProcessor, VideoProcessor
from video import Video

video_processor = VideoProcessor()
image_processor = ImageProcessor()

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(project_path, 'dataset', 'service_j2', 'cut3_sJEWRBgM.mp4')
output_path = os.path.join(project_path, 'img', 'djo_service_cropped')

video = Video.read_video(video_path)
frames = video.frames
cropped_frames = []
#image_processor.save_img(frames[0], os.path.join(output_path, "frame0.png"))
cropped_frames_rembg = []
for frame in frames:
    f = image_processor.normalize_img(frame)
    cropped_frames.append(image_processor.crop_image(f, 590, 645, 64, 171))
for i in range(len(cropped_frames)):
    output = os.path.join(output_path, "frame{}.png".format(i))
    image_processor.save_img(image_processor.remove_background_2(cropped_frames[i]), output)

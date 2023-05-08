from smart.video import Video
import numpy as np
import os

for pathdir, listdirs, _ in os.walk('shadow_test'):
    for i, dir in enumerate(listdirs):
        path_out = os.path.join('shadow_full_output', dir)
        
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        path_in = os.path.join(pathdir, dir)
        for cpt, file in enumerate(os.listdir(path_in)):
            path_file = os.path.join(path_in, file)
            print(path_file)
            video = Video.read_video(path_file)
            vid_no_ext = file.split('.')[0]
            nb_frames = len(video.frames)
            list_nb_frames = np.linspace(0, nb_frames-1, 15, dtype=int)
            video.save_some_frames(list_nb_frames, 'shadow_full_output/{}/{}'.format(dir, vid_no_ext))
import os
from smart.video import Image
import numpy as np
import cv2
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='forehand')
args = parser.parse_args()

array_lab = [f'frame{i}pixel{k}' for i in range(15) for k in range(400)]
array_val = np.zeros(0, dtype=int)
cpt = 0

print("============== Passed files ==============")
print()
for dirpath, dirnames, _ in os.walk(args.folder):
    for dir in dirnames:
        path_fol = os.path.join(dirpath, dir)
        print(path_fol)
        array_vid = np.zeros(0, dtype=int)
        for file in os.listdir(path_fol):
            path_file = os.path.join(path_fol, file)
            print(path_file)
            if not os.path.isfile(path_file):
                continue
            img = Image.load_image(path_file).img
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (20,20), interpolation=cv2.INTER_BITS)
            _, binaire_img = cv2.threshold(resized_img, 0, 1, cv2.THRESH_BINARY)
            array_bin = np.array(binaire_img, dtype=int).flatten()
            array_vid = np.concatenate((array_vid, array_bin))
        array_val = np.concatenate((array_val, array_vid))
        cpt += 1

array_val = np.reshape(array_val, (cpt, 6000))
dic = dict.fromkeys(array_lab, 0)
for i, key in enumerate(dic):
    dic[key] = array_val[:, i]

df = pd.DataFrame(dic)
df['type'] = args.folder
df.to_csv(f'{args.folder}.csv', index=False)
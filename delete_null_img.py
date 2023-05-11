import os
from smart.video import Image
import subprocess
import sys
import numpy as np
import cv2

def command(string, logfile=None):
    """execute `string` as a shell command, optionnaly logging stdout+stderr to a file. return exit status.)"""
    try:
        output=subprocess.check_output(string,stderr=subprocess.STDOUT,shell=True)
        ret= 0
    except subprocess.CalledProcessError as e:
        ret=e.returncode
        output = e.output
    if logfile:
        f=open(logfile,'w')
        print(output.decode(sys.stdout.encoding)+'\n'+'return code: '+str(ret),file=f)
        f.close()
    return ret

print("=================== Type de coup scanne ===================")
print()
list_fol_null = []
for dirpath, dirnames, _ in os.walk('shadow_full_output'):
    for dir in dirnames:
        path_type = os.path.join(dirpath, dir)
        print(path_type)
        for folder in os.listdir(path_type):
            path_fol = os.path.join(path_type, folder)
            if not os.path.isdir(path_fol):
                continue
            sum_img = np.zeros((480, 640))
            for file in os.listdir(path_fol):
                path_file = os.path.join(path_fol, file)
                img = Image.load_image(path_file).img
                cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sum_img += cvt_img
            if np.sum(sum_img) == 0:
                list_fol_null.append(path_fol)

print()
print("=================== Folder to delete ===================")
print()
for fol in list_fol_null:
    print(fol)
    command('rm -rf {}'.format(fol))
                
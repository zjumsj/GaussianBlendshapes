import argparse
import os, glob, zipfile
from tqdm import tqdm

def images_to_video(path, fps=25, video_format='DIVX'):
    import cv2
    img_array = []
    for filename in tqdm(sorted(glob.glob(f'{path}/*.png'))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        out = cv2.VideoWriter(f'{path}/video.avi', cv2.VideoWriter_fourcc(*video_format), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

#############################
#
#   Parser
#
##############################

## support "--X True/False" in command line
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#############################
#
#   Record
#
##############################

def dump_code(current_path, tar_path):
    # root/*.py
    dump_files = glob.glob(os.path.join(current_path, '*.py'))
    # root/config/*.py
    dump_files = dump_files + glob.glob(os.path.join(os.path.join(current_path, "config"), '*.py'))
    # root/FLAME/*
    dump_files = dump_files + glob.glob(os.path.join(os.path.join(current_path, "FLAME"), "*"))
    # root/utils/*.py
    dump_files = dump_files + glob.glob(os.path.join(os.path.join(current_path, "utils"), "*.py"))

    zf = zipfile.ZipFile(os.path.join(tar_path, 'source_code.zip'), mode='w')
    try:
        for f in dump_files:
            zf.write(f)
    finally:
        zf.close()

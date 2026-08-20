import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.patches as patches
import cv2  
import argparse
from pathlib import Path

def pad_to_size(img, target_width, target_height, color=(0, 0, 0)):
    h, w = img.shape[:2]

    if w > target_width or h > target_height:
        raise ValueError("Image larger than target size")

    pad_left   = (target_width  - w) // 2
    pad_right  = target_width  - w - pad_left
    pad_top    = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top

    padded = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )

    return padded

def main_function(sub, output_dir, verbose=False):

    sub_output_dir = os.path.join(output_dir, sub)
    sub_frames_dir = os.path.join(sub_output_dir, f'img_output')
    target_width  = 839
    target_height = 484

    frames_paths = []
    for file in os.listdir(os.path.join(sub_frames_dir, sub)):
        if file.endswith('.png'):
            frames_paths.append(os.path.join(sub_frames_dir, sub, file))
    frames_timepoints = [float(os.path.basename(fp).split('/')[-1].split('.png')[0]) for fp in frames_paths]
    sorted_indices = np.argsort(frames_timepoints)
    frames_paths = [frames_paths[i] for i in sorted_indices]

    # Load each image and make it a video
    frame_list = []
    for frame_path in frames_paths:
        print(frame_path)
        img1 = cv2.imread(frame_path)
        # print img width and height
        try:
            img = pad_to_size(img1, target_width, target_height)
        except ValueError as e:
            print(img1.shape)

        frame_list.append(img)
    # Make each frame last for 30 ms (i.e., 30 fps)
    out = cv2.VideoWriter(os.path.join(sub_output_dir, f"{sub}.avi"), cv2.VideoWriter_fourcc(*'MJPG'), 10, (target_width, target_height))
    for i in range(len(frame_list)):
        out.write(frame_list[i])
    out.release()
    if verbose:
        print(f"Video saved at {os.path.join(sub_output_dir, f'{sub}.avi')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default=None, help='Subject ID (e.g., sub-001). If None, it will iterate through all subject directories in data_dir.')
    parser.add_argument('--output_dir', type=str, default=Path('output'), help='Directory containing the output data for each subject.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print verbose output.')
    args = parser.parse_args()

    # Modify data_dir and output_dir based on validation flag
    args.output_dir = args.output_dir if not args.validation else os.path.join(args.output_dir, 'validation')

    if args.sub is None:
        subs = [f for f in os.listdir(args.output_dir) if 'sub-' in f]
        print(f'Since sub is None, we iterate over subject directories in {args.output_dir}: {subs}')
        if len(subs) == 0:
            print(f'No proper subject directories found in {args.output_dir}. Make sure they follow the naming convention "sub-00X"')
    elif isinstance(args.sub, str):
        subs = [args.sub]
    else:
        raise ValueError('For sub, please provide a string (e.g., sub-001) or None.')
    for curr_sub in subs:
        main_function(sub=curr_sub, output_dir=args.output_dir, verbose=True) #args.verbose)

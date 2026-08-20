
import numpy as np
import matplotlib.pyplot as plt


import os
from scipy.ndimage import binary_dilation




def val_fixation_cross(video_start_data, curr_exp_time, w, h):
    '''
    Given the current experimental timepoint, determine whether there was a fixation point by 
    calculating the difference between current timepoint and each of the onset times of the videos.
    If this difference is past the threshold (9 fixation crosses x 10 seconds each = 60s), return None.
    Otherwise, return location of fix cross for plotting.
    '''
    num_fix_crosses = 9
    secs_per_fix_cross = 10
    video_onset_times = list(video_start_data['onset'])
    for onset_time in video_onset_times:
        sec_diff = round(curr_exp_time - onset_time, 4)
        if sec_diff <  num_fix_crosses * secs_per_fix_cross and sec_diff >= 0:
            cross = None
            if sec_diff < secs_per_fix_cross * 1:
                cross = (w/4, h/4)
            elif sec_diff < secs_per_fix_cross * 2:
                cross = (w/4, h/4 * 2)
            elif sec_diff < secs_per_fix_cross * 3:
                cross = (w/4, h/4 * 3)
            elif sec_diff < secs_per_fix_cross * 4:
                cross = (w/4 * 2, h/4)
            elif sec_diff < secs_per_fix_cross * 5:
                cross = (w/4 * 2, h/4 * 2)
            elif sec_diff < secs_per_fix_cross * 6:
                cross = (w/4 * 2, h/4 * 3)
            elif sec_diff < secs_per_fix_cross * 7:
                cross = (w/4 * 3, h/4)
            elif sec_diff < secs_per_fix_cross * 8:
                cross = (w/4 * 3, h/4 * 2)
            else:
                cross = (w/4 * 3, h/4 * 3)
            return cross, sec_diff
    return None, None



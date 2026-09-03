import argparse
import os
from pathlib import Path
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2 

# ---- Utility functions ---

def gaussian_weight_map(mask, nsc_vec, sigma=100, plot=False):
    H, W   = mask.shape
    gauss   = np.zeros((H, W), dtype=np.float32)
    gauss[int(nsc_vec[1]), int(nsc_vec[0])] = 1.0
    gauss   = gaussian_filter(gauss, sigma=sigma)  #
    weights = gauss * mask         
    weights /= weights.sum()       
    if plot:
        plt.imshow(weights, cmap='inferno')
        plt.colorbar()
        plt.title('Gaussian Weight Map Centered at NSC within FOV Mask')
        plt.axis('off')
        plt.show()
    return weights

def colorfulness(R,G,B):
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return round(colorfulness, 3)


def color_complexity(masked_img, n_bins=32):
    ''' 
    Calculates color complexity (Liu et al., 2025) as the Shannon entropy of the color distribution within given FOV mask.
    '''
    # Quantize each channel into n_bins levels
    pixels = masked_img.astype(np.float64) 
    bin_edges = np.linspace(0, 256, n_bins + 1)
    quantized = np.digitize(pixels, bin_edges[1:-1], right=False)  # shape (N, 3), values in [0, n_bins-1]

    # Map each quantized (R, G, B) triplet to a single integer "color index"
    color_indices = quantized[:, 0] * n_bins**2 + quantized[:, 1] * n_bins + quantized[:, 2]

    # Count occurrences of each distinct color (this gives M and the p(k) distribution)
    counts = np.bincount(color_indices, minlength=n_bins**3)
    counts = counts[counts > 0]   # drop empty bins before taking log
    p_k = counts / counts.sum()

    cc_image = -np.sum(p_k * np.log(p_k))
    return cc_image

def compute_visual_statistics(img, fov_mask, gaze_fixation, weighted_average=False):

    visual_stats = {}

    # Crop to mask bounding box to save computation time
    rows = np.any(fov_mask, axis=1)
    cols = np.any(fov_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 2  # small margin so Laplacian isn't starved of neighbors at the crop edge
    rmin, rmax = max(0, rmin - pad), min(img.shape[0], rmax + pad + 1)
    cmin, cmax = max(0, cmin - pad), min(img.shape[1], cmax + pad + 1)
    img = img[rmin:rmax, cmin:cmax]
    fov_mask = fov_mask[rmin:rmax, cmin:cmax]

    # Build Gaussian filter to build weighted averages
    if weighted_average:   
      weights =  gaussian_weight_map(fov_mask, gaze_fixation)
    
    # Luminance
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    visual_stats['fov_luminance'] = round(np.mean(img_gray[fov_mask]), 3)

    # Constrast
    visual_stats['fov_contrast'] = round(np.std(img_gray[fov_mask]), 3)

    # Saturation
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue_rad = img_hsv[fov_mask, 0].astype(np.float64) * 2 * np.pi / 180.0  # OpenCV H (0-179) -> radians (0-2pi)
    mean_hue_rad = np.arctan2(np.mean(np.sin(hue_rad)), np.mean(np.cos(hue_rad)))
    mean_hue_opencv = (np.degrees(mean_hue_rad) % 360) / 2  # back to OpenCV's 0-179 scale
    visual_stats['fov_hue'] = round(mean_hue_opencv, 3)
    visual_stats['fov_saturation'] = round(np.mean(img_hsv[fov_mask, 1]), 3)
    visual_stats['fov_value'] = round(np.mean(img_hsv[fov_mask, 2]), 3)

    # RGB channels
    R = img[fov_mask, 0].astype(np.float64)
    G = img[fov_mask, 1].astype(np.float64)
    B = img[fov_mask, 2].astype(np.float64)
    visual_stats['fov_R_mean'] = round(np.mean(R), 3)
    visual_stats['fov_G_mean'] = round(np.mean(G), 3)
    visual_stats['fov_B_mean'] = round(np.mean(B), 3)

    # Colorfulness (Hasler & Süsstrunk, 2003)
    visual_stats['fov_colorfulness'] = colorfulness(R,G,B)

    # Color complexity (Liu et al., 2025)
    visual_stats['fov_color_complexity'] = color_complexity(img[fov_mask])

    # Clarity / edge sharpness (Liu et al., 2025)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    visual_stats['fov_clarity'] = round(np.mean(np.abs(laplacian[fov_mask])), 3)

    return visual_stats

# ---- Main function ---

def main_function(sub, validation, fps, chunk_size, stimuli_dir, output_dir, verbose=False):
    '''
    Loads precomputed FOV frustum and gaze fixation for each video frame, and calculates visual statistics. 
    Input: 
        sub: str
            Subject ID (e.g., sub-001).
        validation: bool
            Whether to run in validation mode, using validation video frames instead of the main experimental stimuli. 
        fps: int
            FPS at which data will be trimmed to match extracted video frames.
        chunk_size: int or None
            Size of chunks of mask history. If None, the entire mask history will be loaded from a single file.
        stimuli_dir: str or Path
            Directory containing the video frames.
        output_dir: str or Path
            Directory containing the output data for each subject, including pre-computed frustum mask and gaze fixation history.
        verbose: bool
            Whether to print verbose output.
    Output:
        Saves the visual statistics to a CSV file in the output directory.
    '''

    # Directories
    sub_data_dir = os.path.join(output_dir, sub)
    mask_path = os.path.join(sub_data_dir, f'{sub}_mask-history_{fps}-fps.npz' if chunk_size is None else f'{sub}_mask-history_{fps}-fps_chunked')
    gaze_path = os.path.join(sub_data_dir, f'{sub}_gaze-history_{fps}-fps.npy')
    visual_stats_path = os.path.join(sub_data_dir, f'{sub}_visual-stats-history_{fps}-fps.csv')
    validation_str = 'validation_' if validation else ''
    video_frames_dir = os.path.join(stimuli_dir, f'{validation_str}video_frames_{fps}FPS')
    print(f'Computing visual statistics for subject {sub} {validation_str} using following parameters: fps={fps}')
    validation_str = '(validation)' if validation else ''
    chunk_flush = chunk_size is not None

    # Check if the required files exist
    if not os.path.exists(mask_path) or not os.path.exists(gaze_path):
        print(f'Missing pre-computed frustum mask history or gaze history for subject {sub} {validation_str} at {mask_path} or {gaze_path}')
        print(f'Please run compute_frustum_mask.py for subject {sub} {validation_str} at the correct fps ({fps}) to generate the required files')
        exit(1)

    # Load pre-computed frustum mask history, gaze history, and visual statistics history if applicable
    if verbose:
        print(f'Loading pre-computed frustum mask history and gaze history for subject {sub} {validation_str} stored at {mask_path} and {gaze_path}')
    gaze_history = np.load(gaze_path, allow_pickle=True)
    if chunk_flush:
        prev_chunk_idx = 0
        chunk_history = np.load(os.path.join(mask_path, f'chunk-{prev_chunk_idx}.npz'), allow_pickle=True)
    else:
        mask_history_loaded = np.load(mask_path, allow_pickle=True)
        mask_history = {k: mask_history_loaded[k] for k in mask_history_loaded.files}
    visual_stats_history = {}
    for idx in tqdm.tqdm(range(len(gaze_history)), desc=f'Calculating visual statistics for {sub} {validation_str}'):
       # curr_exp_time = round(idx / fps, 3)
        gaze_fixation = gaze_history[idx]
        if chunk_flush:
            chunk_idx = idx // chunk_size 
            if prev_chunk_idx < chunk_idx:
                prev_chunk_idx = chunk_idx
                chunk_history = np.load(os.path.join(mask_path, f'chunk-{prev_chunk_idx}.npz'), allow_pickle=True)
            fov_mask = chunk_history.get(str(idx))
        else:
            fov_mask = mask_history.get(str(idx))

        img = np.array(Image.open(os.path.join(video_frames_dir, f'frame_{idx}.jpg')))
        visual_stats = compute_visual_statistics(img, fov_mask, gaze_fixation)
        visual_stats_history[idx] = visual_stats

    # Save visual statistics history to CSV
    visual_stats_df = pd.DataFrame.from_dict(visual_stats_history, orient='index')
    visual_stats_df.index.name = 'frame_idx'
    visual_stats_df.to_csv(visual_stats_path)
    if verbose:
        print(f'Finished calculating visual statistics for subject {sub} {validation_str}. Visual statistics saved to {visual_stats_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default=None, help='Subject ID (e.g., sub-001). If None, it will iterate through all subject directories in data_dir.')
    parser.add_argument('--validation', action='store_true', default=False, help='Whether to run in validation mode (uses validation video frames).')
    parser.add_argument('--fps', type=int, default=1, help='FPS at which data will be trimmed to match extracted video frames.')
    parser.add_argument('--chunk_size', type=int, default=None, help='Size of chunks for saving the mask history. If None, the entire mask history will be saved in a single file.')
    parser.add_argument('--stimuli_dir', type=str, default=Path('stimuli'), help='Directory containing the video frames.')
    parser.add_argument('--output_dir', type=str, default=Path('output'), help='Directory containing the output data for each subject.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print verbose output')
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
        main_function(sub=curr_sub, validation=args.validation, fps=args.fps, chunk_size=args.chunk_size, stimuli_dir=args.stimuli_dir, output_dir=args.output_dir, verbose=True) #args.verbose)

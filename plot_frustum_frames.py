import argparse
import os
from pathlib import Path
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---- Utility functions ---

NORM_RANGES = {
    'luminance':   255.0,   # grayscale mean, 0-255
    'contrast':    127.5,   # grayscale std, max possible for [0,255] data is 127.5 (bimodal)
    'hue':         179.0,   # OpenCV hue scale, 0-179 (maps to 0-360 degrees)
    'saturation':  255.0,   # OpenCV S channel, 0-255
    'value':       255.0,   # OpenCV V channel, 0-255
    'colorfulness': 109.0,  # empirical ceiling from Hasler & Süsstrunk's colorfulness categories
    'R':           255.0,
    'G':           255.0,
    'B':           255.0,
}

def plot_imgNfrustum(sub, img, mask, gaze_fixation, visual_stats, curr_exp_time, output_dir, darkening_factor, dpi):
    # Darken the image outside the frustum mask
    img[mask == 0] = img[mask == 0] * (1 - darkening_factor)
    # Plot image with frustum mask and gaze fixation
    plt.figure(figsize=(10, 10))
    ax_img = plt.gca()
    ax_img.imshow(img)
    ax_img.scatter(gaze_fixation[0], gaze_fixation[1], c='red', s=25, label='Gaze Fixation')
    ax_img.axis("off")

    # Plot visual statistics if available
    if visual_stats is not None:

        # Normalize each stat to [0, 1] for consistent bar length
        labels = visual_stats.keys()
        values = visual_stats.values()
        labels_to_remove = []
        for label in visual_stats:
            if label.split('_')[1] in NORM_RANGES:
                visual_stats[label] = visual_stats[label] / NORM_RANGES[label.split('_')[1]]
            else:
                print(f"Warning: No normalization range defined for {label}. Skipping.")
                labels_to_remove.append(label)
        for label in labels_to_remove:
            del visual_stats[label]

        # Inset axes in the top-right corner, in axes-fraction coords
        ax_bars = ax_img.inset_axes([0.83, 0.95 - (0.05 * len(labels)), 0.14, 0.05 * len(labels)])  # [x0, y0, width, height]
        ax_bars.set_facecolor('black')
        ax_bars.patch.set_alpha(0.5)

        y_pos = np.arange(len(labels))
        ax_bars.barh(y_pos, values, color='green', height=0.6)
        ax_bars.set_xlim(0, 1)
        ax_bars.set_xticks([])
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(labels, color='white', fontsize=8)
      #  ax_bars.tick_params(axis='x', colors='white', labelsize=5)
        ax_bars.invert_yaxis()  # first stat on top

        for spine in ax_bars.spines.values():
            spine.set_visible(False)
      
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'{sub}_timestamp-{curr_exp_time}.png'), bbox_inches='tight', dpi=dpi)
    plt.close()

# ---- Main function ---

def main_function(sub, validation, fps, darkening_factor, dpi, chunk_size, plot_visual_stats, stimuli_dir, output_dir, verbose=False):

    # Directories
    sub_data_dir = os.path.join(output_dir, sub)
    mask_path = os.path.join(sub_data_dir, f'{sub}_mask-history_{fps}-fps.npz' if chunk_size is None else f'{sub}_mask-history_{fps}-fps_chunked')
    gaze_path = os.path.join(sub_data_dir, f'{sub}_gaze-history_{fps}-fps.npy')
    visual_stats_path = os.path.join(sub_data_dir, f'{sub}_visual-stats-history_{fps}-fps.csv')
    validation_str = 'validation_' if validation else ''
    video_frames_dir = os.path.join(stimuli_dir, f'{validation_str}video_frames_{fps}FPS')
    img_output_dir = os.path.join(sub_data_dir, f'frustum_frames_{fps}FPS')
    validation_str = '(validation)' if validation else ''
    os.makedirs(img_output_dir, exist_ok=True)

    # Check if the required files exist
    if not os.path.exists(mask_path) or not os.path.exists(gaze_path):
        print(f'Missing pre-computed frustum mask history or gaze history for subject {sub} {validation_str} at {mask_path} or {gaze_path}')
        print(f'Please run compute_frustum_mask.py for subject {sub} {validation_str} to generate the required files')
        exit(1)
    if plot_visual_stats and not os.path.exists(visual_stats_path):
        print(f'Missing pre-computed visual statistics history for subject {sub} {validation_str} at {visual_stats_path}.')
        print(f'Please run compute_visual_stats.py for subject {sub} {validation_str} to generate the required file')
        exit(1)

    # Load pre-computed frustum mask history, gaze history, and visual statistics history if applicable
    if verbose:
        print(f'Loading pre-computed frustum mask history and gaze history for subject {sub} {validation_str} stored at {mask_path} and {gaze_path}')
    gaze_history = np.load(gaze_path, allow_pickle=True)
    if chunk_size is None:
        mask_history_loaded = np.load(mask_path, allow_pickle=True)
        mask_history = {k: mask_history_loaded[k] for k in mask_history_loaded.files}
    else: 
        chunk_files = sorted([f for f in os.listdir(mask_path) if f.startswith('chunk-') and f.endswith('.npz')])

    visual_stats_history = None
    if plot_visual_stats:
        visual_stats_history = pd.read_csv(visual_stats_path, index_col=0).to_dict(orient='index')

    for idx in tqdm.tqdm(range(len(gaze_history)), desc=f'Plotting frames with frustum for {sub} {validation_str}'):
        curr_exp_time = round(idx / fps, 3)
        gaze_fixation = gaze_history[idx]
        if chunk_size is None:
            fov_mask = mask_history.get(str(idx))
        else:
            chunk_history = np.load(os.path.join(mask_path, f'chunk-{idx//chunk_size}.npz'), allow_pickle=True)
            fov_mask = chunk_history.get(str(idx%chunk_size))
            print(f"For chunk '{idx//chunk_size}': getting mask at index {idx%chunk_size}")
        visual_stats = visual_stats_history[idx] if plot_visual_stats else None
        img = np.array(Image.open(os.path.join(video_frames_dir, f'frame_{idx}.jpg')))
        plot_imgNfrustum(sub, img, fov_mask, gaze_fixation, visual_stats, curr_exp_time, output_dir=img_output_dir, darkening_factor=darkening_factor, dpi=dpi)
    if verbose:
        print(f'Frames with frustum plotted and saved at {img_output_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default=None, help='Subject ID (e.g., sub-001). If None, it will iterate through all subject directories in data_dir.')
    parser.add_argument('--validation', action='store_true', default=False, help='Whether to run in validation mode (uses validation video frames).')
    parser.add_argument('--fps', type=int, default=1, help='FPS at which data will be trimmed to match extracted video frames.')
    parser.add_argument('--darkening_factor', type=float, default=0.75, help='Factor by which to darken the image outside the frustum mask (between 0 and 1).')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for the saved images.')
    parser.add_argument('--chunk_size', type=int, default=None, help='Size of chunks for saving the mask history. If None, the entire mask history will be saved in a single file.')
    parser.add_argument('--visual_stats', action='store_true', default=False, help='Whether to plot (normalized) visual statistics within each frame using activity bars.')
    parser.add_argument('--stimuli_dir', type=str, default=Path('stimuli'), help='Directory containing the video frames.')
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
        main_function(sub=curr_sub, validation=args.validation, fps=args.fps, darkening_factor=args.darkening_factor, dpi=args.dpi, chunk_size=args.chunk_size, plot_visual_stats=args.visual_stats, stimuli_dir=args.stimuli_dir, output_dir=args.output_dir, verbose=True) #args.verbose)

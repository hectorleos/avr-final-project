import argparse
import os
from pathlib import Path
import tqdm
import compute_visual_stats
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation
from compute_visual_stats import compute_visual_statistics
import resource

# ---- Utility functions ---

def data_subset_fps(data, fps, total_video_length):
    ''' Gets a subset of the data at desired FPS by selecting the row closest to each second. '''
   # data = data.reset_index(drop=True)
    second_indices = []
    step = 1 / fps
    for sec in np.arange(0, int(total_video_length), step):
        diffs = data['exp_time'] - sec
        # Keep only positive values
        diffs = diffs.where(diffs >= 0)
        idx = diffs.idxmin()
        second_indices.append(int(idx))
    subset_data = data.iloc[second_indices].reset_index(drop=True)
    return subset_data

def degree_to_vector(pitch, yaw):
    ''' Converts pitch and yaw angles (in degrees) to a 3D unit vector in Cartesian coordinates. '''
    yaw_rad   = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    x =  np.cos(pitch_rad) * np.cos(yaw_rad)
    y =  np.cos(pitch_rad) * np.sin(yaw_rad)
    z =  np.sin(pitch_rad)
    return np.array([x, y, z])

def rotate_around(vec, axis, angle_rad):
    ''' Applies Rodrigues' rotation formula to rotate a vector around a given axis by a specified angle in radians. '''
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1 - c)


# ---- Main functions ---

def compute_fov_frustum(data, video_width, video_height, hfov, vfov, N_rays=64, binary_dilation_iters=2):
    """
    ***CREATED WITH THE HELP OF CLAUDE***
    Projects a rectilinear FOV frustum onto the equirectangular sphere and returns the UV pixel coordinates of the sampled grid.

    Input: 
        data: pandas DataFrame
            Contains head and gaze movement data for a participant in one experimental trial.
        video_width, video_height: int
            Dimensions of the equirectangular video frame.
        hfov, vfov: float
            Horizontal and vertical field of view in degrees.
        N_rays: int
            Resolution of the grid (N_rays × N_rays rays sampled inside the FOV).
        binary_dilation_iters: int
            Number of iterations for binary dilation to fill in gaps between sampled rays.
    Output:
        mask: numpy array of shape (video_height, video_width)
            Boolean mask indicating which pixels are inside the FOV frustum.
        gaze_vector: tuple of floats
            Pixel coordinates (u, v) of the gaze fixation point in the equirectangular frame.
    """

    # --- 1. Build the head's orthonormal frame (forward / right / up) ---

    # Convert head pitch/yaw to a unit vector in 3D space (i.e., spherical-to-Cartesian transformation)
    head_forward_vec = degree_to_vector(data['head_pitch'], data['head_yaw'])

    # Create a "world-up vector" pointing up from the origin
    world_up = np.array([0.0, 0.0, 1.0])
    # In case participant is looking up, apply "gimbal-lock fallback" to avoid a cross product of 0
    if abs(np.dot(head_forward_vec, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])

    # Compute cross product between vectors to obtain a "right" vector perpendicular to both
    head_right_vec = np.cross(head_forward_vec, world_up)
    head_right_vec /= np.linalg.norm(head_right_vec)
    head_up_vec = np.cross(head_right_vec, head_forward_vec)   # already unit length

    # Apply head roll around the forward axis (Rodrigues rotation)
    head_right_vec = rotate_around(head_right_vec, head_forward_vec, -np.radians(data['head_roll']))  
    head_up_vec    = rotate_around(head_up_vec,    head_forward_vec, -np.radians(data['head_roll']))

    # --- 2. Cast an N_rays×N_rays grid of rays through the FOV frustum ---

    # We assume screen is placed one unit away from observer. Then we can obtain half-width/half-height of the FOV in tangent space
    half_h = np.tan(np.radians(hfov / 2))
    half_v = np.tan(np.radians(vfov / 2))

    # Create grid of N_rays evenly-spaced positions in this screen where rays will be cast
    xs = np.linspace(-half_h, half_h, N_rays)   
    ys = np.linspace(-half_v, half_v, N_rays)  
    gx, gy = np.meshgrid(xs, ys)          
    gx, gy = gx.ravel(), gy.ravel()

    # Each ray: forward + offset along right/up, then renormalize onto sphere
    rays = (head_forward_vec[None, :]               # (1, 3)
            + gx[:, None] * head_right_vec[None, :] # horizontal sweep
            + gy[:, None] * head_up_vec[None, :])   # vertical sweep
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    # --- 3. Convert unit rays to equirectangular pixel coordinates ---

    # Convert rays to spherical coordinates (longitude, latitude)
    rx, ry, rz = rays[:, 0], rays[:, 1], rays[:, 2]
    lon = np.arctan2(ry, rx)                    # [-π, π]
    lat = np.arcsin(np.clip(rz, -1.0, 1.0))     # [-π/2, π/2]

    # Scale to pixel coordinates in equirectangular frame
    u = ((np.degrees(lon) + 180) / 360) * video_width
    v = ((90 - np.degrees(lat)) / 180) * video_height

    # Wrap u horizontally (handles antimeridian crossing cleanly)
    u = u % video_width

    # --- 4. Obtain a boolean mask of the FOV region in pixel space ---
    mask = np.zeros((video_height, video_width), dtype=bool)
    ui = np.clip(np.round(u).astype(int), 0, video_width  - 1)
    vi = np.clip(np.round(v).astype(int), 0, video_height - 1)
    mask[vi, ui] = True
    # Dilate the mask to fill in gaps between sampled rays (since N_rays is discrete)
    mask = binary_dilation(mask, iterations=binary_dilation_iters)  

    # --- Repeat same process for gaze vector to get its pixel coordinates --- #
    
    # 1: Convert head pitch/yaw to a unit vector in 3D space
    gaze_x = np.tan(np.radians(-data['gaze_yaw']))    # offset along head's right axis
    gaze_y = np.tan(np.radians(-data['gaze_pitch']))  # offset along head's up axis

    # 2: Construct gaze ray with head vectors and normalize onto the sphere
    gaze_ray = head_forward_vec + gaze_x * head_right_vec + gaze_y * head_up_vec
    gaze_ray /= np.linalg.norm(gaze_ray)   

    # 3: Project gaze ray onto equirectangular pixel coordinates
    gaze_lon = np.arctan2(gaze_ray[1], gaze_ray[0])
    gaze_lat = np.arcsin(np.clip(gaze_ray[2], -1.0, 1.0))
    gaze_u = ((np.degrees(gaze_lon) + 180) / 360) * video_width
    gaze_v = ((90 - np.degrees(gaze_lat)) / 180) * video_height
    gaze_u = gaze_u % video_width
    gaze_vector = (gaze_u, gaze_v)

    return mask, gaze_vector


def main_function(sub, validation=False, fps=2, hfov=100, vfov=100, n_rays=100, binary_dilation_iters=2, chunk_size=None, compute_visual_stats=False, stimuli_dir=None, output_dir=None, verbose=True):
    '''
    Loads preprocessed data for a given subject, computes the FOV frustum and gaze fixation for each video frame, and saves the results.
    Input: 
        sub: str
            Subject ID (e.g., sub-001).
        validation: bool
            Whether to run in validation mode, using validation video frames instead of the main experimental stimuli. 
        fps: int
            FPS at which data will be trimmed to match extracted video frames.
        hfov: float
            Horizontal field of view in degrees.
        vfov: float
            Vertical field of view in degrees.
        n_rays: int
            Number of rays to sample for each frame.
        chunk_size: int
            Size of chunks for saving the mask history.
        compute_visual_stats: bool
            Whether to also compute visual statistics for each frame.
        stimuli_dir: str or Path
            Directory containing the video frames.
        output_dir: str or Path
            Directory containing the output data for each subject.
        verbose: bool
            Whether to print verbose output.
    Output:
        Saves the frustum mask history and gaze history for the subject in the output directory.
    '''
        
    # Directories
    sub_data_dir = os.path.join(output_dir, sub)
    mask_path = os.path.join(sub_data_dir, f'{sub}_mask-history_{fps}-fps.npz')
    gaze_path = os.path.join(sub_data_dir, f'{sub}_gaze-history_{fps}-fps.npy')
    visual_stats_path = os.path.join(sub_data_dir, f'{sub}_visual-stats-history_{fps}-fps.csv')
    validation_str = 'validation_' if validation else ''
    video_frames_dir = os.path.join(stimuli_dir, f'{validation_str}video_frames_{fps}FPS')
    validation_str = '(validation)' if validation else ''

    print(f'Computing FOV frustum and gaze fixation for subject {sub} {validation_str} using following parameters: fps={fps}, hfov={hfov}, vfov={vfov}')

    # Input validation
    if not os.path.exists(video_frames_dir):
        raise FileNotFoundError(f'Video frames directory {video_frames_dir} does not exist. Please make sure to generate the video frames first with the correct FPS with video_to_frames.py')

    # Load preprocessed subject data and trim to desired FPS
    if verbose:
        print(f'Loading preprocessed data for subject {sub} {validation_str} in directory {sub_data_dir}')
    with open(os.path.join(sub_data_dir, f'{sub}_preprocessed_data.csv'), 'r') as f:
        sub_data  = pd.read_csv(f)
    total_video_length = round(float(sub_data['exp_time'].iloc[-1]), 2)
    sub_data_trimmed = data_subset_fps(sub_data, fps, total_video_length)
    n_frames = sub_data_trimmed.shape[0]
    if verbose:
        print(f'Note: Data was trimmed to {n_frames} frames ({total_video_length} total seconds at {fps} FPS)')

    # Iterate over rows of data and compute FOV frustum and gaze fixation for each frame
    mask_history = {}
    gaze_history = {}
    visual_stats_history = {} # In case we want to compute visual statistics at the same time
    chunk_flush = chunk_size is not None
    if chunk_flush:
        mask_path = os.path.join(sub_data_dir, f'{sub}_mask-history_{fps}-fps_chunked')
        os.makedirs(mask_path, exist_ok=True)
        chunk_idx = 0     # If using chunked memory-mapping...
        mask_chunk = {}
    for idx in tqdm.tqdm(range(n_frames), desc=f'Computing frustum masks for {sub} {validation_str} ({n_frames} frames)'):
        # Get matching row data + curr_exp_time + image based on index
        curr_row = sub_data_trimmed.iloc[idx]
      #  curr_exp_time = round(curr_row['exp_time'], 3)
        img = np.array(Image.open(os.path.join(video_frames_dir, f'frame_{idx}.jpg')))
        w, h = img.shape[1], img.shape[0]
        fov_mask, gaze_fixation = compute_fov_frustum(curr_row, video_width=w, video_height=h, hfov=hfov, vfov=vfov, N_rays=n_rays, binary_dilation_iters=binary_dilation_iters)
        if chunk_flush:
            mask_chunk[str(idx)] = fov_mask
            if len(mask_chunk) >= chunk_size:
                np.savez(os.path.join(mask_path, f'chunk-{str(chunk_idx)}.npz'), **mask_chunk)
                mask_chunk = {} 
                chunk_idx += 1
                print(f'Chunk {str(chunk_idx)} saved to {mask_path}')
                peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"[chunk {chunk_idx}] peak memory so far: {peak_mb:.1f} MB", flush=True)
        else:
            mask_history[idx] = fov_mask
        gaze_history[idx] = gaze_fixation

        if compute_visual_stats:
            visual_stats = compute_visual_statistics(img, fov_mask, gaze_fixation)
            visual_stats_history[idx] = visual_stats

    # Save files
    if mask_chunk:
        np.savez(os.path.join(mask_path, f'chunk-{str(chunk_idx)}.npz'), **mask_chunk)
        print(f'Chunk {str(chunk_idx)} saved to {mask_path}')
        final_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"FINAL peak memory usage: {final_peak_mb:.1f} MB")
    else:
        np.savez(mask_path, **{str(k): v for k, v in mask_history.items()})
    np.save(gaze_path, np.array(list(gaze_history.values())))
    if compute_visual_stats:
        visual_stats_df = pd.DataFrame.from_dict(visual_stats_history, orient='index')
        visual_stats_df.index.name = 'frame_idx'
        visual_stats_df.to_csv(visual_stats_path)

    if verbose:
        print(f'Finished processing subject {sub} {validation_str}. Frustum mask history and gaze history saved to {sub_data_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default=None, help='Subject ID (e.g., sub-001). If None, it will iterate through all subject directories in data_dir.')
    parser.add_argument('--validation', action='store_true', default=False, help='Whether to run in validation mode (uses validation video frames).')
    parser.add_argument('--fps', type=int, default=1, help='FPS at which data will be trimmed to match extracted video frames.')
    parser.add_argument('--hfov', type=float, default=100.0, help='Horizontal field of view in degrees')
    parser.add_argument('--vfov', type=float, default=100.0, help='Vertical field of view in degrees')
    parser.add_argument('--n_rays', type=int, default=100, help='Number of rays to use for frustum computation.')
    parser.add_argument('--binary_dilation_iters', type=int, default=2, help='Number of iterations for binary dilation.')
    parser.add_argument('--chunk_size', type=int, default=None, help='Size of chunks for saving the mask history. If None, the entire mask history will be saved in a single file.')
    parser.add_argument('--compute_visual_stats', action='store_true', default=False, help='Whether to compute visual statistics for each frame after computing the frustum mask and gaze fixation.')
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
        main_function(sub=curr_sub, validation=args.validation, fps=args.fps, hfov=args.hfov, vfov=args.vfov, n_rays=args.n_rays, binary_dilation_iters=args.binary_dilation_iters, chunk_size=args.chunk_size, compute_visual_stats=args.compute_visual_stats, stimuli_dir=args.stimuli_dir, output_dir=args.output_dir, verbose=True) #args.verbose)

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# GLOBAL VARIABLES
TOTAL_VIDEO_LENGTH = 1396.9  # seconds
GLOBAL_YAW_OFFSET = -90 # degrees
YAW_CUSTOM_ROTATIONS = {'invasion': -90,
                        'asteroids': 90,
                        'underwood': 90}

# ---- Utility functions --- +-+-

def correct_to_180(df):
    ''' Corrects head movement values to be within [-180, 180] degrees '''
    df = df.apply(lambda x: x - 360 if x > 180 else x)
    df = df.apply(lambda x: x + 360 if x < -180 else x)
    return df

def plot_head_movement(data, video_start_data, sub, output_dir=None):
    ''' Plots head tracking data for given subject '''
    i = 1
    plt.figure(figsize=(15, 5))
    for head_movement in ['yaw', 'pitch', 'roll']:
        time_in_minutes = data['exp_time'] / 60
        plt.subplot(3, 1, i)
        plt.scatter(time_in_minutes, data['head_'+head_movement], s=0.1, color='black', zorder=1000)
        plt.xlabel('Time (minutes)')
        plt.ylabel(f'{head_movement.capitalize()} (degrees)')
        plt.xlim(0, time_in_minutes.max())

        # For x lim print all integers
        plt.xticks(np.arange(0, np.ceil(time_in_minutes.max()) + 1, 1))

        # Plot horizontal lines at relevant degrees
        angles = [-180, -90, 0, 90, 180] if head_movement == 'yaw' else [0]
        for angle in angles:
            plt.axhline(y=angle, color='darkblue', alpha=0.7, linewidth=0.5, zorder=0)
        # Plot each entry in video_start_data as a vertical line
        for _, row in video_start_data.iterrows():
            plt.axvline(x=row['onset'] / 60, color='red', linestyle='--', alpha=0.7, zorder=1)
            if i == 1:
                plt.text(x=row['onset'] / 60, y=220, s=row['trial_type'], color='red', fontsize=8, rotation=25)
        i += 1
    plt.suptitle(f'{sub} head tracking data')
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'{sub}_head_movement.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ---- Main function ---

def preprocess_subject_data(sub, validation=False, plot_head_tracking=True, data_dir=None, output_dir=None, verbose=False):
    '''
    Preprocesses the raw AVR data for a given subject and returns a combined dataframe with eye tracking and head movement data, 
    as well as a dataframe with the start times of each video. Optionally, saves a plot with head tracking data.
     Input: 
        sub: str
            Subject ID (e.g., sub-001).
        validation: bool
            Whether to run in validation mode, using validation experimental data. 
        plot_head_tracking: bool
            Whether to generate and save a plot of head tracking data (pitch/yaw/roll over time) for visual quality control.
        data_dir: str or Path
            Directory containing the experimental data for each subject. If None, a default data directory is used.
        output_dir: str or Path
            Directory where processed dataframes and plots are saved. If None, a default output directory is used.
        verbose: bool, optional
            Whether to print progress and warning messages during preprocessing.
    Output:
        combined_df: pd.DataFrame
            Dataframe combining eye tracking and head movement data (aligned by timestamp) for the subject.
        video_start_times: pd.DataFrame
            Dataframe containing the start time of each video/stimulus for the subject.
    '''
    sub_data_dir = os.path.join(data_dir, sub)
    validation_str = '(validation)' if validation else ''
    print(f'Processing experimental data for subject {sub} {validation_str} at {sub_data_dir}...')
    task_name = 'Default' if validation else 'AVR'

    # Load eye tracking data (Note: using cyclopedian eye data)
    gaze_direc = os.path.join(sub_data_dir, 'eyetrack', f'{sub}_task-{task_name}_recording-eye_physio.tsv.gz')
    gaze_raw_data = pd.read_csv(gaze_direc, sep="\t", low_memory=False)
  #  raw_data.rename(columns={'x_coordinate': f'x', 'y_coordinate': f'y'}, inplace=True)
    gaze_raw_data.rename(columns={'pitch': 'gaze_pitch', 'yaw': 'gaze_yaw', 'roll': 'gaze_roll'}, inplace=True)
    gaze_raw_data = gaze_raw_data.apply(pd.to_numeric, errors='coerce')
    gaze_raw_data = gaze_raw_data[['timestamp', 'gaze_pitch', 'gaze_yaw', 'gaze_roll']].copy()

    # Load head tracking data
    my_hm_direc = os.path.join(sub_data_dir, 'motion', f'{sub}_task-{task_name}_tracksys-headmovement_motion.tsv.gz')
    hm_raw_data = pd.read_csv(my_hm_direc, sep='\t', compression='gzip')
    hm_raw_data.rename(columns={'x.1': 'head_pitch', 'y.1': 'head_yaw', 'z.1': 'head_roll'}, inplace=True)
    hm_raw_data['timestamp'] = hm_raw_data['timestamp'].astype(float)
    hm_raw_data = hm_raw_data[['timestamp', 'head_pitch', 'head_yaw', 'head_roll']].copy()

  
    for head_movement in ['head_pitch', 'head_roll', 'head_yaw']:
        hm_raw_data[head_movement] = correct_to_180(hm_raw_data[head_movement])
    if verbose:
        print('Warning: Head movement values corrected to be within [-180, 180]')

    # Invert pitch to match coordinate system
    hm_raw_data['head_pitch'] = - hm_raw_data['head_pitch']  
    if verbose:
        print('Warning: Head movement pitch values inverted to match coordinate system')

    # Merge head movement data with raw data based on timestamp
    merged_raw_data = pd.merge_asof(
        gaze_raw_data, 
        hm_raw_data, 
        on='timestamp', 
        direction='nearest'
    )

    # Load behavioral data to get the start and end timestamps of the video
    my_beh_direc = os.path.join(sub_data_dir, 'beh', f'{sub}_task-{task_name}_events.tsv')
    beh_data= pd.read_csv(my_beh_direc, sep = "\t", low_memory=False)
    start_timestamp = beh_data[beh_data['trial_type'] == 'play_video']['onset'].iloc[0 if validation else 1]
    start_timestamp = float(start_timestamp)
    end_timestamp = beh_data[beh_data['trial_type'] == 'video_end']['onset'].iloc[-1]
    end_timestamp = float(end_timestamp)

    # Obtain start time of each video & rename
    video_start_data = beh_data[beh_data['trial_type'] == 'play_video'][['onset', 'trial_type']].iloc[0 if validation else 1:] # Drop first video
    video_start_data['onset'] = video_start_data['onset'].astype(float) - np.float64(start_timestamp)
    video_start_data['trial_type'] = ['start_spaceship', 
                                        'start_invasion', 
                                        'start_spaceship',
                                        'start_asteroids',
                                        'start_spaceship',
                                        'start_underwood',
                                        'start_spaceship']

    # Assert statement: total time should be around 1396 seconds (i.e., video length: 23 minutes and 16 seconds)
    total_time = end_timestamp - start_timestamp
    assert np.isclose(total_time, TOTAL_VIDEO_LENGTH, atol=1), f"Subject {sub}: Total time {total_time} seconds is not close to expected {TOTAL_VIDEO_LENGTH} seconds."
    if verbose:
        print(f"Subject {sub} | start_timestamp: {start_timestamp} - end_timestamp: {end_timestamp} for a total time {total_time} seconds is close to expected {TOTAL_VIDEO_LENGTH} seconds :)))")

    # Filter the merged data to only include rows associated with the video only
    merged_raw_data.rename(columns={'timestamp': 'exp_time'}, inplace=True) 
    filtered_merged_raw_data = merged_raw_data[(merged_raw_data['exp_time'] >= start_timestamp) & (merged_raw_data['exp_time'] <= end_timestamp)]
    filtered_merged_raw_data.loc[:, 'exp_time'] = filtered_merged_raw_data['exp_time'] - np.float64(start_timestamp)
    # Apply global yaw offset + correct to be within [-180, 180]
    filtered_merged_raw_data.loc[:,'head_yaw'] = correct_to_180(filtered_merged_raw_data['head_yaw'] + GLOBAL_YAW_OFFSET)

    # Modify yaw on each video independently based on yaw_custom_rotations
    video_start_data = video_start_data.reset_index(drop=True)
    for video_name in YAW_CUSTOM_ROTATIONS:
        # Get indeces of current video
        curr_video_onset_idx = video_start_data[video_start_data['trial_type'] == f'start_{video_name}'].index[0]
        curr_video_onset = video_start_data.iloc[curr_video_onset_idx]['onset']
        next_video_onset = video_start_data.iloc[curr_video_onset_idx + 1]['onset']
        # Select indeces of filtered_merged_raw_data corresponding to current video
        indices_curr_video = filtered_merged_raw_data[(filtered_merged_raw_data['exp_time'] >= curr_video_onset) 
                                                      & (filtered_merged_raw_data['exp_time'] < next_video_onset)].index
        # Apply custom rotation dedicated to current video
        selected_yaw_values = filtered_merged_raw_data.loc[indices_curr_video, 'head_yaw'] + YAW_CUSTOM_ROTATIONS[video_name]
        # Correct to be within [-180, 180]
        filtered_merged_raw_data.loc[indices_curr_video, 'head_yaw'] = correct_to_180(selected_yaw_values)
    if verbose:
        print(f"Warning: Yaw values corrected 1) for global offset ({GLOBAL_YAW_OFFSET}) and 2) according to custom rotations ({YAW_CUSTOM_ROTATIONS})")

    # Save the combined data to a CSV file as well as head movement figures
    if output_dir is not None:
        sub_output_dir = os.path.join(output_dir, sub)
        os.makedirs(sub_output_dir, exist_ok=True) 
        filtered_merged_raw_data.to_csv(os.path.join(sub_output_dir, f'{sub}_preprocessed_data.csv'), index=False)
        if plot_head_tracking:
            plot_head_movement(filtered_merged_raw_data, video_start_data, sub, output_dir=sub_output_dir)
        if verbose:
            print(f"Preprocessed data for subject {sub} saved to {sub_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default=None, help='Subject ID (e.g., sub-001). If None, it will iterate through all subject directories in data_dir.')
    parser.add_argument('--validation', action='store_true', default=False, help='Whether to run in validation mode (uses validation video frames).')
    parser.add_argument('--plot_head_tracking', action='store_true', default=True, help='Whether to save plots displaying head tracking data.')
    parser.add_argument('--data_dir', type=str, default=Path('data'), help='Directory containing the experimental data for each subject.')
    parser.add_argument('--output_dir', type=str, default=Path('output'), help='Output directory.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print verbose output')
    args = parser.parse_args()

    # Modify data_dir and output_dir based on validation flag
    args.data_dir = args.data_dir if not args.validation else os.path.join(args.data_dir, 'validation')
    args.output_dir = args.output_dir if not args.validation else os.path.join(args.output_dir, 'validation')
    os.makedirs(args.output_dir, exist_ok=True) 

    if args.sub is None:
        subs = [f for f in os.listdir(args.data_dir) if 'sub-' in f]
        print(f'Since sub is None, we iterate over subject directories in {args.data_dir}: {subs}')
        if len(subs) == 0:
            print(f'No proper subject directories found in {args.data_dir}. Make sure they follow the naming convention "sub-00X"')
    elif isinstance(args.sub, str):
        subs = [args.sub]
    else:
        raise ValueError('For sub, please provide a string (e.g., sub-001) or None.')
    for curr_sub in subs:
        preprocess_subject_data(sub=curr_sub, 
                                validation=args.validation, 
                                plot_head_tracking=args.plot_head_tracking,
                                data_dir=args.data_dir, 
                                output_dir=args.output_dir,
                                verbose=args.verbose)
import os
import argparse
import cv2  

AVR_VIDEO_NAMES = ['scifi_fixcross.mp4', 
               'black.mp4', 
               'invasion_fixcross.mp4', 
               'black.mp4', 
               'scifi_fixcross.mp4',
               'black.mp4', 
                'asteroids_fixcross.mp4', 
                'black.mp4', 
                'scifi_fixcross.mp4', 
                'black.mp4', 
                'underwood_fixcross.mp4', 
                'black.mp4', 
                'scifi_fixcross.mp4']

def video_to_frames(video_dir, video_names, video_fps, desired_fps, frame_quality, crop_half):
    '''
    Extracts frames from video or list of videos at the preferred FPS.
    Input: 
        video_dir: str
            Input video directory (Default: stimuli/videos)
        video_names: list[str] | None
            Filenames of MP4 videos whose frames are to be extracted, in their correct order.
            If None, filename(s) will be read automatically from video directory.
        video_fps: int
            FPS of which video(s)
        desired_fps: int
            FPS at which frames should be extracted.
        frame_quality: int
            Image quality of each frame from 0 (worse) to 100 (best).
        crop_half: bool
            Whether to crop frame to keep only upper half, given some video setups for VR.
    Output:
        Video frames will be saved at the output directory. The filenames will include the frame count following 
        the desired FPS (e.g., if desired_fps=1 sec, then the frame count will be equivalent to full seconds).
    '''
    # Retrieve video filenames manually if necessary
    if video_names is None:
        video_names = AVR_VIDEO_NAMES # os.listdir(video_dir)
        print(f'Warning: Retrieving videos in {video_dir}. Order will be arbitrary.')
    print(f'Frames will be extracted according to the following order: {video_names}')

    # Input validation
    if not (0 <= frame_quality <= 100):
        raise ValueError(f"frame_quality must be between 0 and 100, got {frame_quality}")
    if desired_fps <= 0:
        raise ValueError(f"desired_fps must be positive, got {desired_fps}")
    non_video_files = [f for f in video_names if not f.endswith('.mp4')]
    if len(non_video_files) > 0:
        raise ValueError(f"video_names must all end in '.mp4', got invalid entries: {non_video_files}")
    
    # Create output directory
    output_dir =  os.path.join('stimuli', f'validation_video_frames_{desired_fps}FPS')
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each video in video list
    old_frame_count = 0
    new_frame_count = 0
    frame_step = video_fps / desired_fps  
    for video_name in video_names:
        print(f'---Extracting frames from video {video_name}')

        vidcap = cv2.VideoCapture(os.path.join(video_dir, video_name))
        success, image = vidcap.read()
        while success:

            # Only obtain frames at the desired FPS
            if old_frame_count % frame_step == 0:

                # Crop image to keep only upper half if needed
                if crop_half:
                    height = image.shape[0]
                    image = image[:height//2]

                # Save frame with desired quality
                frame_filename = os.path.join(output_dir, f"frame_{new_frame_count}.jpg")
                print(f'Frame saved at {frame_filename}')
                cv2.imwrite(frame_filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), frame_quality])
                new_frame_count += 1

            success, image = vidcap.read()
            old_frame_count += 1
        vidcap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default=os.path.join('stimuli', 'validation_videos'),
                     help='Directory containing the input videos.')
    parser.add_argument('--video_names', type=str, nargs='+', default=None,
                     help='Filenames of videos whose frames are to be extracted, in their correct order (e.g.,  --video_names first_video.mp4 second_video.mp4 third_video.mp4).' \
                     'If None, filename(s) will be read automatically from video directory.')
    parser.add_argument('--video_fps', type=int, default=30, help='Video FPS.')
    parser.add_argument('--desired_fps', type=int, default=1, help='FPS of extracted frames.')
    parser.add_argument('--frame_quality', type=int, default=10, help='Image quality of each frame from 0 (worse) to 100 (best).')
    parser.add_argument('--crop_half', action='store_true', default=False, help='Whether to crop frame to keep only upper half.')
    args = parser.parse_args()
    video_to_frames(video_dir=args.video_dir,
                    video_names=args.video_names, 
                    video_fps=args.video_fps, 
                    desired_fps=args.desired_fps, 
                    frame_quality=args.frame_quality, 
                    crop_half=args.crop_half)
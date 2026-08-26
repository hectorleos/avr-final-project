#!/bin/bash -l
#
#SBATCH -J avr_preprocess
#SBATCH -o ./logs/job_%j.out
#SBATCH -e ./logs/job_%j.err
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000MB
#SBATCH --time=10:00:00  


#ssbatchh comment --time=00:01:15 per 100 frames (--fps 30 --n_rays 300 --binary_dilation_iters 1 --chunk_size 1000 --compute_visual_stats)
# meaning 418 (41880 frames) * 1 min 15 sec = 8.7, make that 10 hours?

module purge
module load anaconda/3/2023.03
source $(conda info --base)/etc/profile.d/conda.sh
conda activate avr_env

# MAIN

srun python compute_frustum_mask.py --sub sub-001 --fps 30 --n_rays 300 --binary_dilation_iters 5 --chunk_size 1000 --compute_visual_stats --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output 
#srun python plot_frustum_frames.py --sub sub-002 --fps 1 --chunk_size 1000 --visual_stats --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output



# OLDER

#srun python video_to_frames.py --stimuli_dir /ptmp/hleosme/avr_stimuli --video_fps 30 --desired_fps 30 --frame_quality 90 --crop_half --verbose

#srun python compute_visual_stats.py --sub sub-001 --fps 30 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output
#srun python plot_frustum_frames.py  --sub sub-001 --fps 30 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output 

#srun python video_to_frames.py --stimuli_dir /ptmp/hleosme/avr_stimuli --video_fps 30 --desired_fps 30 --frame_quality 90 --crop_half --verbose
#srun python preprocess_exp_data.py --data_dir /ptmp/hleosme/avr_data/rawdata
#srun python compute_frustum_mask.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output 
#srun python compute_visual_stats.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output 
#srun python plot_frustum_frames.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --visual_stats 

# ------------------ RUN NOTEBOOK ------------------
# 25.08.2026 - job_29608199 - computed 30 FPS video frames again with --frame_quality 90

# 26.08.2026 - job_29642644 - computed 1 FPS video frames along with visual stats --sub sub-002 --fps 1 --n_rays 300 --binary_dilation_iters 10 --chunk_size 1000 --compute_visual_stats
# -- Computing frustum masks for sub-002  (1396 frames):  72%|███████▏  | 1000/1396 [07:43<54:24,  8.24s/it] with --binary_dilation_iters 10

# 26.08.2026 - job_29642846 - Same as job_29642644 but with --binary_dilation_iters 1
# -- Computing frustum masks for sub-002  (1396 frames):  72%|███████▏  | 1000/1396 [06:37<58:47,  8.91s/it]
# -- FINAL peak memory usage: 5712.9 MB

# 26.08.2026 - job_29643610 - Plotting frames form job_29642846 - delete log 

# 26.08.2026 - job_29643692 - RUNNING ENTIRE PIPELINE FOR SUB 1! 
# -- python compute_frustum_mask.py --sub sub-001 --fps 30 --n_rays 300 --binary_dilation_iters 5 --chunk_size 1000 --compute_visual_stats
# -- SBATCH --mem=10000MB
# -- SBATCH --time=10:00:00  
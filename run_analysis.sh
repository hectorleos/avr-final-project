#!/bin/bash -l
#
#SBATCH -J avr_preprocess
#SBATCH -o ./logs/job_%A_%a.out
#SBATCH -e ./logs/job_%A_%a.err
#SBATCH -D ./
#SBATCH --array=1,46 #----SBATCH --array=2-47
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000MB
#SBATCH --time=06:00:00

module purge
module load anaconda/3/2023.03
source $(conda info --base)/etc/profile.d/conda.sh
conda activate avr_env

# build zero-padded subject ID, e.g. task 2 -> sub-002
SUB=$(printf "sub-%03d" ${SLURM_ARRAY_TASK_ID})

# MAIN
srun python compute_frustum_mask.py --sub ${SUB} --fps 30 --n_rays 300 --chunk_size 1000 --compute_visual_stats --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output

# OLDER

#srun python compute_frustum_mask.py --sub sub-001 --fps 30 --n_rays 300 --binary_dilation_iters 5 --chunk_size 1000 --compute_visual_stats --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output 
#srun python plot_frustum_frames.py --sub sub-001 --fps 30 --chunk_size 1000 --dpi 10 --visual_stats --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output
#srun python video_to_frames.py --stimuli_dir /ptmp/hleosme/avr_stimuli --video_fps 30 --desired_fps 30 --frame_quality 90 --crop_half --verbose

#srun python compute_visual_stats.py --sub sub-001 --fps 30 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output
#srun python plot_frustum_frames.py  --sub sub-001 --fps 30 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output 

#srun python video_to_frames.py --stimuli_dir /ptmp/hleosme/avr_stimuli --video_fps 30 --desired_fps 30 --frame_quality 90 --crop_half --verbose
#srun python preprocess_exp_data.py --data_dir /ptmp/hleosme/avr_data/rawdata
#srun python compute_frustum_mask.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output 
#srun python compute_visual_stats.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output 
#srun python plot_frustum_frames.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --visual_stats 


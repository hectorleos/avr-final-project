#!/bin/bash -l
#
#SBATCH -J avr_preprocess
#SBATCH -o ./logs/job_%j.out
#SBATCH -e ./logs/job_%j.err
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000MB
#SBATCH --time=01:00:00  

module purge
module load anaconda/3/2023.03
source $(conda info --base)/etc/profile.d/conda.sh
conda activate avr_env

srun python video_to_frames.py --stimuli_dir /ptmp/hleosme/avr_stimuli --video_fps 30 --desired_fps 1 --crop_half --verbose
#srun python preprocess_exp_data.py --data_dir /ptmp/hleosme/avr_data/rawdata
#srun python compute_frustum_mask.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --validation
#srun python compute_visual_stats.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --validation
#srun python plot_frustum_frames.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --visual_stats --validation


python compute_visual_stats.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --validation
python plot_frustum_frames.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --visual_stats --validation
python compute_frustum_mask.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --sub sub-001
python compute_visual_stats.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --sub sub-001
python plot_frustum_frames.py --fps 1 --stimuli_dir /ptmp/hleosme/avr_stimuli --output_dir /ptmp/hleosme/avr_output --visual_stats --sub sub-001
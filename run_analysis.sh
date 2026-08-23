#!/bin/bash -l
#
#SBATCH -J avr_preprocess
#SBATCH -o ./logs/job_%j.out
#SBATCH -e ./logs/job_%j.err
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000MB
#SBATCH --time=01:00:00  

module purge
module load anaconda/3/2023.03
source $(conda info --base)/etc/profile.d/conda.sh
conda activate avr_env

srun python preprocess_exp_data.py --data_dir /ptmp/hleosme/avr_data/rawdata
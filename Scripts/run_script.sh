#!/bin/bash
#
# Usage: sbatch --export=SCRIPT=your_script.py run_script.sh
#

#SBATCH --partition=general
#SBATCH --time=01-00:00:00
#SBATCH --mem=128G
#SBATCH --job-name=run_script
#SBATCH --output=/gpfs2/scratch/pwormser/research/slurm_out/slurm_%x_%j.out
#SBATCH --begin=now

module load anaconda  # or miniconda, or whatever your system uses

# Activate your environment
source activate data-mountain-query

# Get base name (without .py) for logging or later use
SCRIPT_NAME=$(basename "$SCRIPT" .py)

# Run the script
python ~/scratch/research/Scripts/"$SCRIPT"


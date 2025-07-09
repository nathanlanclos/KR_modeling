#!/bin/bash
#SBATCH --job-name=xiang_kr_trial_pred      # Job name
#SBATCH --partition=savio3_gpu         # Partition
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --time=10:00:00                # Time limit
#SBATCH --account=fc_pkss              # Account
#SBATCH --gres=gpu:A40:1               # Request 1 A40 GPU
#SBATCH --qos=a40_gpu3_normal         # QoS for A40 GPU

# Load Anaconda for Conda environments
module load anaconda3

# Activate your correctly-built Conda environment
source activate boltz_2

# Set the BOLTZ_CACHE environment variable
export BOLTZ_CACHE="/global/scratch/users/nlanclos/boltz/boltz_cache"

cd /global/scratch/users/nlanclos/boltz

# Run your wrapper script
python boltz_wrapper.py \
  --input_csv xiang_kr_db_structure_pred_test.csv \
  --output_dir xiang_kr_db_structure_pred_test_2 \
  --max_time 75 \
  --num_replicates 1 \
  --summary_csv_name final_summary.csv \
  --run_boltz_extra --accelerator gpu --use_msa_server --output_format pdb
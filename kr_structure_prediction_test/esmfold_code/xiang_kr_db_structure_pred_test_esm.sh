#!/bin/bash
#SBATCH --job-name=xiang_kr_trial_pred_esm      # Job name
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
source activate esm

cd /global/scratch/users/nlanclos/ESMFold_v1

INPUT_CSV="./xiang_kr_db_structure_pred_test.csv"
OUTPUT_DIRECTORY="./xiang_kr_db_structure_pred_test"
MODEL_NAME="facebook/esmfold_v1"
CHUNK_SIZE=256

echo "Starting protein structure prediction..."

python3 esm_generate_structures.py \
    --input_csv "$INPUT_CSV" \
    --output_directory "$OUTPUT_DIRECTORY" \
    --model_name "$MODEL_NAME" \
    --chunk_size "$CHUNK_SIZE" \
    --use_fp16 \
    --allow_tf32

echo "Prediction script finished."
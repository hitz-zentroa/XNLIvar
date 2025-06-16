#!/usr/bin/env bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=xnli_gemmainstruct27
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-sxm4
#SBATCH --output=/scratch/jbengoetxea/phd/XNLIvar/scripts/generative/logs/xnli_gemmainstruct27_%a.log
#SBATCH --error=/scratch/jbengoetxea/phd/XNLIvar/scripts/generative/logs/xnli_gemmainstruct27_%a.err
#SBATCH --time=01:00:00 #ee-hh:mm:ss
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=jaione.bengoetxea@ehu.eus   
#SBATCH --array=0-29%2 


export TRANSFORMERS_CACHE="/scratch/jbengoetxea/.cache"

# Values for the 2 loops:
DATASET_VALUES=(xnli-eu-var xnli-eu-native xnli-eu xnli-es-var xnli-es-native xnli-es)  
PROMPT_TYPE_VALUES=(chain nli-zero nli-few qa-zero qa-few)             

N=${#PROMPT_TYPE_VALUES[@]}  # Number of items in the second level (VALUES2)

# Decode SLURM_ARRAY_TASK_ID to get the two indices
IDX1=$((SLURM_ARRAY_TASK_ID / N))
IDX2=$((SLURM_ARRAY_TASK_ID % N))

# Use IDX1 and IDX2 for your two-level loops
DATASET="${DATASET_VALUES[${IDX1}]}"
PROMPT_TYPE="${PROMPT_TYPE_VALUES[${IDX2}]}"


TASK=trilabel
MODEL=gemmainstruct27
OUTPUT=/scratch/jbengoetxea/phd/XNLIvar/scripts/generative/results/$DATASET/$MODEL

python3 /scratch/jbengoetxea/phd/XNLIvar/scripts/generative/scripts/zero_shot.py \
    --dataset "${DATASET}" \
    --model $MODEL \
    --output_dir $OUTPUT \
    --task $TASK \
    --prompt_type "${PROMPT_TYPE}" 

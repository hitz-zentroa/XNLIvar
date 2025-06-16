#!/usr/bin/env bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=xnli_llamainstruct70
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-sxm4
#SBATCH --output=/scratch/jbengoetxea/phd/XNLIvar/scripts/generative/logs/xnli-llamainstruct70_%a.log
#SBATCH --error=/scratch/jbengoetxea/phd/XNLIvar/scripts/generative/logs/xnli-llamainstruct70_%a.err
#SBATCH --time=01:00:00 #ee-hh:mm:ss
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=jaione.bengoetxea@ehu.eus   
#SBATCH --array=0-35%2 


export TRANSFORMERS_CACHE="/scratch/jbengoetxea/.cache"

# Values
DATASET_VALUES=(xnli-eu-var xnli-eu-native xnli-eu xnli-es-var xnli-es-native xnli-es)  
PROMPT_TYPE_VALUES=(contradiction entailment neutral)
TASK_VALUES=(qa-zero qa-few)

# Get job array working
D=${#DATASET_VALUES[@]}
P=${#PROMPT_TYPE_VALUES[@]}
T=${#TASK_VALUES[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

IDX_D=$((TASK_ID / (P * T)))
IDX_P=$(((TASK_ID / T) % P))
IDX_T=$((TASK_ID % T))

DATASET="${DATASET_VALUES[$IDX_D]}"
PROMPT_TYPE="${PROMPT_TYPE_VALUES[$IDX_P]}"
TASK="${TASK_VALUES[$IDX_T]}"

# Final values and run script
MODEL=llama3instruct70
OUTPUT=/scratch/jbengoetxea/phd/XNLIvar/scripts/generative/results/$DATASET/$MODEL/$TASK

python3 /scratch/jbengoetxea/phd/XNLIvar/scripts/generative/scripts/zero_shot.py \
    --dataset "${DATASET}" \
    --model $MODEL \
    --output_dir $OUTPUT \
    --task $TASK \
    --prompt_type "${PROMPT_TYPE}" 

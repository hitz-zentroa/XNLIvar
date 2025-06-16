#!/bin/bash
#SBATCH --job-name=nli-es
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=model-tran.log
#SBATCH --error=model-tran.err

 
for seed in 23 27 33
do

    for model in FacebookAI/xlm-roberta-base FacebookAI/xlm-roberta-large microsoft/mdeberta-v3-base 
  
    do
        python fine-tuning/run_xnli_es.py \
        --model_name_or_path $model \
        --language es \
        --train_language en \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 32 \
        --learning_rate 10e-6 \
        --num_train_epochs 10.0 \
        --max_seq_length 128 \
        --output_dir /gaueko1/users/jbengoetxea004/phd/xnli-paraphrasing/xnli-es/models/model-transfer/$model/$seed \
        --save_steps 50000 \
        --load_best_model_at_end 1 \
        --metric_for_best_model accuracy \
        --seed $seed \
        --report_to="wandb" \
        --evaluation_strategy steps \
        --eval_steps 5000 \
        --save_total_limit 2 
    done
done
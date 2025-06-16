#!/bin/bash
#SBATCH --job-name=pred-model-transfer
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=pred-model-transfer.log
#SBATCH --error=pred-model-transfer.err

for seed in 27 23 33
do
    for model in FacebookAI/xlm-roberta-large FacebookAI/xlm-roberta-base microsoft/mdeberta-v3-base
    do
        for dataset in es native var
        do 
            python fine-tuning/run_xnli_es.py \
            --model_name_or_path /gaueko1/users/jbengoetxea004/phd/xnli-paraphrasing/xnli-es/models/model-transfer/$model/$seed \
            --language es \
            --train_language en \
            --test_data $dataset \
            --do_predict \
            --per_device_train_batch_size 32 \
            --num_train_epochs 10.0 \
            --max_seq_length 128 \
            --output_dir /gaueko1/users/jbengoetxea004/phd/xnli-paraphrasing/xnli-es/results/model-transfer/$model/$dataset/$seed \
            --save_steps 50000 \
            --load_best_model_at_end 1 \
            --metric_for_best_model accuracy \
            --seed $seed \
            --evaluation_strategy steps \
            --eval_steps 5000 \
            --save_total_limit 2 
        done
    done
done
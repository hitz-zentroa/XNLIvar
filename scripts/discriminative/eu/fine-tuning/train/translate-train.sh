#!/bin/bash
#SBATCH --job-name=var-nli-tra
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=translate-train.log
#SBATCH --error=translate-train.err


for seed in 33
do
    for model in ixa-ehu/berteus-base-cased
    do
        python fine-tuning/run_xnli_eus.py \
        --model_name_or_path $model \
        --language eu \
        --train_language eu \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 10.0 \
        --max_seq_length 128 \
        --output_dir /gaueko1/users/jbengoetxea004/phd/xnli-paraphrasing/xnli-eu/models/translate-train/$model/$seed \
        --save_steps 50000 \
        --load_best_model_at_end 1 \
        --metric_for_best_model accuracy \
        --seed $seed \
        --evaluation_strategy steps \
        --eval_steps 5000 \
        --save_total_limit 2 
    done

    for model in microsoft/mdeberta-v3-base FacebookAI/xlm-roberta-large ixa-ehu/roberta-eus-euscrawl-large-cased
    do
        python fine-tuning/run_xnli_eus.py \
        --model_name_or_path $model \
        --language eu \
        --train_language eu \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 32 \
        --learning_rate 10e-6 \
        --num_train_epochs 10.0 \
        --max_seq_length 128 \
        --output_dir /gaueko1/users/jbengoetxea004/phd/xnli-paraphrasing/xnli-eu/models/translate-train/$model/$seed \
        --save_steps 50000 \
        --load_best_model_at_end 1 \
        --metric_for_best_model accuracy \
        --seed $seed \
        --evaluation_strategy steps \
        --eval_steps 5000 \
        --save_total_limit 2
    done

done
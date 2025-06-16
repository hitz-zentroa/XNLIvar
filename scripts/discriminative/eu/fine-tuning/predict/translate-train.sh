#!/bin/bash
#SBATCH --job-name=var-nli-prediction
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=var-nli-tr.log
#SBATCH --error=var-nli-tr.err

for seed in 27 23 33
do
    for model in microsoft/mdeberta-v3-base FacebookAI/xlm-roberta-large ixa-ehu/roberta-eus-euscrawl-large-cased ixa-ehu/berteus-base-cased
    do    
        for dataset in eu native var
        do
            python fine-tuning/run_xnli_eus.py \
            --model_name_or_path /gaueko1/users/jbengoetxea004/phd/xnli-paraphrasing/xnli-eu/models/translate-train/$model/$seed \
            --language eu \
            --train_language eu \
            --test_data $dataset \
            --do_predict \
            --per_device_train_batch_size 32 \
            --num_train_epochs 10.0 \
            --max_seq_length 128 \
            --output_dir /gaueko1/users/jbengoetxea004/phd/xnli-paraphrasing/xnli-eu/results/translate-train/$model/$dataset/$seed \
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
#!/bin/bash

while true; do
    python3 qlora.py \
        --model_name_or_path huggyllama/llama-30b \
        --truffle_size mini \
        --output_dir /scratch/data/models/output/dreamshow-30b-lora-new \
        --dataset /home/srikanth-deepshard/src/data.csv \
        --dataset_format self-instruct \
        --logging_steps 10 \
        --save_strategy steps \
        --save_steps 40 \
        --save_total_limit 40 \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --eval_dataset_size 1024 \
        --max_eval_samples 1000 \
        --per_device_eval_batch_size 1 \
        --dataloader_num_workers 3 \
        --group_by_length \
        --logging_strategy steps \
        --remove_unused_columns False \
        --do_train \
        --do_eval \
        --lora_r 64 \
        --lora_alpha 16 \
        --double_quant \
        --quant_type nf4 \
        --bf16 \
        --bits 8 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type constant \
        --gradient_checkpointing \
        --learning_rate 3e-4 \
        --adam_beta2 0.999 \
        --max_grad_norm 0.3 \
        --lora_dropout 0.05 \
        --weight_decay 0.0 \
        --full_finetune False \
        --per_device_train_batch_size 6 \
        --finetune_id "dreamshow-30b-lora"
    
    status=$?
    if [ $status -eq 123 ]; then
        echo "Special exit code encountered: $status. Exiting loop..."
        break
    elif [ $status -ne 0 ]; then
        echo "qlora.py exited with status $status. Restarting process..." >&2
    fi
    
    sleep 1
done
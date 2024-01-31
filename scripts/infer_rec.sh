#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python infer_rec.py \
    --base_model './gpt-3.5-turbo' \
    --lora_weights './gpt-3.5-turbo' \
    --use_lora True \
    --instruct_dir './data/beauty_sequential_single_prompt_test.json' \
    --prompt_template 'rec_template' \
    --max_new_tokens 256 \
    --num_return_sequences 10 \
    --num_beams 10
CUDA_VISIBLE_DEVICES=0 python infer_rec.py \
    --base_model './gpt-3.5-turbo' \
    --lora_weights './gpt-3.5-turbo' \
    --use_lora True \
    --instruct_dir './data/beauty_sequential_single_prompt_test.json' \
    --prompt_template 'rec_template' \
    --max_new_tokens 256 \
    --num_return_sequences 10 \
    --num_beams 10
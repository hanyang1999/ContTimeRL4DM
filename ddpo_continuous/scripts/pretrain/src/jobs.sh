#!/bin/bash

# Run the first job and wait for it to complete
accelerate launch --multi_gpu --num_processes=8 pretrain.py --config=config/base_both_non_denoised.py

sleep 60

# Run the second job after the first one completes
accelerate launch --multi_gpu --num_processes=8 pretrain.py --config=config/base_both_denoised.py

sleep 60

# Run the second job after the first one completes
accelerate launch --multi_gpu --num_processes=8 pretrain.py --config=config/base_cos_sin.py
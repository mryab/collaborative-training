#!/bin/bash

python run_trainer.py \
  --output_dir ./outputs --overwrite_output_dir \
  --logging_dir ./logs --logging_first_step --logging_steps 100 \
  --initial_peers 127.0.0.1:1337 --seed 0
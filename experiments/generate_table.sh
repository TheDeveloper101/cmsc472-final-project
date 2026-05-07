#!/bin/bash

python3 src/experiments.py --num-images 5000 --images-per-batch 1250 --job-name-prefix "baseline" --output "results/experiments/benchmark_baseline.json" --device "Samsung Galaxy S25 (Family)"


python3 src/experiments.py --calibration-samples 5000 --quantize int8 --num-images 5000 --images-per-batch 1250 --job-name-prefix "int8" --output "results/experiments/benchmark_quantize_int8.json" --device "Samsung Galaxy S25 (Family)"


python3 src/experiments.py --calibration-samples 5000 --quantize int16 --num-images 5000 --images-per-batch 1250 --job-name-prefix "int16" --output "results/experiments/benchmark_quantize_int16.json" --device "Samsung Galaxy S25 (Family)"

python3 src/experiments.py --calibration-samples 5000 --quantize w8a16 --num-images 5000 --images-per-batch 1250 --job-name-prefix "w8a16" --output "results/experiments/benchmark_quantize_w8a16.json" --device "Samsung Galaxy S25 (Family)"


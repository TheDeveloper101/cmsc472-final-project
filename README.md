# 472 Final Project - Image-Text Retrieval on Edge Devices
This repo explores the effect of quantization on performance and resource usage, quantizing MobileClip models of various sizes at varying levels of quantization, and exploring the effects this has on recall accuracy, latency, and memory usage. We evaluate these effects by running these quantized models on a Samsung Galaxy S25 device accessible through Qualcomm AI Hub (QAI Hub). 

Our process is as follows:

1. Exporting models to ONNX
2. Uploading evaluation datasets to QAI Hub
3. Compiling and quantizing for a target device
4. Running inference and computing Recall@10

The main “end-to-end” entrypoint is `src/experiments.py`.

## Environment Setup

**Prereqs**

- Python 3.10
- A working [QAI Hub account](https://workbench.aihub.qualcomm.com/) and credentials configured locally, required to submit compilation/inference jobs on the edge device
```bash
conda create -n clipenv python=3.10
conda activate clipenv
python -m pip install --upgrade pip
pip install -r requirements.txt
qai-hub configure --api_token API_TOKEN
```
## Project Configuration

This project uses `config.ini` for a couple repo-relative paths:

- `COCO_PATH`: expected local COCO location for some scripts
- `RESULTS_PATH`: where outputs (ONNX artifacts, experiment summaries, etc.) are written

Default:

```ini
[DEFAULT]
COCO_PATH=data/coco
RESULTS_PATH=results
```


## Reproducibility

To reproduce all results shown in the paper, run `experiments/reproduce.sh`. 
This will evaluate all results shown on Qualcomm AI Hub, running on a Samsung Galaxy S25. 
This will take around a day if run without paralleization on a single machine. 
To produce individual results, run `src/experiments.py` with arguments for the desired result, as described below.

We define our seeding function in `src/utils.py`, and invoke it wherever we run our model (starting in `src/experiments.py`, before running main). We alalso define hyperparameters like dataset size there as well. 

## Using `experiments.py`

`src/experiments.py` runs the full pipeline:

1. Upload image/text datasets to QAI Hub (with caching in `datasets.json`)
2. Export ONNX artifacts under `results/onnx_experiments/...`
3. Submit compile jobs once per model
4. Submit quantization jobs once per model (if `--quantize` is set)
5. Submit inference jobs (text encoder, image encoder in batches, and top-k)
6. Write a JSON summary to `results/experiments/experiments_<timestamp>.json` or a custom path.

### Basic Runs

To reproduce a particular baseline on a particular model, run the following command. 
```bash
python src/experiments.py --models <model name, defaults to all models> --num-images 5000 --images-per-batch 1250 --job-name-prefix "prefix" --output "results/experiments/results_name.json" --device "Samsung Galaxy S25 (Family)"
```

To reproduce a particular quantization level on a particular model, run the following command. 
```bash
python src/experiments.py

python src/experiments.py --models <model name, defaults to all models> --quantize <int8, int16, w8a16> --calibration-samples 5000 --num-images 5000 --images-per-batch 1250 --job-name-prefix "prefix" --output "results/experiments/results_name.json" --device "Samsung Galaxy S25 (Family)"
```
### CLI Help (`--help`)

The flags for `src/experiments.py` can be accessed via `src/experiments.py --help`:

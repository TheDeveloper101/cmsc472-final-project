import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from utils import NUM_CALIBRATION_SAMPLES

import qai_hub

# Input JSON produced by src/experiments.py
INPUT_JSON = "results/old_experiments/benchmark_quantize_w8a16.json"

# Output JSON path (will not overwrite INPUT_JSON). If None, write alongside input with a timestamp suffix.
OUTPUT_JSON = "results/experiments/w8a16_profile_test.json"

# Assumed calibration sample counts when experiments.py was run with --calibration-samples (default 5000 in runner.sh).
# Used only to populate num_image_calibration_samples / num_text_calibration_samples.

# Parallelism for submitting + downloading profile jobs.
MAX_PROFILE_WORKERS = 4


def _out_path() -> str:
    if OUTPUT_JSON is not None:
        if os.path.abspath(OUTPUT_JSON) == os.path.abspath(INPUT_JSON):
            raise RuntimeError("Refusing to overwrite INPUT_JSON.")
        if os.path.exists(OUTPUT_JSON):
            raise RuntimeError(f"Refusing to overwrite existing OUTPUT_JSON: {OUTPUT_JSON}")
        os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
        return OUTPUT_JSON

    base, _ext = os.path.splitext(INPUT_JSON)
    return f"{base}_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def _dataset_size_for_image_batch(*, batch_index: int, num_image_samples: int, images_per_batch: int) -> int:
    start = batch_index * images_per_batch
    if start >= num_image_samples:
        return 0
    return min(images_per_batch, num_image_samples - start)


def _profile_one(*, device: qai_hub.Device, model_name: str, quantize_type: str | None, kind: str, compiled_job_id: str):
    compiled = qai_hub.get_job(compiled_job_id).get_target_model()
    qlab = quantize_type if quantize_type is not None else "fp"
    name = f"profile :: {model_name} :: {kind} :: {qlab}"
    pj = qai_hub.submit_profile_job(
        model=compiled,
        device=device,
        name=name,
        options="--max_profiler_iterations 100",
    )
    raw = pj.download_profile() or {}
    summary = (raw.get("execution_summary") or {}) if isinstance(raw, dict) else {}
    keep = (
        "estimated_inference_time",
        "estimated_inference_peak_memory",
        "first_load_time",
        "first_load_peak_memory",
        "warm_load_time",
        "warm_load_peak_memory",
    )
    trimmed_summary = {k: summary[k] for k in keep if k in summary}
    data = {"execution_summary": trimmed_summary}
    return {
        "job_id": pj.job_id,
        "url": pj.url,
        "name": name,
        "data": data,
    }


with open(INPUT_JSON, "r") as f:
    doc = json.load(f)

device_name = doc.get("device")
if not device_name:
    raise RuntimeError("Missing top-level 'device' in input JSON.")
device = qai_hub.Device(device_name)

num_image_samples = int(doc.get("num_image_samples") or 0)
num_text_samples = int(doc.get("num_text_samples") or 0)
images_per_batch = int(doc.get("images_per_batch") or 0)

runs_in = doc.get("runs") or []

# Prebuild output records (one per run) and a list of profile tasks (text+image).
records: list[dict] = []
profile_tasks: list[tuple[int, str, str, str | None, str]] = []
# tuple: (record_index, kind, compiled_job_id, quantize_type, model_name)

for run in runs_in:
    model_name = run.get("model")
    quantize_type = run.get("quantize")
    recall = run.get("recall_at_10")
    jobs = run.get("jobs") or {}
    snap = run.get("job_ids_snapshot") or {}

    topk_jobs = (jobs.get("topk") or {}).get("jobs") or []
    images_per_topk_batch = (jobs.get("topk") or {}).get("batch_size")

    # Compiled job ids
    text_compiled_id = (snap.get("text") or {}).get("compiled_id")
    image_compiled_id = (snap.get("image") or {}).get("compiled_id")
    topk_compiled_id = (snap.get("topk") or {}).get("compiled_id")

    # Resolve compile job url/name via Hub.
    def compile_info(job_id: str | None):
        if not job_id:
            return None
        j = qai_hub.get_job(job_id)
        return {"job_id": job_id, "url": j.url, "name": getattr(j, "name", None)}

    compiled_block = {
        "text": compile_info(text_compiled_id),
        "image": compile_info(image_compiled_id),
        "topk": compile_info(topk_compiled_id),
    }

    # Inference: text
    text_ds_id = (snap.get("text") or {}).get("dataset_id")
    text_job = jobs.get("text") or {}
    inference_text = {
        "job_id": text_job.get("job_id"),
        "url": text_job.get("url"),
        "name": text_job.get("name"),
        "dataset_id": text_ds_id,
    }

    # Inference: images (already has dataset_id + batch_index)
    inference_images = []
    for rec in (jobs.get("images") or []):
        bi = int(rec.get("batch_index") or 0)
        inference_images.append(
            {
                "job_id": rec.get("job_id"),
                "url": rec.get("url"),
                "name": rec.get("name"),
                "dataset_id": rec.get("dataset_id"),
                "dataset_size": _dataset_size_for_image_batch(
                    batch_index=bi,
                    num_image_samples=num_image_samples,
                    images_per_batch=images_per_batch,
                ),
                "batch_index": bi,
            }
        )
    inference_images.sort(key=lambda r: int(r.get("batch_index") or 0))
    for r in inference_images:
        r.pop("batch_index", None)

    # Inference: topk jobs
    # NOTE: experiments.py does not record a dataset_id for topk jobs, and we intentionally
    # avoid scraping it from Hub job internals here. Include it only if it exists in the JSON.
    inference_topk = []
    for rec in topk_jobs:
        jid = rec.get("job_id")
        out = {
            "job_id": jid,
            "url": rec.get("url"),
            "name": rec.get("name"),
            "dataset_size": int(rec.get("batch_valid") or 0),
            "batch_index": int(rec.get("batch_index") or 0),
        }
        ds_id = rec.get("dataset_id")
        if ds_id:
            out["dataset_id"] = ds_id
        inference_topk.append(out)
    inference_topk.sort(key=lambda r: int(r.get("batch_index") or 0))
    for r in inference_topk:
        r.pop("batch_index", None)

    record = {
        "model": model_name,
        "quantize": quantize_type,
        "recall": recall,
        "device": device_name,
        "num_image_samples": num_image_samples,
        "num_text_samples": num_text_samples,
        "images_per_batch": images_per_batch,
        "images_per_topk_batch": images_per_topk_batch,
        "num_image_calibration_samples": int(NUM_CALIBRATION_SAMPLES),
        "num_text_calibration_samples": int(NUM_CALIBRATION_SAMPLES),
        "compiled": compiled_block,
        "inference": {
            "text": inference_text,
            "image": inference_images,
            "topk": inference_topk,
        },
        "profile": {
            "text": None,
            "image": None,
        },
    }

    idx = len(records)
    records.append(record)

    # Queue profile tasks (only if we have compiled IDs)
    if text_compiled_id:
        profile_tasks.append((idx, "text", text_compiled_id, quantize_type, model_name))
    if image_compiled_id:
        profile_tasks.append((idx, "image", image_compiled_id, quantize_type, model_name))


# Run profile jobs in parallel and fill records.
if profile_tasks:
    with ThreadPoolExecutor(max_workers=int(MAX_PROFILE_WORKERS)) as ex:
        fut_to_task = {
            ex.submit(
                _profile_one,
                device=device,
                model_name=model_name,
                quantize_type=quantize_type,
                kind=kind,
                compiled_job_id=compiled_job_id,
            ): (rec_idx, kind)
            for (rec_idx, kind, compiled_job_id, quantize_type, model_name) in profile_tasks
        }
        for fut in as_completed(fut_to_task):
            rec_idx, kind = fut_to_task[fut]
            records[rec_idx]["profile"][kind] = fut.result()


out_path = _out_path()
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

with open(out_path, "w") as f:
    json.dump(records, f, indent=2)

print("Wrote:", out_path)

import numpy as np
import qai_hub
from sklearn.metrics.pairwise import cosine_similarity

from constants import SPLIT, LIMIT
from utils.ground_truth import get_ground_truth
from utils.job_utils import JOB_IDS
from utils.refcoco_utils import RefCocoSplit


def run_inference(model, device, input_dataset, job_name=None):
    """Submits an inference job for the model and returns the output data."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1",
        name=job_name
    )
    # return inference_job.download_output_data()
    return inference_job.job_id


def evaluate_track1(img_output, txt_output, split: RefCocoSplit, limit=None, k=10):
    """
    Compute Recall@K between image and text embeddings using HF RefCOCO annotations.
    """

    # Stack them into a single 2D array: [batch, D]
    img_embeds = np.vstack([x for x in img_output])  # shape: [N, D]
    txt_embeds = np.vstack([x for x in txt_output])  # shape: [M, D]

    # Normalize
    img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)

    # Now similarity will work
    sim_matrix = cosine_similarity(img_embeds, txt_embeds)

    txt_id, gt = get_ground_truth(split, limit=limit)

    assert len(img_embeds) == len(gt), (len(img_embeds), len(gt))
    assert len(txt_embeds) == len(txt_id), (len(txt_embeds), len(txt_id))

    # Build mapping from image_id to all captions for that image
    # Also track the first index where each image_id appears
    image_id_to_captions = {}
    image_id_to_first_index = {}

    for i, entry in enumerate(gt):
        img_id = entry["image_id"]
        if img_id not in image_id_to_captions:
            image_id_to_captions[img_id] = set()
            image_id_to_first_index[img_id] = i
        image_id_to_captions[img_id].update(entry["captions"])

    recalls = []

    # Evaluate only once per unique image_id (using first occurrence)
    for img_id, first_idx in image_id_to_first_index.items():
        # Get all captions for this image_id
        gt_ids = image_id_to_captions[img_id]

        # Top-K text indices by similarity (use the first occurrence's embedding)
        top_k = np.argsort(-sim_matrix[first_idx])[:k]

        # Map to real text IDs
        predicted_txt_ids = [txt_id[idx] for idx in top_k]

        # Fractional recall: how many GTs are in top-K
        matched = len(set(predicted_txt_ids) & set(gt_ids))
        recall_i = matched / len(gt_ids)
        recalls.append(recall_i)

    return np.mean(recalls)


#Define target device
device = qai_hub.Device("XR2 Gen 2 (Proxy)")


inference_jobs = {}

for task_name, info in JOB_IDS:
    compiled_id = info["compiled_id"]
    input_dataset = qai_hub.get_dataset(info["dataset_id"])

    job = qai_hub.get_job(compiled_id)
    compiled_model = job.get_target_model()

    print(f"Submitting inference for {task_name} model {compiled_model.model_id} on device {device.name}")

    inference_id = run_inference(compiled_model, device, input_dataset)
    inference_jobs[task_name] = qai_hub.get_job(inference_id)

# Then collect outputs
outputs = {}

for task_name, inference_job in inference_jobs.items():
    inference_output = inference_job.download_output_data()  # waits here
    outputs[task_name] = inference_output["output_0"]

text_output = outputs["text"]
image_output = outputs["image"]

result = evaluate_track1(image_output, text_output, SPLIT, limit=LIMIT)
print(result)

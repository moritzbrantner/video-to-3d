"""Local Grounding DINO + SAM 2.1 semantic sidecar (protocol video-to-3d-semantic-sidecar/1).

Invoked by the Rust `semantic_inventory` operation as

    video-to-3d-semantic-sidecar <request.json> <response.json>

It reads the browser keyframes the request names (raw RGBA8 on local disk),
proposes vocabulary-labelled boxes with Grounding DINO, segments them with
SAM 2.1, links them across frames by propagating each candidate's mask with
the SAM 2.1 video predictor, and writes detections, masks, tracks and honest
gaps. Rust validates everything before it is recorded. Inference is local:
weights come from the Hugging Face cache, fetched once on first need.
Without a CUDA device it answers `unavailable` (Atlas 4.6).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROTOCOL = "video-to-3d-semantic-sidecar/1"
DETECTOR = "IDEA-Research/grounding-dino-tiny"
SEGMENTER = "facebook/sam2.1-hiera-small"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
# A propagated mask must overlap a same-label detection this much to link it.
TRACK_IOU = 0.5
# Masks whose SAM 2.1 IoU prediction is below this are kept but reported weak.
WEAK_MASK = 0.5


def write(path: str, value: dict) -> None:
    Path(path).write_text(json.dumps(value))


def unavailable(path: str, reason: str) -> None:
    write(path, {"protocol": PROTOCOL, "status": "unavailable", "reason": reason})


def rle(mask) -> list[int]:
    """Row-major run lengths alternating background/foreground, background first."""
    import numpy as np

    flat = np.asarray(mask, dtype=bool).reshape(-1)
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], changes, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        runs = [0] + runs
    return [int(run) for run in runs]


def clip_mask_to_box(mask, box):
    """Keep only foreground pixels that intersect the detection box."""
    import numpy as np

    height, width = mask.shape
    x0, y0, x1, y1 = box
    columns = np.arange(width)
    rows = np.arange(height)
    keep_columns = (columns + 1 > x0) & (columns < x1)
    keep_rows = (rows + 1 > y0) & (rows < y1)
    return mask & keep_rows[:, None] & keep_columns[None, :]


def label_for(phrase: str, terms: list[str]) -> str | None:
    """Map a detector phrase back to the vocabulary term it came from."""
    phrase = phrase.strip().lower()
    if not phrase:
        return None
    exact = [term for term in terms if term.lower() == phrase]
    if exact:
        return exact[0]
    contained = [term for term in terms if term.lower() in phrase or phrase in term.lower()]
    return max(contained, key=len) if contained else None


def mask_iou(first, second) -> float:
    union = (first | second).sum()
    return float((first & second).sum() / union) if union else 0.0


def main(request_path: str, response_path: str) -> None:
    request = json.loads(Path(request_path).read_text())
    if request.get("protocol") != PROTOCOL:
        raise SystemExit(f"unsupported protocol {request.get('protocol')!r}")
    import numpy as np
    import torch
    import transformers
    from PIL import Image

    if not torch.cuda.is_available():
        unavailable(
            response_path,
            "no CUDA device is available; local segmentation needs a GPU, so the "
            "observation-only build continues without semantic evidence",
        )
        return
    device = "cuda"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    terms: list[str] = request["vocabulary"]
    frames = request["frames"]
    images = []
    for frame in frames:
        width, height = frame["width"], frame["height"]
        rgba = np.frombuffer(Path(frame["rgba_path"]).read_bytes(), dtype=np.uint8)
        images.append(Image.fromarray(rgba.reshape(height, width, 4)[:, :, :3].copy()))

    detector_processor = transformers.AutoProcessor.from_pretrained(DETECTOR)
    detector = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(device)
    detector_version = getattr(detector.config, "_commit_hash", None) or transformers.__version__
    prompt = ". ".join(term.lower() for term in terms) + "."
    detections = []  # (frame position, label, box, score)
    with torch.inference_mode():
        for position, image in enumerate(images):
            inputs = detector_processor(images=image, text=prompt, return_tensors="pt").to(device)
            outputs = detector(**inputs)
            result = detector_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[image.size[::-1]],
            )[0]
            labels = result.get("text_labels", result.get("labels"))
            for box, score, phrase in zip(result["boxes"].tolist(), result["scores"].tolist(), labels):
                label = label_for(str(phrase), terms)
                width, height = image.size
                x0, y0, x1, y1 = (
                    max(0.0, box[0]),
                    max(0.0, box[1]),
                    min(float(width), box[2]),
                    min(float(height), box[3]),
                )
                if label is None or not (x0 < x1 and y0 < y1):
                    continue
                detections.append((position, label, [x0, y0, x1, y1], float(score)))
    del detector
    torch.cuda.empty_cache()

    segmenter_processor = transformers.Sam2Processor.from_pretrained(SEGMENTER)
    segmenter = transformers.Sam2Model.from_pretrained(SEGMENTER).to(device)
    segmenter_version = getattr(segmenter.config, "_commit_hash", None) or transformers.__version__
    masks: list = [None] * len(detections)
    mask_scores: list = [None] * len(detections)
    with torch.inference_mode():
        for position, image in enumerate(images):
            members = [index for index, item in enumerate(detections) if item[0] == position]
            if not members:
                continue
            boxes = [[detections[index][2] for index in members]]
            inputs = segmenter_processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
            outputs = segmenter(**inputs, multimask_output=False)
            predicted = segmenter_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]
            scores = outputs.iou_scores.cpu().reshape(len(members), -1)[:, 0].tolist()
            for slot, index in enumerate(members):
                mask = clip_mask_to_box(predicted[slot, 0].numpy().astype(bool), detections[index][2])
                if mask.any():
                    masks[index] = mask
                    mask_scores[index] = float(min(max(scores[slot], 0.0), 1.0))
    del segmenter
    torch.cuda.empty_cache()

    # Tracks: propagate every unassigned candidate's mask through all frames
    # with the SAM 2.1 video predictor and link same-label detections it covers.
    video_processor = transformers.Sam2VideoProcessor.from_pretrained(SEGMENTER)
    video_model = transformers.Sam2VideoModel.from_pretrained(SEGMENTER).to(device)
    track_of = [None] * len(detections)
    tracks = []
    order = sorted(
        (index for index in range(len(detections)) if masks[index] is not None),
        key=lambda index: -detections[index][3],
    )
    with torch.inference_mode():
        for seed in order:
            if track_of[seed] is not None:
                continue
            position, label, box, _ = detections[seed]
            session = video_processor.init_video_session(
                video=images, inference_device=device, dtype=torch.float32
            )
            video_processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=position, obj_ids=1, input_boxes=[[box]]
            )
            video_model(inference_session=session, frame_idx=position)
            propagated = {}
            for reverse in (False, True):
                for output in video_model.propagate_in_video_iterator(
                    session, start_frame_idx=position, reverse=reverse
                ):
                    mask = video_processor.post_process_masks(
                        [output.pred_masks], original_sizes=[[images[0].size[1], images[0].size[0]]], binarize=True
                    )[0]
                    propagated[output.frame_idx] = mask[0, 0].cpu().numpy().astype(bool)
            members = [seed]
            overlaps = []
            for other in order:
                if other == seed or track_of[other] is not None:
                    continue
                other_position, other_label, _, _ = detections[other]
                if other_label != label or other_position == position:
                    continue
                if any(detections[member][0] == other_position for member in members):
                    continue
                candidate = propagated.get(other_position)
                if candidate is None:
                    continue
                overlap = mask_iou(candidate, masks[other])
                if overlap >= TRACK_IOU:
                    members.append(other)
                    overlaps.append(overlap)
            if len(members) >= 2:
                for member in members:
                    track_of[member] = len(tracks)
                tracks.append((label, sorted(members, key=lambda index: detections[index][0]), float(np.mean(overlaps))))

    observations = []
    uncertainties = []
    for index, (position, label, box, score) in enumerate(detections):
        frame_index = frames[position]["frame_index"]
        mask = masks[index]
        observations.append(
            {
                "id": f"obs-{index}",
                "frame_index": frame_index,
                "label": label,
                "bbox": box,
                "detection_confidence": min(max(score, 0.0), 1.0),
                "mask": None
                if mask is None
                else {"encoding": "rle_row_major", "width": mask.shape[1], "height": mask.shape[0], "counts": rle(mask)},
                "mask_confidence": mask_scores[index],
            }
        )
        if mask is None:
            uncertainties.append(
                {"kind": "weak", "frame_index": frame_index, "label": label, "observation": f"obs-{index}",
                 "reason": "SAM 2.1 produced no mask inside the detection box"}
            )
        elif mask_scores[index] < WEAK_MASK:
            uncertainties.append(
                {"kind": "weak", "frame_index": frame_index, "label": label, "observation": f"obs-{index}",
                 "reason": f"SAM 2.1 predicted mask IoU {mask_scores[index]:.2f} is below {WEAK_MASK}"}
            )
    # Frames where a tracked object is expected (inside its span) but missing.
    track_entries = []
    for number, (label, members, confidence) in enumerate(tracks):
        track_entries.append(
            {"id": f"track-{number}", "label": label, "observations": [f"obs-{member}" for member in members],
             "association_confidence": min(max(confidence, 0.0), 1.0)}
        )
        seen = {detections[member][0] for member in members}
        for position in range(min(seen), max(seen) + 1):
            if position not in seen:
                uncertainties.append(
                    {"kind": "missing", "frame_index": frames[position]["frame_index"], "label": label,
                     "observation": None, "reason": f"track-{number} has no linked detection in this frame"}
                )

    write(
        response_path,
        {
            "protocol": PROTOCOL,
            "status": "available",
            "models": [
                {"role": "detector", "id": DETECTOR, "version": detector_version,
                 "url": f"https://huggingface.co/{DETECTOR}"},
                {"role": "segmenter", "id": SEGMENTER, "version": segmenter_version,
                 "url": f"https://huggingface.co/{SEGMENTER}"},
            ],
            "runtime": {
                "name": "pytorch",
                "version": torch.__version__,
                "device": device,
                "device_name": torch.cuda.get_device_name(0),
            },
            "observations": observations,
            "tracks": track_entries,
            "uncertainties": uncertainties,
            "measurements": {
                "elapsed_seconds": time.monotonic() - started,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            },
        },
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: video-to-3d-semantic-sidecar <request.json> <response.json>")
    main(sys.argv[1], sys.argv[2])

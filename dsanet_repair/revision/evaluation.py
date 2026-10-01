"""Keep the official metric path and export replayable clip-level predictions."""

import hashlib
import importlib
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from dsanet_repair.numerics import fresh_evaluation
from .cache import save_atomic


class Recorder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.rows = []

    def forward(self, *args, **kwargs):
        # The official test functions make one model call per video but split a
        # long video into batch rows of maxlen chunks. Stateful follow-up
        # modules must see those rows as one video, not independent videos.
        stateful = hasattr(self.model, "official_video_chunks")
        previous = self.model.official_video_chunks if stateful else None
        if stateful:
            self.model.official_video_chunks = True
        try:
            output = self.model(*args, **kwargs)
        finally:
            if stateful:
                self.model.official_video_chunks = previous
        self.rows.append((output[1].detach().float().cpu().reshape(-1).numpy(),
                          output[2].detach().float().cpu().reshape(-1, output[2].shape[-1]).numpy()))
        return output


def frame_diagnostics(binary_rows, probabilities, ground_truth):
    if len(binary_rows) != len(probabilities):
        raise ValueError("Per-video prediction count mismatch")
    offset, within_auc, within_ap, normal_count, normal_false = 0, [], [], 0, 0
    for raw, prob in zip(binary_rows, probabilities):
        size = len(prob)
        target = np.asarray(ground_truth[offset:offset + size])
        score = np.repeat(torch.as_tensor(raw[:size // 16]).sigmoid().numpy(), 16)
        if len(target) != size or len(score) != size:
            raise ValueError("Prediction/ground-truth timeline mismatch")
        offset += size
        if len(np.unique(target)) == 2:
            within_auc.append(float(roc_auc_score(target, score)))
            within_ap.append(float(average_precision_score(target, score)))
        elif not target.any():
            normal_count += size
            normal_false += int((score >= .5).sum())
    if offset != len(ground_truth):
        raise ValueError("Unconsumed frame ground truth")
    return {"within_video_auc_macro": float(np.mean(within_auc)) if within_auc else None,
            "within_video_ap_macro": float(np.mean(within_ap)) if within_ap else None,
            "mixed_videos": len(within_auc), "all_videos": len(probabilities),
            "normal_frame_fpr_at_0_5": normal_false / normal_count if normal_count else None,
            "normal_frames": normal_count,
            "note": "Macro within-video metrics use only videos containing both frame labels; threshold 0.5 is fixed, not optimized."}


def abnormal_event_metrics(predictions, segments, labels, class_order, nms):
    """Same proposal rule as DSANet; omit NORMAL CLASS, retain normal VIDEOS.

    This diagnostic differs from historical include-normal mAP, so report it
    separately. No-proposal classes have AP=0, not an early whole-table zero.
    """
    if not len(predictions) == len(segments) == len(labels):
        raise ValueError("Per-video detection annotation count mismatch")
    ious = [.1, .2, .3, .4, .5]
    per_class = {}
    for channel, name in enumerate(class_order[1:], 1):
        proposals = []
        truth = {}
        for video, (scores, gt_intervals, gt_classes) in enumerate(zip(predictions, segments, labels)):
            tmp = scores[:, channel]
            top_count = max(1, int(len(tmp) / 16))
            class_score = np.sort(tmp)[::-1][:top_count].mean()
            if class_score <= 0:
                tmp = tmp * 0
            local = []
            for threshold_fraction in np.arange(.6, .7, .1):
                threshold = tmp.max() - (tmp.max() - tmp.min()) * threshold_fraction
                binary = np.r_[0, (tmp > threshold).astype(np.int32), 0]
                starts, ends = np.where(np.diff(binary) == 1)[0], np.where(np.diff(binary) == -1)[0]
                local.extend([start, end, float(tmp[start:end].max() + .7 * class_score)]
                             for start, end in zip(starts, ends) if end - start >= 2)
            if local:
                local = np.asarray(local)
                local = local[np.argsort(-local[:, 2])]
                _, keep = nms(local, .6)
                proposals.extend((video, *local[index]) for index in keep)
            truth[video] = [(int(left), int(right)) for (left, right), category in zip(gt_intervals, gt_classes)
                            if str(category) == name]
        positives = sum(map(len, truth.values()))
        if not positives:
            per_class[name] = [None] * len(ious)
            continue
        proposals.sort(key=lambda value: -value[3])
        values = []
        for threshold in ious:
            remaining = {video: list(intervals) for video, intervals in truth.items()}
            true_positives, precision_sum = 0, 0.
            for rank, (video, start, end, _score) in enumerate(proposals, 1):
                overlaps = []
                for left, right in remaining[video]:
                    intersection = max(0, min(end, right) - max(start, left))
                    union = end - start + right - left - intersection
                    overlaps.append(intersection / union if union > 0 else 0.)
                if overlaps and max(overlaps) >= threshold:
                    remaining[video].pop(int(np.argmax(overlaps)))
                    true_positives += 1
                    precision_sum += true_positives / rank
            values.append(100 * precision_sum / positives)
        per_class[name] = values
    supported = [values for values in per_class.values() if values[0] is not None]
    mean = np.mean(supported, axis=0).tolist() if supported else [None] * len(ious)
    return {"iou": ious, "map": mean, "average_map": float(np.mean(mean)) if supported else None,
            "per_class_ap": per_class, "normal_videos_retained": True, "normal_class_included": False}


def evaluate(model, dataset, cfg, label_map, test_module, data_module, tools, device, output_dir=None):
    cls = data_module.UCFDataset if dataset == "ucf" else data_module.XDDataset
    data = cls(cfg.visual_length, cfg.test_list, True, label_map)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    gt = np.load(cfg.gt_path)
    segments = np.load(cfg.gt_segment_path, allow_pickle=True)
    categories = np.load(cfg.gt_label_path, allow_pickle=True)
    prompt = tools.get_prompt_text(label_map)
    recorder, captured = Recorder(model), {}
    original_map = test_module.dmAP

    def capture(predictions, supplied_segments, supplied_labels, excludeNormal=False):
        result = original_map(predictions, supplied_segments, supplied_labels, excludeNormal)
        captured.update({"predictions": predictions, "map": list(map(float, result[0])),
                         "iou": list(map(float, result[1]))})
        return result

    test_module.dmAP = capture
    try:
        with fresh_evaluation(model):
            if dataset == "ucf":
                auc, ap = test_module.test(recorder, loader, cfg.visual_length, prompt, gt, segments,
                                          categories, False, device, cfg)
            else:
                auc, ap, _ = test_module.test(recorder, loader, cfg.visual_length, prompt, gt, segments,
                                             categories, False, cfg, device)
    finally:
        test_module.dmAP = original_map
    if not captured or len(recorder.rows) != len(data):
        raise RuntimeError("Official evaluation did not produce the expected per-video predictions")
    metrics = {"auc": float(auc), "ap": float(ap), "primary": float(auc if dataset == "ucf" else ap),
               "official_iou": captured["iou"], "official_map": captured["map"],
               "official_average_map": float(np.mean(captured["map"])),
               "official_map_normal_class_included": True}
    metrics["frame_diagnostics"] = frame_diagnostics([row[0] for row in recorder.rows], captured["predictions"], gt)
    map_module = importlib.import_module(f"utils.{dataset}_detectionMAP")
    metrics["abnormal_event"] = abnormal_event_metrics(captured["predictions"], segments, categories,
                                                       list(label_map), map_module.nms)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        prediction_dir = output / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        manifest = []
        for index, ((binary, semantic), probability) in enumerate(zip(recorder.rows, captured["predictions"])):
            length = len(probability) // 16
            filename = f"{index:04d}.npz"
            target = prediction_dir / filename
            np.savez_compressed(target, binary_logits=binary[:length], semantic_logits=semantic[:length],
                                refined_probabilities=probability[::16], frame_stride=np.int64(16))
            manifest.append({"index": index, "video": Path(data.paths[index]).name,
                             "file": filename, "clips": length})
        save_atomic({"dataset": dataset, "temperature": cfg.temp, "rows": manifest,
                     "test_csv_sha256": hashlib.sha256(Path(cfg.test_list).read_bytes()).hexdigest()},
                    prediction_dir / "index.json")
        save_atomic(metrics, output / "metrics.json")
    return metrics

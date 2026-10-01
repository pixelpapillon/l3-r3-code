"""Preserve official metric formulas; mask event support to true video length.

The 2026-09-28 fix can change event predictions on exact-multiple-length videos.
It is an inference correction, NOT merely an exporter formatting change.
"""

import numpy as np
import torch
from pathlib import Path

from dsanet_repair.revision.evaluation import evaluate as official_evaluate
from dsanet_repair.revision.cache import save_atomic


class _LengthAwareTest:
    """Relay raw dataset length before upstream mutates its chunk counter.

    Official test/dmAP functions remain unchanged. The shared evaluator hooks
    dmAP on this proxy; forwarding that assignment is essential because the
    original test function resolves dmAP in its own module globals.
    """
    def __init__(self, module, model):
        self.module, self.model = module, model

    @property
    def dmAP(self):
        return self.module.dmAP

    @dmAP.setter
    def dmAP(self, value):
        self.module.dmAP = value

    def test(self, recorder, loader, *args, **kwargs):
        model = self.model

        class LengthLoader:
            def __len__(self):
                return len(loader)

            def __iter__(self):
                for item in loader:
                    raw_length = torch.as_tensor(item[2])
                    if raw_length.numel() != 1:
                        raise ValueError("Event evaluation requires one video per loader item")
                    model.official_feature_length = int(raw_length.item())
                    yield item

        previous = model.official_feature_length
        try:
            return self.module.test(recorder, LengthLoader(), *args, **kwargs)
        finally:
            model.official_feature_length = previous


def evaluate(model, dataset, cfg, labels, test_module, data_module, utils, device, output_dir=None):
    descriptions, timeline_audit = [], []

    def capture(current, inputs, _output):
        if current.official_feature_length is None:
            raise ValueError("Official event evaluation lost the true video length")
        legacy_length = int(torch.as_tensor(inputs[3]).sum())
        timeline_audit.append({"index": len(timeline_audit), "feature_clips": current.official_feature_length,
                               "upstream_marked_clips": legacy_length})
        if current.event_core is None or current.event_state is None:
            return
        _hidden, _text, _centers, _widths, valid, bounds = current.event_state
        if bounds is not None:
            # Recorder already set official_video_chunks=True for this forward.
            descriptions.append({"intervals": bounds.detach().float().cpu().numpy(),
                                 "clips": int(valid.sum())})

    handle = model.register_forward_hook(capture)
    try:
        metrics = official_evaluate(model, dataset, cfg, labels, _LengthAwareTest(test_module, model),
                                    data_module, utils, device, output_dir)
    finally:
        handle.remove()
    metrics["event_timeline"] = {
        "protocol": "true-feature-length-event-support-v2",
        "videos": len(timeline_audit),
        "upstream_length_mismatches": sum(r["feature_clips"] != r["upstream_marked_clips"] for r in timeline_audit),
        "note": "Only cross-chunk event support corrected; frozen anchor and official score formulas unchanged"}
    if output_dir is not None:
        directory = Path(output_dir)
        rows = []
        if descriptions:
            destination = directory / "event_hypotheses"
            destination.mkdir(exist_ok=True)
            for index, record in enumerate(descriptions):
                file = f"{index:04d}.npz"
                np.savez_compressed(destination / file, intervals=record["intervals"], clips=record["clips"])
                rows.append({"index": index, "file": file, "clips": record["clips"]})
        # Keep this distinct from the existing all-normal-video FPR.
        index = directory / "predictions/index.json"
        if index.exists():
            import json
            prediction_index = json.loads(index.read_text())
            if descriptions:
                if len(descriptions) != len(prediction_index["rows"]):
                    raise ValueError("Event/prediction video count mismatch")
                for event, prediction in zip(rows, prediction_index["rows"]):
                    # Both counts are CLIP steps: official frame predictions
                    # have already been divided by stride 16 in the manifest.
                    if event["clips"] != prediction["clips"]:
                        raise ValueError("Event/prediction clip count mismatch after true-length masking")
                    event["video"] = prediction["video"]
                    event["prediction_clips"] = prediction["clips"]
            gt = np.load(cfg.gt_path)
            offset, count, false = 0, 0, 0
            for row in prediction_index["rows"]:
                with np.load(index.parent / row["file"]) as prediction:
                    stride = int(prediction["frame_stride"])
                    score = np.repeat(torch.from_numpy(prediction["binary_logits"]).sigmoid().numpy(), stride)
                truth = gt[offset:offset + len(score)]
                if len(truth) != len(score):
                    raise ValueError("Background diagnostic frame alignment mismatch")
                offset += len(score)
                if truth.any() and not truth.all():
                    background = truth == 0
                    count += int(background.sum())
                    false += int(((score >= .5) & background).sum())
            if offset != len(gt):
                raise ValueError("Unconsumed background-diagnostic ground truth")
            metrics["abnormal_video_background"] = {"frames": count,
                "fpr_at_0_5": false / count if count else None,
                "note": "Background frames inside mixed-label abnormal videos; fixed threshold"}
            save_atomic(metrics, directory / "metrics.json")
        if descriptions:
            save_atomic({"class_order": list(labels)[1:], "class_names": list(labels.values())[1:],
                         "coordinate": "normalized original-feature time",
                         "note": "Soft candidate intervals, not GT or official thresholded detections", "rows": rows},
                        directory / "event_hypotheses/index.json")
        save_atomic({"protocol": metrics["event_timeline"]["protocol"], "rows": timeline_audit},
                    directory / "event_timeline_audit.json")
        save_atomic(metrics, directory / "metrics.json")
    return metrics

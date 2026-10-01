"""Unchanged official metrics plus compact, label-free R mechanism records."""

import json
from pathlib import Path
import torch
from dsanet_repair.eventstudy.evaluation import evaluate as event_evaluate
from dsanet_repair.revision.cache import save_atomic
from .modules import CoreEnvelope


def evaluate(model, dataset, cfg, labels, module, data, utils, device, output_dir=None):
    records = []

    def capture(current, _inputs, _output):
        if not isinstance(current.event_core, CoreEnvelope):
            return
        state = current.event_core.last
        _hidden, _text, centers, widths, valid, _bounds = current.event_state
        core = state["core"].detach()
        effective = 1 / (core.square() / widths[:, None, None].clamp_min(1e-8)).sum(-1).clamp_min(1e-8)
        records.append(dict(index=len(records), clips=int(valid.sum()),
                            evidence_centers=(core * centers[:, None, None]).sum(-1).cpu().tolist(),
                            evidence_effective_width=effective.cpu().tolist(),
                            core_to_envelope_fraction=state["concentration"].detach().cpu().tolist(),
                            selected_slots=state["selection"].detach().cpu().tolist(),
                            diagnostics={k: float(v) for k, v in current.module_diagnostics.items()}))

    handle = model.register_forward_hook(capture)
    try:
        metrics = event_evaluate(model, dataset, cfg, labels, module, data, utils, device, output_dir)
    finally:
        handle.remove()
    if output_dir is not None:
        prediction_file = Path(output_dir) / "predictions/index.json"
        if records and prediction_file.is_file():
            predictions = json.loads(prediction_file.read_text())["rows"]
            if len(predictions) != len(records):
                raise ValueError("Mechanism/prediction video count mismatch")
            for row, prediction in zip(records, predictions):
                if row["clips"] != prediction["clips"]:
                    raise ValueError("Mechanism/prediction timeline mismatch")
                row["video"] = prediction["video"]
        save_atomic(dict(class_order=list(labels)[1:], class_names=list(labels.values())[1:],
                         coordinate="normalized original-feature time",
                         note="Candidate diagnostics, NOT ground truth or official detections", rows=records,
                         readout_reference=model.readout_reference.detach().cpu().tolist()),
                    Path(output_dir) / "mechanism_diagnostics.json")
    return metrics

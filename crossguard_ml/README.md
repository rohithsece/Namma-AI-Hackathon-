# CROSSGUARD AI — ML package

This package contains an actual pretrained checkpoint and an actual exported ExecuTorch model, plus offline inference, per-frame counting, saved detection images, and measured timing reports. It is scoped to the ML and postprocessing work; no dashboard or hardware electronics are required.

**Status:** executable development baseline. The supplied model is SSDLite320 MobileNetV3 Large with the original torchvision COCO_V1 weights. It was not newly trained on the project poster or Indian road footage. The mandatory requirements explicitly call for an approved pretrained detector; new supervised training is optional and needs labeled data and organiser permission. Organiser approval of this particular architecture has not been verified.

## Start here

1. Extract this ZIP and open a terminal inside `crossguard_ml`.
2. For a Linux PC, WSL Ubuntu, or 64-bit Raspberry Pi OS with Python and venv installed:

```bash
bash scripts/setup_linux.sh
```

This installs dependencies online once and runs the two included public test photos. Model inference itself uses only bundled files and works offline after installation. Python 3.12 is tested; the required packages must have wheels for your Python/OS combination. Setup on the actual Pi has not been tested here. Use a new output directory for each run.

3. Activate the environment for subsequent runs:

```bash
source .venv/bin/activate
```

4. Run the organiser's clip on the Raspberry Pi:

```bash
python run.py --source data/organiser_clip.mp4 --output outputs/organiser_pi --backend executorch --provenance organiser
```

For 10–20 organiser-supplied images instead:

```bash
python run.py --source data/organiser_images --output outputs/organiser_pi_images --backend executorch --provenance organiser
```

The input paths are examples: put the actual provided files there. `--provenance organiser` records your declaration; it cannot authenticate the source. Use the entire supplied 10–20 second clip (default stride 1), or the full provided image set. Video decoding can take longer than playback time; do not claim real-time operation without measuring it.

## Windows development preview

Install Python 3.12 **64-bit**, including the Python launcher, from https://www.python.org/downloads/windows/. In PowerShell inside the extracted project:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_torch.ps1
```

This explicit PyTorch preview is for Windows development. For the mandatory ExecuTorch run, use Raspberry Pi OS; WSL Ubuntu can also be used for development/export. The native Windows setup script was not executed in this environment. A Windows PyTorch result does not satisfy the Raspberry Pi + ExecuTorch requirement.

## What is included

| File | Purpose |
|---|---|
| `models/crossguard_ssdlite_fp32.pte` | Ready-exported FP32 ExecuTorch XNNPACK model; approximately 13.85 MB |
| `models/ssdlite320_mobilenet_v3_large_coco.pth` | Original trained torchvision checkpoint; approximately 14.07 MB |
| `run.py` | Images/video → detections → per-frame counts → annotated JPEGs and reports |
| `reports/sample_outputs/` | Two executed public-photo examples and the actual host measurement report |
| `reports/export_validation.json` | Export, runtime load, random-input numerical parity results |
| `reports/real_image_parity.json` | Numerical parity checks on the included real photos |
| `crossguard/pipeline.py` | Preprocessing, per-class NMS, counting, optional zone analysis |
| `export_model.py` | Rebuild an artifact from the original or separately fine-tuned checkpoint |
| `train.py`, `evaluate.py` | Optional supervised fine-tuning and labeled COCO evaluation |
| `tests/` | NMS, scaling, zero detections, zone logic, preprocessing and model-wrapper checks |
| `scripts/check_files.py`, `manifest.json` | Verify packaged file hashes |

## Mandatory deliverables mapping

| Requirement you supplied | Package support | Remaining evidence |
|---|---|---|
| Approved pretrained detector, organiser 10–20 second clip or 10–20 images, ExecuTorch on Pi | Actual `.pte`, local checkpoint, file-based runner | Confirm approved architecture; obtain organiser input; run on your Pi |
| At least one category and counts for each selected frame/image | Six supported classes; terminal JSON and `counts.csv` | Run against the supplied input |
| Category/counts plus at least two detection outputs | All frames saved under `annotated/`; two public demonstration outputs included | Submit at least two **organiser-input** outputs from the Pi run |
| Mean inference time, model, approximate throughput | `report.json` and `report.md`, warm-up excluded, hardware/backend recorded | Use the report measured on your actual Pi |
| Explain issue, benefit, application in at most 2 minutes | `TWO_MINUTE_EXPLANATION.md` | Present it alongside your measured demonstration |

The included public examples are development evidence, not a replacement for the organiser's required input. No hackathon completion, approved-model status, Raspberry Pi speed or road-scene accuracy is claimed.

## Supported categories

| Output | Native COCO ID |
|---|---:|
| person | 1 |
| bicycle | 2 |
| car | 3 |
| motorcycle | 4 |
| bus | 6 |
| truck | 8 |

“Bike” is kept separate as bicycle/motorcycle. Auto-rickshaws, ambulances, helmets, number plates and traffic lights are not dedicated output classes in this package. The checkpoint retains the full 91-index COCO head, and inference returns the six selected categories. Class probabilities are computed with the complete softmax before selecting columns.

## Pipeline and model contract

The runner reads BGR images, resizes directly to 320×320 with OpenCV bilinear interpolation, converts to RGB, and forms float32 input `[1,3,320,320]` in `[0,1]`. The exported graph performs normalization `(input - 0.5) / 0.5`, the detector backbone/head, anchor decoding and clipping. It returns boxes `[3234,4]` in 320-pixel xyxy space and selected class scores `[3234,6]`.

Python postprocessing applies a confidence threshold (default 0.45), a per-class top-300 candidate cap, NMS (IoU 0.55), rescales to the original image size and keeps at most 100 detections. NMS and zone logic run locally outside the neural-network graph. The annotated output adds a top banner without cutting off source pixels.

Counts represent visible instances **per frame**. Summing across a video double-counts the same moving object. This version does not track identities or produce unique vehicle flow totals. It runs the input files fully offline after dependencies have been installed; it contains no cloud inference API.

## Outputs and honest measurements

Each run creates:

- `annotated/*.jpg`: all processed frame/image outputs, including zero-detection frames.
- `counts.csv`: source, frame index, timestamp where known, six category counts, optional alert state and inference time.
- `detections.jsonl`: boxes, confidence scores, categories, counts and optional zone details.
- `report.json` and `report.md`: model, hash, runtime/backend, device, versions, settings and timing.

`average_inference_ms` measures the synchronous model execute/forward call, excluding preprocessing, NMS, annotation, file output and three warm-up calls. `inference_only_fps` is 1000 divided by that mean. The separate processing throughput includes input decoding, annotation and writes. It is neither camera latency nor proof of live real-time performance. The two-image sample run is only a smoke test; collect a larger organiser run for a useful average. `--threads` records the PyTorch setting; ExecuTorch delegate scheduling depends on the runtime build.

The public bus example detects the main bus but misses people at the image edges; the crossing example detects two foreground pedestrians while some distant objects are missed. A small, 320-pixel detector has practical limits on distant, occluded and low-light objects. No local mAP or accuracy percentage is reported without ground truth.

## Optional crossing-zone heuristic

After calibrating polygons for a fixed camera, copy `configs/zones.example.json` and replace its normalized coordinates. Use original input image coordinates divided by original width/height; exclude the annotation banner. The example polygons are illustrative, not calibrated to the supplied photos.

```bash
python run.py --source data/organiser_clip.mp4 --output outputs/organiser_zones --zones configs/my_camera.json --provenance organiser
```

The bottom-center of each detection is used as a ground-contact approximation. A potential conflict is flagged when a person occupies the pedestrian polygon and at least one bicycle/motorcycle/car/bus/truck occupies the vehicle polygon **in the same frame**. Alerts are disabled unless a zone file is supplied. This is a rule, not a learned safety classifier. It does not estimate speed, direction, time to collision, right of way, or collision probability, and static co-occupancy may generate false alerts. It supports a monitoring PoC; it should not control traffic or provide a safety guarantee.

## Reproduce checks or export

```bash
python -m pip install -r requirements-training.txt
python -m pytest -q
python verify_model.py
python export_model.py
python scripts/check_files.py
```

Re-exporting or regenerating reports changes hashes; the packaged manifest describes the delivered version. The export compares the wrapper with the loaded ExecuTorch graph on three fixed random tensors, and `verify_model.py` checks the two real photos. This proves numerical fidelity of those inputs, not road-scene accuracy. The included test and export reports were generated on Linux x86_64. XNNPACK targets Arm and x86 CPUs; actual Raspberry Pi execution must still be measured. No INT8 artifact or calibration claim is included.

## Optional training, only after permission and labeled footage

Use organizer-approved pretrained weights as-is for the mandatory baseline. If fine-tuning is permitted, annotate representative real road images using COCO JSON. Use **native category IDs** from the table, not contiguous IDs 0–5. Every selected-class instance should be labeled, including small/occluded objects where visible. Split by camera/location/recording session before sampling frames; adjacent frames from the same recording must not leak across train/validation/test. Keep a separate untouched test set.

```bash
python train.py --train-images data/train/images --train-annotations data/train/annotations.json --val-images data/val/images --val-annotations data/val/annotations.json --output training/road_v1 --epochs 20 --batch-size 4
python evaluate.py --images data/test/images --annotations data/test/annotations.json --weights training/road_v1/best.pth --output reports/heldout_test.json
python export_model.py --weights training/road_v1/best.pth --output models/crossguard_finetuned.pte
python run.py --source data/test/images --output outputs/finetuned_test --model models/crossguard_finetuned.pte
```

The trainer checks duplicate image bytes across train/validation, uses seeded shuffling, horizontal flips, SGD and a cosine learning-rate schedule, and selects `best.pth` by validation COCO mAP. The byte check cannot detect adjacent-frame or same-session leakage; splitting correctly remains your responsibility. At least two samples per training batch are needed by BatchNorm; incomplete final batches are dropped. CUDA is used if available, otherwise CPU. Keep calibration/evaluation separate from training. A newly fine-tuned model must be re-approved if required by the rules.

`evaluate.py` uses torchvision's eager image transform and standard SSD postprocessing, whereas the deployed wrapper uses explicitly documented OpenCV preprocessing and selected-class postprocessing. Its mAP measures the eager model; it is not automatically an end-to-end ExecuTorch accuracy result. Validate deployed counts/boxes separately on held-out labeled input. Optional training code was integration-smoke-tested; no project-specific trained checkpoint or labeled road benchmark is included.

## Sources and data provenance

- Model and weight description: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- Original weight URL: https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth
- XNNPACK export: https://docs.pytorch.org/executorch/stable/ios-xnnpack.html
- Python runtime: https://docs.pytorch.org/executorch/stable/runtime-python-api-reference.html
- Raspberry Pi platform guidance: https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html (platform setup context; its model example is Llama, not this detector)

The model and dataset are credited in `THIRD_PARTY_NOTICES.md`; project-specific code is licensed separately in `LICENSE`. See `MODEL_CARD.json` and `reports/` for precise artifacts and validation scope.

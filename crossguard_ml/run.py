"""Offline CLI. No dashboard, cloud, implicit model downloads, or camera required."""
import argparse, csv, hashlib, importlib.metadata, json, platform, time
from pathlib import Path
import cv2
import numpy as np
import torch
from crossguard.model import EdgeDetector, load_detector, CLASSES, ROOT
from crossguard.pipeline import preprocess, postprocess, analyse, annotate, validate_zones

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}

def frames(source: str | Path, stride: int = 1):
    """Yield frames sequentially from a single image, image directory, or video file.
    
    Supports common image formats (.jpg, .jpeg, .png, .bmp, .webp) and video streams.
    When reading video, applies frame striding to sample at regular intervals.
    
    Args:
        source: File path to an image, a directory of images, or a video file (.mp4, etc.).
        stride: Frame sampling interval for video inputs (e.g. 1 = every frame, 3 = every 3rd frame).
        
    Yields:
        tuple[int, float | None, str, np.ndarray]:
            - index: Zero-indexed integer position of the frame.
            - timestamp: Elapsed video time in seconds (None for still images).
            - filename: Generated or original filename for the frame.
            - bgr: Decoded image matrix in OpenCV BGR uint8 format.
            
    Raises:
        ValueError: If input directory contains no supported images, an image cannot be read,
            or the video file cannot be opened.
    """
    p = Path(source)
    if p.is_dir():
        # Load all supported image files in alphabetical order
        files = sorted(x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f'No supported images in input directory: {p}')
        for i, f in enumerate(files):
            im = cv2.imread(str(f))
            if im is None:
                raise ValueError(f'Unreadable image: {f}')
            yield i, None, f.name, im
    elif p.suffix.lower() in IMAGE_EXTS:
        # Process single still image
        im = cv2.imread(str(p))
        if im is None:
            raise ValueError(f'Unreadable image: {p}')
        yield 0, None, p.name, im
    else:
        # Open and decode video file
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {p}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        i = 0
        try:
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    yield i, (i / fps if fps > 0 else None), f'{p.stem}_{i:06d}.jpg', im
                i += 1
        finally:
            cap.release()


def main():
    """Main execution routine for the offline inference and verification runner.
    
    Parses CLI flags, prepares output directories, instantiates the selected execution backend
    (ExecuTorch XNNPACK or PyTorch Eager), performs warmup, executes frame-by-frame inference,
    computes object counts and zone alerts, writes detailed logs and annotated JPEGs, and
    compiles timing and evidence reports into report.json and report.md.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='Image, directory, or video file')
    p.add_argument('--output',required=True,help='New or empty output folder')
    p.add_argument('--backend',choices=['executorch','torch'],default='executorch')
    p.add_argument('--model',type=Path,default=ROOT/'models/crossguard_ssdlite_fp32.pte')
    p.add_argument('--weights',type=Path,default=ROOT/'models/ssdlite320_mobilenet_v3_large_coco.pth')
    p.add_argument('--confidence',type=float,default=0.45)
    p.add_argument('--iou',type=float,default=0.55)
    p.add_argument('--stride',type=int,default=1,help='Video frame sampling interval')
    p.add_argument('--limit',type=int,default=0,help='0 means all frames/images')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--zones',type=Path,help='Camera-specific normalized zone polygons')
    p.add_argument('--provenance',choices=['organiser','public_sample','own'],default='own')
    a=p.parse_args()
    if a.stride<1 or a.threads<1 or a.limit<0 or not 0<a.confidence<=1 or not 0<a.iou<=1: p.error('Invalid range for numeric argument')
    if a.output and Path(a.output).exists() and any(Path(a.output).iterdir()): p.error('Output folder must be empty to avoid mixing runs')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); (out/'annotated').mkdir()
    torch.set_num_threads(a.threads)
    config=validate_zones(json.loads(a.zones.read_text())) if a.zones else None
    if a.backend=='executorch':
        from executorch.runtime import Runtime
        runtime=Runtime.get()
        program=runtime.load_program(a.model)
        method=program.load_method('forward')
        def forward(x): return method.execute([x])
        artifact=a.model
    else:
        detector=EdgeDetector(load_detector(a.weights)).eval()
        def forward(x): return detector(x)
        artifact=a.weights
    # Warm-up excluded from all aggregate timings; inference only means execute/forward.
    with torch.inference_mode():
        for _ in range(3): forward(torch.zeros(1,3,320,320))
    rows=[]; timings=[]
    started=time.perf_counter()
    with (out/'detections.jsonl').open('w') as detailed, torch.inference_mode():
        for index,stamp,name,bgr in frames(a.source,a.stride):
            if a.limit and len(rows)>=a.limit: break
            t0=time.perf_counter(); x=torch.from_numpy(preprocess(bgr)); t1=time.perf_counter()
            boxes,scores=forward(x); t2=time.perf_counter()
            det=postprocess(boxes.detach().cpu().numpy(),scores.detach().cpu().numpy(),bgr.shape[1],bgr.shape[0],a.confidence,a.iou)
            counts,zone=analyse(det,config,bgr.shape[1],bgr.shape[0]); t3=time.perf_counter()
            marked=annotate(bgr,det,counts,zone,config)
            output_name=f'{len(rows):06d}_{Path(name).stem}.jpg'
            if not cv2.imwrite(str(out/'annotated'/output_name),marked): raise OSError('Could not save output image')
            record={'source':name,'frame_index':index,'timestamp_seconds':stamp,'counts':counts,
                'zone_analysis':zone,'detections':det,'annotated_file':f'annotated/{output_name}'}
            detailed.write(json.dumps(record)+'\n')
            detailed.flush()
            row={'source':name,'frame_index':index,'timestamp_seconds':stamp,**counts,
                'potential_conflict':zone.get('potential_conflict'),'inference_ms':(t2-t1)*1000}
            rows.append(row); timings.append([(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000])
            print(json.dumps({'frame':index,'counts':counts,'inference_ms':round((t2-t1)*1000,2)}), flush=True)
    elapsed=time.perf_counter()-started
    if not rows: raise ValueError('No frames processed')
    if len((out/'detections.jsonl').read_text().splitlines()) != len(rows):
        raise OSError('Detailed output record count mismatch')
    with (out/'counts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    times=np.asarray(timings); mean_inf=float(times[:,1].mean())
    is_pi=False; board=None
    board_path=Path('/proc/device-tree/model')
    if board_path.exists():
        board=board_path.read_bytes().decode(errors='replace').strip('\x00'); is_pi='Raspberry Pi' in board
    versions={}
    for package in ['torch','torchvision','executorch','numpy','opencv-python-headless']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    report={'model':'torchvision SSDLite320 MobileNetV3 Large COCO_V1','model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'backend':a.backend,'precision':'FP32','classes':CLASSES,'input_shape':[1,3,320,320],
        'num_processed':len(rows),'input_provenance':a.provenance,'confidence':a.confidence,'nms_iou':a.iou,'video_stride':a.stride,
        'counts_semantics':'per-frame visible instances; do not sum across video frames as unique traffic',
        'average_preprocess_ms':float(times[:,0].mean()),'average_inference_ms':mean_inf,
        'p50_inference_ms':float(np.percentile(times[:,1],50)),'p95_inference_ms':float(np.percentile(times[:,1],95)),
        'average_postprocess_ms':float(times[:,2].mean()),'inference_only_fps':1000/mean_inf,
        'processed_fps_including_decode_annotation_and_writes':len(rows)/elapsed,'measured_seconds':elapsed,
        'warmup_runs_excluded':3,'threads_requested':a.threads,
        'thread_note':'PyTorch thread setting recorded; ExecuTorch delegate thread behavior may differ by build',
        'host':{'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'board':board,'python':platform.python_version()},
        'versions':versions,'raspberry_pi_detected':is_pi,'organiser_approved_model':'not verified; check organiser model list',
        'mandatory_evidence':{'executorch':a.backend=='executorch','raspberry_pi_run':is_pi,'organiser_input':a.provenance=='organiser',
            'at_least_two_outputs':len(rows)>=2,'organiser_image_count_or_clip_duration':'verify manually against original source'},
        'accuracy_metrics':None,'accuracy_note':'No ground-truth labels supplied; no local mAP/precision/recall claimed.',
        'zone_alerts':'disabled' if config is None else 'heuristic co-occupancy only; camera calibration required'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'report.md').write_text(f"# CROSSGUARD AI measurement\n\nModel: {report['model']}\n\nBackend: {a.backend}; precision: FP32\n\nProcessed inputs: {len(rows)}\n\nMean inference: {mean_inf:.2f} ms; inference throughput: {1000/mean_inf:.2f} FPS\n\nProcessing throughput including file output: {len(rows)/elapsed:.2f} FPS\n\nRaspberry Pi detected: {is_pi}\n\nInput provenance declared by runner: {a.provenance}\n\nCounts are per frame, not unique traffic totals. See report.json for full measurement scope and evidence status.\n")
    print(f'Saved {len(rows)} annotated outputs, counts.csv, detections.jsonl and reports to {out}', flush=True)

if __name__=='__main__': main()

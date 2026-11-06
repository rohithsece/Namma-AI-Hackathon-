"""Optional labeled validation. Not needed for mandatory detection/counting."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from crossguard.dataset import CocoRoadDataset
from crossguard.model import load_detector,CLASSES,WEIGHTS

@torch.inference_mode()
def evaluate(model: torch.nn.Module, dataset: CocoRoadDataset, device: str, confidence: float = 0.45) -> dict:
    """Evaluate object detector accuracy and counting precision on a labeled COCO validation dataset.
    
    Computes standard COCO detection evaluation metrics (mAP@0.50:0.95, AP50) using pycocotools,
    as well as class-wise count Mean Absolute Error (MAE) against labeled ground truth.
    
    Args:
        model: PyTorch detection model in evaluation mode.
        dataset: Instantiated CocoRoadDataset containing validation imagery and annotations.
        device: PyTorch device string ('cpu' or 'cuda').
        confidence: Confidence threshold for counting visible class instances. Defaults to 0.45.
        
    Returns:
        dict: Evaluation summary containing:
            - 'mAP_50_95' (float): Mean Average Precision averaged over IoU 0.50 to 0.95.
            - 'AP50' (float): Average Precision at IoU threshold 0.50.
            - 'count_MAE_by_class' (dict): Mean absolute counting error per target category.
            - 'images' (int): Total evaluated image count.
            - 'count_confidence' (float): Threshold used for count error evaluation.
            - 'scope' (str): Technical explanation of evaluation scope and environment.
            - 'note' (str): Ground-truth handling notes.
    """
    model.eval()
    predictions = []
    errors = []
    
    for image, target in dataset:
        # Run eager inference on input tensor
        pred = model([image.to(device)])[0]
        iid = int(target['image_id'])
        
        # Tally ground truth instances per class
        gt = {c: 0 for c in CLASSES}
        detected = {c: 0 for c in CLASSES}
        for label in target['labels'].tolist():
            gt[label] += 1
            
        # Parse detections and record valid predictions
        for box, score, label in zip(pred['boxes'].cpu(), pred['scores'].cpu(), pred['labels'].cpu()):
            cid = int(label)
            if cid not in CLASSES:
                continue
            x1, y1, x2, y2 = box.tolist()
            predictions.append({
                'image_id': iid,
                'category_id': cid,
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # Convert [x1, y1, x2, y2] to COCO [x, y, w, h]
                'score': float(score)
            })
            if score >= confidence:
                detected[cid] += 1
                
        # Compute absolute counting error for each class on this frame
        errors.append([abs(detected[c] - gt[c]) for c in CLASSES])
        
    # Evaluate COCO mAP if predictions were produced
    if predictions:
        result = dataset.coco.loadRes(predictions)
        ev = COCOeval(dataset.coco, result, 'bbox')
        ev.params.imgIds = dataset.ids
        ev.params.catIds = list(CLASSES)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        ap, ap50 = float(ev.stats[0]), float(ev.stats[1])
    else:
        ap = ap50 = 0.0
        
    return {
        'mAP_50_95': ap,
        'AP50': ap50,
        'count_MAE_by_class': dict(zip(CLASSES.values(), np.mean(errors, axis=0).tolist())),
        'images': len(dataset),
        'count_confidence': confidence,
        'scope': 'PyTorch eager torchvision transform/postprocess on labeled validation set; not ExecuTorch deployment parity',
        'note': 'Count MAE excludes crowd annotations; COCO AP uses standard crowd handling. AP=-1 means no evaluable positives.'
    }


def main():
    """CLI entry point for running accuracy evaluation on a custom labeled COCO dataset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=WEIGHTS); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(4)
    result=evaluate(load_detector(a.weights),CocoRoadDataset(a.images,a.annotations),'cpu')
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()

"""CrossGuard ML pipeline utilities for preprocessing, postprocessing, zone analysis, and annotation."""
import cv2
import numpy as np
from .model import CLASSES


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Preprocess a raw BGR image into the normalized tensor format expected by the model.
    
    Steps performed:
    1. Resizes input to fixed 320x320 resolution using bilinear interpolation.
    2. Converts color channel layout from BGR (OpenCV default) to RGB.
    3. Transposes layout from HWC (Height, Width, Channels) to NCHW (Batch, Channels, Height, Width).
    4. Normalizes pixel values from [0, 255] uint8 to [0.0, 1.0] float32.
    
    Args:
        bgr: Input image in OpenCV BGR format with shape (H, W, 3) and uint8 dtype.
        
    Returns:
        np.ndarray: Preprocessed contiguous float32 array with shape (1, 3, 320, 320) in range [0.0, 1.0].
    """
    # Resize to model input dimensions (320x320)
    resized = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_LINEAR)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Convert HWC to NCHW, enforce C-contiguous memory layout, and normalize to [0, 1]
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.55, max_output: int = 100) -> list[int]:
    """Perform Non-Maximum Suppression (NMS) on bounding boxes based on Intersection-over-Union (IoU).
    
    Iteratively selects the candidate with the highest score and eliminates overlapping
    boxes whose IoU exceeds the specified threshold.
    
    Args:
        boxes: Array of bounding boxes in [x1, y1, x2, y2] format with shape (N, 4).
        scores: 1D array of confidence scores corresponding to each box with shape (N,).
        threshold: IoU overlap threshold above which candidate boxes are suppressed. Defaults to 0.55.
        max_output: Maximum number of box indices to retain. Defaults to 100.
        
    Returns:
        list[int]: Indices of the retained bounding boxes, sorted in descending order of score.
    """
    # Sort candidate indices by confidence score in descending order
    order = np.argsort(-scores, kind='stable')
    keep = []
    
    while len(order) and len(keep) < max_output:
        # Pick the highest-scoring candidate
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        
        # Calculate intersection rectangle coordinates against remaining candidates
        lo = np.maximum(boxes[i, :2], boxes[rest, :2])
        hi = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        wh = np.maximum(hi - lo, 0)
        inter = wh[:, 0] * wh[:, 1]
        
        # Calculate box areas and Intersection-over-Union (IoU)
        area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        
        # Keep only candidates with IoU overlap below the threshold
        order = rest[iou <= threshold]
        
    return keep


def postprocess(boxes: np.ndarray, scores: np.ndarray, width: int, height: int,
                confidence: float = 0.45, iou: float = 0.55) -> list[dict]:
    """Postprocess raw detector outputs into filtered, rescaled detections.
    
    Processes the raw anchor boxes and class scores by:
    1. Filtering predictions per supported class using the confidence threshold.
    2. Capping top candidates per class to 300 (torchvision SSD convention).
    3. Applying per-class Non-Maximum Suppression (NMS).
    4. Rescaling bounding boxes from model coordinate space (320x320) back to original image dimensions.
    
    Args:
        boxes: Raw decoded boxes in 320x320 space with shape (3234, 4) in [x1, y1, x2, y2].
        scores: Softmax probabilities for the 6 tracked classes with shape (3234, 6).
        width: Original image width in pixels.
        height: Original image height in pixels.
        confidence: Minimum confidence score to consider a detection. Defaults to 0.45.
        iou: IoU threshold for Non-Maximum Suppression. Defaults to 0.55.
        
    Returns:
        list[dict]: List of detection dictionaries (up to 100), each containing:
            - 'class_id' (int): Native COCO class ID.
            - 'class' (str): Class name (e.g., 'person', 'car', 'bus').
            - 'confidence' (float): Detection confidence score.
            - 'box' (list[float]): Bounding box [x1, y1, x2, y2] in original image coordinates.
    """
    detections = []
    # Iterate through the 6 monitored traffic categories
    for col, (class_id, name) in enumerate(CLASSES.items()):
        # Filter candidates meeting the confidence threshold for this class
        selected = np.flatnonzero(scores[:, col] >= confidence)
        # torchvision SSD uses a per-class pre-NMS cap of 300 candidates
        selected = selected[np.argsort(-scores[selected, col], kind='stable')[:300]]
        
        # Apply NMS on selected candidates
        for k in nms(boxes[selected], scores[selected, col], iou):
            idx = selected[k]
            # Rescale box from 320x320 space to original image resolution
            box = (boxes[idx] * np.array([width / 320, height / 320, width / 320, height / 320])).tolist()
            # Discard degenerate zero/negative area boxes
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append({
                'class_id': class_id,
                'class': name,
                'confidence': float(scores[idx, col]),
                'box': box
            })
            
    # Sort all detections by confidence descending and retain up to 100 objects
    return sorted(detections, key=lambda d: -d['confidence'])[:100]


def point_in_zone(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Test whether a 2D point lies inside or on the boundary of a polygonal zone.
    
    Args:
        point: 2D coordinate pair (x, y) in pixel space.
        polygon: List of polygon vertices [[x1, y1], [x2, y2], ...] in pixel space.
        
    Returns:
        bool: True if point is inside or on the edge of the polygon; False otherwise.
    """
    # cv2.pointPolygonTest returns positive for inside, 0 for on-edge, negative for outside
    return bool(cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, point)), False) >= 0)


def analyse(detections: list[dict], config: dict | None, width: int, height: int) -> tuple[dict, dict]:
    """Calculate per-frame object counts and evaluate spatial zone co-occupancy heuristics.
    
    Computes total visible counts for each of the 6 traffic classes. When a zone configuration
    is provided, projects ground-contact points (bottom-center of bounding boxes) into
    calibrated pedestrian and vehicle zones to flag potential conflict scenarios.
    
    Args:
        detections: List of detection dictionaries produced by postprocess().
        config: Optional dictionary containing normalized polygon coordinates for
            'pedestrian_zone' and 'vehicle_zone'. None if zone analysis is disabled.
        width: Original image width in pixels.
        height: Original image height in pixels.
        
    Returns:
        tuple[dict, dict]:
            - counts: Dictionary mapping class names to instance counts in the current frame.
            - zone_analysis: Dictionary containing zone status, pedestrian/vehicle counts in zones,
              and whether a potential conflict condition was detected.
    """
    # Initialize count tallies for all 6 target classes
    counts = {name: 0 for name in CLASSES.values()}
    for d in detections:
        counts[d['class']] += 1
        
    # Return basic counts if zone configuration is not active
    if config is None:
        return counts, {'enabled': False, 'potential_conflict': None}
        
    # Helper to convert normalized [0, 1] polygon coordinates to pixel coordinates
    def polygon(key):
        return [[p[0] * width, p[1] * height] for p in config[key]]
        
    pzone, vzone = polygon('pedestrian_zone'), polygon('vehicle_zone')
    pedestrians, vehicles = [], []
    
    for d in detections:
        x1, y1, x2, y2 = d['box']
        # Foot heuristic: bottom-center of bounding box approximates road-contact position
        foot = ((x1 + x2) / 2, y2)
        if d['class'] == 'person' and point_in_zone(foot, pzone):
            pedestrians.append(d)
        if d['class'] != 'person' and point_in_zone(foot, vzone):
            vehicles.append(d)
            
    return counts, {
        'enabled': True,
        'pedestrians_in_crossing': len(pedestrians),
        'vehicles_in_approach': len(vehicles),
        'potential_conflict': bool(pedestrians and vehicles),
        'method': 'zone co-occupancy heuristic; no speed, trajectory or collision prediction'
    }


def validate_zones(config: dict) -> dict:
    """Validate zone configuration schema, coordinate bounds, and polygon geometry.
    
    Ensures that required polygon keys exist, coordinates are finite floats in [0, 1],
    polygons have at least 3 vertices, and contour areas are strictly non-zero.
    
    Args:
        config: Dictionary containing 'pedestrian_zone' and 'vehicle_zone' coordinate lists.
        
    Returns:
        dict: The validated configuration dictionary.
        
    Raises:
        ValueError: If required keys are missing, coordinates are invalid, or polygon area is zero.
    """
    for key in ['pedestrian_zone', 'vehicle_zone']:
        if key not in config:
            raise ValueError(f"Missing required zone key: '{key}'")
        points = np.asarray(config[key], dtype=float)
        # Check dimensionality and normalized coordinate range [0, 1]
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any()):
            raise ValueError(f'{key}: need >=3 normalized [x,y] points in [0,1]')
        # Ensure the polygon forms a non-degenerate area
        if abs(cv2.contourArea(points.astype(np.float32))) < 1e-5:
            raise ValueError(f'{key}: polygon has zero area')
    return config


def annotate(frame: np.ndarray, detections: list[dict], counts: dict,
             zone_result: dict, config: dict | None = None) -> np.ndarray:
    """Draw visual overlays, bounding boxes, labels, zones, and an information banner.
    
    Constructs an annotated output image without destroying source pixels by prepending
    a dedicated status header banner displaying frame statistics and conflict alerts.
    
    Args:
        frame: Original input image in OpenCV BGR format.
        detections: List of detection dictionaries from postprocess().
        counts: Dictionary of per-frame class counts.
        zone_result: Analysis dictionary containing conflict status from analyse().
        config: Optional zone configuration dictionary with polygon vertices.
        
    Returns:
        np.ndarray: Annotated BGR image with top information banner and bounding overlays.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    
    # Draw configured safety zones if provided
    if config:
        for key, color in [('pedestrian_zone', (50, 200, 50)), ('vehicle_zone', (50, 50, 240))]:
            pts = np.array([[x * w, y * h] for x, y in config[key]], np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            
    # Draw bounding boxes and confidence labels for detected objects
    for d in detections:
        x1, y1, x2, y2 = map(int, d['box'])
        color = (50, 220, 50) if d['class'] == 'person' else (255, 150, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d['class']} {d['confidence']:.2f}",
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
                    
    # Format per-class counts summary text
    text = '  '.join(f'{k}:{v}' for k, v in counts.items())
    
    # Prepend 56-pixel dark header banner to preserve all original video pixels
    out = cv2.copyMakeBorder(out, 56, 0, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    
    # Display conflict alert or nominal operating status
    status = ('POTENTIAL CONFLICT: zone co-occupancy'
              if zone_result.get('potential_conflict')
              else 'CROSSGUARD AI | per-frame counts')
    cv2.putText(out, status, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 180, 255), 1)
    
    return out

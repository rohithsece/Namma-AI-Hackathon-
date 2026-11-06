"""Model architecture definitions and weight loaders for CrossGuard ML."""
from pathlib import Path
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.image_list import ImageList

# Native COCO dataset class identifiers mapped to monitored road traffic categories
CLASSES = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'models/ssdlite320_mobilenet_v3_large_coco.pth'


def load_detector(weights: Path = WEIGHTS) -> torch.nn.Module:
    """Load the standard torchvision SSDLite320 MobileNetV3 Large detection model from local checkpoint.
    
    Instantiates the model architecture with the standard 91 COCO classes and loads
    the local weight file without initiating any internet downloads.
    
    Args:
        weights: Path to the PyTorch .pth checkpoint file. Defaults to bundled COCO weights.
        
    Returns:
        torch.nn.Module: The detector model initialized in evaluation mode (.eval()).
    """
    # Instantiate architecture with 91 classes; weights=None avoids any implicit internet requests
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    # Load state dict strictly from local file
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    return model.eval()


class EdgeDetector(torch.nn.Module):
    """Deployable fixed-tensor graph wrapper for SSDLite320 MobileNetV3 Large.
    
    Accepts normalized RGB float32 tensors with shape [1, 3, 320, 320] in range [0, 1].
    Encapsulates internal normalization, MobileNetV3 backbone feature extraction, SSD head
    classification/regression, anchor generation, and box decoding.
    
    Important design properties:
    - Pre-computed anchors are cached as buffers for deterministic execution.
    - Full 91-class softmax is computed before filtering down to the 6 target categories,
      preserving the original probability distribution of the pretrained COCO head.
    - Decoded boxes are returned in 320x320 pixel coordinate space.
    """
    
    def __init__(self, model: torch.nn.Module):
        """Initialize the edge detector by extracting components from the eager torchvision model.
        
        Args:
            model: Eager torchvision SSDLite detector returned by load_detector().
        """
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        
        # Pre-compute and freeze anchor grid for a fixed 320x320 input tensor
        with torch.no_grad():
            x = torch.zeros(1, 3, 320, 320)
            features = list(self.backbone(x).values())
            anchors = model.anchor_generator(ImageList(x, [(320, 320)]), features)[0]
            
        # Register cached anchors and selected target class indices as non-parameter buffers
        self.register_buffer('anchors', anchors)
        self.register_buffer('selected', torch.tensor(list(CLASSES), dtype=torch.long))
        self.coder = model.box_coder

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute forward inference pass on preprocessed image tensors.
        
        Args:
            images: Batch of float32 tensors with shape (B, 3, 320, 320) in range [0.0, 1.0].
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - boxes: Decoded bounding boxes with shape (3234, 4) in [x1, y1, x2, y2]
                  within [0.0, 320.0] pixel range.
                - scores: Class probabilities for the 6 tracked categories with shape (3234, 6).
        """
        # torchvision SSDLite standard normalization: (x - 0.5) / 0.5 maps [0, 1] to [-1, 1]
        normalized = (images - 0.5) / 0.5
        features = list(self.backbone(normalized).values())
        output = self.head(features)
        
        # Decode regression offsets into absolute pixel coordinates based on anchor priors
        boxes = self.coder.decode_single(output['bbox_regression'][0], self.anchors)
        boxes = boxes.clamp(0.0, 320.0)
        
        # Compute complete softmax across all 91 classes first, then slice out the 6 target classes
        scores = output['cls_logits'][0].softmax(-1).index_select(1, self.selected)
        return boxes, scores

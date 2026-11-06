"""COCO-format dataset loader and preprocessing for optional supervised fine-tuning."""
from pathlib import Path
import random
import cv2
import torch
from pycocotools.coco import COCO
from .model import CLASSES


class CocoRoadDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for road detection imagery with COCO annotations.
    
    Loads images and bounding box annotations, verifies category integrity against the
    supported native COCO classes, and applies optional geometric data augmentation.
    """
    
    def __init__(self, images: str | Path, annotations: str | Path, augment: bool = False):
        """Initialize the COCO road dataset parser and validate annotations.
        
        Args:
            images: Path to directory containing image files.
            annotations: Path to the COCO JSON annotations file.
            augment: Whether to apply random horizontal flip data augmentation. Defaults to False.
            
        Raises:
            ValueError: If dataset contains no images, category IDs do not match native COCO IDs,
                or annotations contain unsupported class IDs.
        """
        self.root = Path(images).resolve()
        self.coco = COCO(str(annotations))
        self.augment = augment
        self.ids = sorted(self.coco.imgs)
        
        if not self.ids:
            raise ValueError('Dataset contains no images')
            
        # Verify that dataset categories use official native COCO IDs and names
        for c in self.coco.dataset['categories']:
            if c['id'] not in CLASSES or c['name'] != CLASSES[c['id']]:
                raise ValueError(f'Use native COCO IDs and names {CLASSES}; got {c}')
                
        # Validate that all annotations belong to the supported classes
        for ann in self.coco.dataset.get('annotations', []):
            if ann['category_id'] not in CLASSES:
                raise ValueError(f"Unsupported class ID {ann['category_id']}; expected one of {list(CLASSES.keys())}")

    def __len__(self) -> int:
        """Return the total number of images in the dataset.
        
        Returns:
            int: Image count.
        """
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load an image and its corresponding bounding box annotations by index.
        
        Args:
            index: Integer index within the sorted image IDs list.
            
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - image: Float32 RGB tensor with shape (3, H, W) normalized to [0.0, 1.0].
                - target: Dictionary containing:
                    - 'boxes': Float32 tensor of shape (N, 4) in [x1, y1, x2, y2] format.
                    - 'labels': Int64 tensor of shape (N,) with native COCO class IDs.
                    - 'image_id': Scalar int64 tensor identifying the image.
                    
        Raises:
            ValueError: If image path traverses outside dataset root, file is unreadable,
                dimensions mismatch metadata, or bounding boxes are out of bounds.
        """
        iid = self.ids[index]
        meta = self.coco.imgs[iid]
        path = (self.root / meta['file_name']).resolve()
        
        # Security check: ensure path does not escape dataset root directory
        if not path.is_relative_to(self.root):
            raise ValueError(f'Image path leaves dataset root: {path}')
            
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f'Missing image {path}')
            
        h, w = bgr.shape[:2]
        if (w, h) != (meta['width'], meta['height']):
            raise ValueError(f"Annotation/image size mismatch for {path}: ({w}, {h}) vs ({meta['width']}, {meta['height']})")
            
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid))
        boxes = []
        labels = []
        
        for ann in anns:
            # Crowd annotations remain in COCO validation ground truth but are excluded from training
            if ann.get('iscrowd', 0):
                continue
            x, y, bw, bh = ann['bbox']
            # Validate box geometry within image boundaries
            if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > w + 0.01 or y + bh > h + 0.01:
                raise ValueError(f'Invalid bounding box {ann["bbox"]} in image {path}')
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann['category_id'])
            
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # Convert BGR (OpenCV) to RGB, permute HWC -> CHW, and normalize to [0, 1]
        image = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float() / 255.0
        
        # Apply random horizontal flip data augmentation with 50% probability
        if self.augment and random.random() < 0.5:
            image = image.flip(-1)
            if len(boxes):
                # Invert horizontal coordinates: new_x1 = width - old_x2; new_x2 = width - old_x1
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                
        return image, {
            'boxes': boxes,
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor(iid)
        }


def collate(batch: list[tuple]) -> tuple:
    """Collate batch items into separate tuples of images and target dictionaries.
    
    Used by PyTorch DataLoader since detection samples have variable numbers of objects.
    
    Args:
        batch: List of tuples (image, target_dict) yielded by CocoRoadDataset.__getitem__().
        
    Returns:
        tuple: Tuple containing:
            - tuple of image tensors.
            - tuple of target annotation dictionaries.
    """
    return tuple(zip(*batch))

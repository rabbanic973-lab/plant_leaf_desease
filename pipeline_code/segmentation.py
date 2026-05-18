"""
LeafGuard AI — Leaf Instance Segmentation (Reference Code)
Uses YOLOv8s-seg for detecting and segmenting individual leaves
from plant/tree/branch images.
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np


def get_segmentation_model():
    """
    Load YOLOv8s-seg (small variant) for instance segmentation.
    This model produces both bounding boxes and pixel-level masks.
    """
    model = YOLO("yolov8s-seg.pt")
    return model


def segment_leaves(image_path: str, model: YOLO, conf_threshold: float = 0.20):
    """
    Given an image path and a YOLOv8-seg model, detect and segment
    individual leaves/plant regions.

    Returns a list of dictionaries with:
    - 'box': [ymin%, xmin%, ymax%, xmax%] bounding box in percentage
    - 'mask': binary mask array for the leaf (if available)
    - 'confidence': detection confidence score
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    img_np = np.array(image)

    results = model(img_np, conf=conf_threshold, verbose=False)

    leaves = []

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        masks = result.masks

        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                box_area = (x2 - x1) * (y2 - y1)
                image_area = width * height

                # Filter out extremely large or small detections
                if box_area < 0.005 * image_area or box_area > 0.92 * image_area:
                    continue

                leaf = {
                    "box": [
                        (y1 / height) * 100,
                        (x1 / width) * 100,
                        (y2 / height) * 100,
                        (x2 / width) * 100,
                    ],
                    "confidence": conf,
                }

                # Extract pixel mask if available
                if masks is not None and i < len(masks):
                    mask_data = masks[i].data.cpu().numpy().squeeze()
                    leaf["mask"] = mask_data

                leaves.append(leaf)

                if len(leaves) >= 20:
                    break

    return leaves


# Example Usage:
# model = get_segmentation_model()
# leaves = segment_leaves("branch_with_7_leaves.jpg", model)
# print(f"Detected {len(leaves)} individual leaves")
# for i, leaf in enumerate(leaves):
#     print(f"  Leaf {i+1}: box={leaf['box']}, conf={leaf['confidence']:.2f}")

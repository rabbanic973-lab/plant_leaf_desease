import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import torchvision.transforms as T

def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def segment_leaves(image_path, model, device):
    """
    Given an image path and a trained Mask R-CNN model, return the masks,
    bounding boxes, and labels for the detected leaves.
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)

    # prediction contains 'boxes', 'labels', 'scores', 'masks'
    return prediction[0]

# Example Usage:
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# num_classes = 2 # Background + Leaf
# model = get_model_instance_segmentation(num_classes)
# model.load_state_dict(torch.load('leaf_maskrcnn.pth'))
# model.to(device)
# results = segment_leaves('sample_leaf.jpg', model, device)

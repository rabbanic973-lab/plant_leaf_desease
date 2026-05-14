import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

class LeafDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LeafDiseaseClassifier, self).__init__()
        # Using a ResNet18 as the backbone for disease classification
        self.backbone = models.resnet18(pretrained=True)
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

def classify_crop(image_crop, model, device, class_names):
    """
    Takes a single segmented leaf crop, runs it through the CNN,
    and returns the predicted disease class.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    class_idx = preds.item()
    confidence = probabilities[0][class_idx].item()
    
    return class_names[class_idx], confidence

# Example Usage:
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# class_names = ['Healthy', 'Powdery Mildew', 'Rust', 'Blight']
# model = LeafDiseaseClassifier(num_classes=len(class_names)).to(device)
# model.load_state_dict(torch.load('disease_cnn.pth'))
# # Assume `leaf_img` is a PIL Image cropped from the segmentation step
# disease, conf = classify_crop(leaf_img, model, device, class_names)

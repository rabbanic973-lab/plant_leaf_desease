"""
LeafGuard AI — Disease Classification (Reference Code)
Uses a pre-trained PlantVillage MobileNetV2 model from HuggingFace
to classify leaf diseases across 38 plant-disease combinations.
"""

from transformers import pipeline as hf_pipeline
from PIL import Image


# 38 PlantVillage disease classes
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def get_classifier():
    """
    Load the pre-trained PlantVillage disease classifier from HuggingFace.
    Model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
    """
    classifier = hf_pipeline(
        "image-classification",
        model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
        device=-1,  # CPU
    )
    return classifier


def format_disease_name(raw_name: str) -> tuple:
    """
    Convert raw class names like 'Tomato___Early_blight' into
    human-readable (plant, disease, is_healthy) tuples.
    """
    parts = raw_name.split("___")
    plant = parts[0].replace("_", " ").strip()
    disease_raw = parts[1] if len(parts) > 1 else "Unknown"
    is_healthy = disease_raw.lower() == "healthy"
    disease = disease_raw.replace("_", " ").strip()
    return plant, disease, is_healthy


def classify_crop(image_crop: Image.Image, classifier) -> dict:
    """
    Takes a single segmented leaf crop, runs it through the
    PlantVillage classifier, and returns the predicted disease class.
    """
    image_crop = image_crop.convert("RGB").resize((224, 224), Image.LANCZOS)

    predictions = classifier(image_crop, top_k=3)

    if predictions and len(predictions) > 0:
        top = predictions[0]
        plant, disease, is_healthy = format_disease_name(top["label"])

        return {
            "plant": plant,
            "disease": disease if not is_healthy else None,
            "status": "Healthy" if is_healthy else "Diseased",
            "confidence": round(top["score"], 4),
            "full_label": f"{plant} — {disease}",
        }

    return {
        "plant": "Unknown",
        "disease": "Unknown",
        "status": "Diseased",
        "confidence": 0.0,
        "full_label": "Unknown — Unknown",
    }


# Example Usage:
# classifier = get_classifier()
# leaf_img = Image.open("single_leaf.jpg")
# result = classify_crop(leaf_img, classifier)
# print(f"Plant: {result['plant']}")
# print(f"Disease: {result['disease']}")
# print(f"Status: {result['status']}")
# print(f"Confidence: {result['confidence']:.2%}")

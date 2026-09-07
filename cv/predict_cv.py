from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from .train_cv import BreastCNN

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cv_model.pt"

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    architecture = checkpoint.get("architecture", "BreastCNN-v1")

    if architecture == "resnet18_imagenet":
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    else:
        model = BreastCNN(len(class_names))

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint

def build_preprocessing(checkpoint):
    image_size = int(checkpoint.get("image_size", 128))
    architecture = checkpoint.get("architecture", "BreastCNN-v1")

    if architecture == "resnet18_imagenet":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def predict_image(file_or_path):
    model, checkpoint = load_model()
    transform = build_preprocessing(checkpoint)

    image = Image.open(file_or_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    confidence, predicted_index = torch.max(probabilities, dim=0)
    class_names = checkpoint["class_names"]

    return {
        "label": class_names[int(predicted_index.item())],
        "confidence": float(confidence.item()),
        "probabilities": {
            class_names[i]: float(probabilities[i].item())
            for i in range(len(class_names))
        },
    }

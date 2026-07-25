import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Setup Device for Hardware Acceleration
# This automatically routes processing to your local GPU if available, or falls back to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Inference running on: {device}")

# 2. Rebuild the Model Architecture
# We must build the exact same structure we trained in Colab before loading the weights
def load_model(model_path):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Normal, Pneumonia
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # Set to evaluation mode
    return model

# 3. Define the Image Preprocessing Pipeline
# These are the exact same transforms used during validation in Colab
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Create the Prediction Function
def predict_image(image_path, model):
    classes = ['Normal', 'Pneumonia']
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return classes[predicted_idx.item()], confidence.item() * 100

# Quick test block (runs only if you execute this specific file)
if __name__ == "__main__":
    # We will test this once you download best_model.pth and have a sample image
    print("Inference script is ready!")
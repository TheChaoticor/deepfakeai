import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torchvision.transforms as transforms

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the DeepFakeDetector model (same as in training)
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=None)  # Initialize without pretrained weights
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, 1)  # Binary classification

    def forward(self, x):
        return self.mobilenet(x)

# Load the model
def load_model():
    model = DeepFakeDetector()
    checkpoint = torch.load("./weights/best_model (1).pth", map_location=torch.device('cpu'))
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(new_state_dict, strict=True)  # strict=True ensures all keys match exactly
    
    model.eval()
    return model

model = load_model()

# Debugging: Check expected and loaded keys
checkpoint = torch.load("./weights/best_model (1).pth", map_location=torch.device('cpu'))
print("Checkpoint Keys:", checkpoint.keys())  # Check saved keys
print("Model State Dict Keys:", model.state_dict().keys())  # Check expected keys

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Preprocess
    image_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(image_tensor).squeeze()
        probability = torch.sigmoid(output).item()
    print(f"Probability: {probability}, isDeepfake: {probability < 0.5}")
    return {
        "isDeepfake": probability < 0.5,
        "confidence": float(probability * 100)
    }
    

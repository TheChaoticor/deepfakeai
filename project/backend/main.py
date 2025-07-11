import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torchvision.transforms as transforms

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)


import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.resnet = models.resnet50(pretrained=False)  
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1) 
    def forward(self, x):
        return self.resnet(x)



def load_model():
    model = DeepFakeDetector()
    checkpoint = torch.load("./weights/mesonet_best (1).pth", map_location=torch.device('cpu'))
    
   
    new_state_dict = {k.replace("resnet.", "base_model."): v for k, v in checkpoint.items()}

   
    model.load_state_dict(new_state_dict, strict=False)  
    
    model.eval()
    return model

model = load_model()


checkpoint = torch.load("./weights/mesonet_best (1).pth", map_location=torch.device('cpu'))
print("Checkpoint Keys:", checkpoint.keys()) 
print("Model State Dict Keys:", model.state_dict().keys())  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
   
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

   
    image_tensor = transform(image).unsqueeze(0)


    with torch.no_grad():
        output = model(image_tensor).squeeze()
        probability = torch.sigmoid(output).item()
    print(f"Probability: {probability}, isDeepfake: {probability > 0.5}")
    return {
        "isDeepfake": probability > 0.5,
        "confidence": float(probability * 100)
    }
    

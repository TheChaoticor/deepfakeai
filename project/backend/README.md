# DeepFake Detection Backend

This is the backend service for the DeepFake detection application. It uses PyTorch for the deep learning model and FastAPI for the REST API.

## Project Structure

```
backend/
├── main.py           # FastAPI application and model inference
├── train.py         # Training script
├── requirements.txt # Python dependencies
└── weights/         # Directory for model weights
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

1. Prepare your dataset in the following structure:
```
data/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

2. Run training:
```bash
python train.py
```

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- POST `/analyze`: Analyze an image for deepfake detection
  - Input: Form data with file
  - Output: JSON with `isDeepfake` and `confidence` fields

## Model Architecture

The model uses a custom CNN architecture with the following features:
- Three convolutional layers with batch normalization
- Max pooling layers
- Dropout for regularization
- Two fully connected layers
- Sigmoid activation for binary classification
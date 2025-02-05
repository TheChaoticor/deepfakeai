import torch

checkpoint = torch.load("ffpp_c40.pth", map_location="cpu")
print("Checkpoint Keys:", checkpoint.keys())

# If the model's state_dict is inside a "state_dict" key:
if "state_dict" in checkpoint:
    print("State Dict Keys:", list(checkpoint["state_dict"].keys())[:10])  # First 10 keys
else:
    print("State Dict Keys:", list(checkpoint.keys())[:10])  # First 10 keys

import torch

checkpoint = torch.load("ffpp_c40.pth", map_location="cpu")
print("Checkpoint Keys:", checkpoint.keys())

if "state_dict" in checkpoint:
    print("State Dict Keys:", list(checkpoint["state_dict"].keys())[:10])  
else:
    print("State Dict Keys:", list(checkpoint.keys())[:10]) 

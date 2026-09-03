import torch

checkpoints = [
    "cavity_checkpoint.pt",
    "cylinder_checkpoint.pt",
    "naca_checkpoint.pt"
]

for filename in checkpoints:
    data = torch.load(filename, map_location="cpu", weights_only=False)
    print(f"Loaded {filename}: {data['pod_r']} modes, {data['pod_N']} points")

import os


model_path = "baseline_model.pth"

size = os.path.getsize(model_path) / (1024 * 1024)  # MB

print(f"Baseline Model Size: {size:.2f} MB")
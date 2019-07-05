import torch

print("Device:", torch.cuda.current_device())
print("Device number:", torch.cuda.device(0))
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
print("Available:", torch.cuda.is_available())
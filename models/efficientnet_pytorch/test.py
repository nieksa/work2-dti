from model import EfficientNet
import torch

inputs = torch.rand(1, 3, 224, 224)
model = EfficientNet.from_pretrained('efficientnet-b0')
model.eval()
outputs = model(inputs)
print(outputs.shape)
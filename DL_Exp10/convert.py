import torch
from train import CNN

model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model created!")
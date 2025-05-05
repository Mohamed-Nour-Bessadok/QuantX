import torch

def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    model = torch.quantization.convert(model, inplace=True)
    return model

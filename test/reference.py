import torch

def fullyconnected_reference(a, b):
    v = a[:,:,None]*b
    valmax, argmax = torch.max(v, axis=-2)
    valmin, argmin = torch.min(v, axis=-2)
    return valmax + valmin, argmax, argmin
import torch

def fullyconnected_reference(a, b, bias=0.0, beta=0):
    v = a[:,:,None]*b
    valmax, argmax = torch.max(v, axis=-2)
    valmin, argmin = torch.min(v, axis=-2)

    y = a@b*beta + (valmax+valmin)*(1.0-beta) + bias

    return y, argmax, argmin
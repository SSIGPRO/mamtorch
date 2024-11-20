import torch

def fullyconnected_reference(a, b, bias=0.0, beta=0):
    v = a[:,:,None]*b
    valmax, argmax = torch.max(v, axis=-2)
    valmin, argmin = torch.min(v, axis=-2)

    y = a@b*beta + (valmax+valmin)*(1.0-beta) + bias

    return y, argmax, argmin

def fullyconnected_backward_reference(a, b, cgrad, argmax, argmin, beta=0.0):
    # MAC component
    agrad1 = cgrad@b.T
    bgrad1 = a.T@cgrad
    # MAM component
    agrad2 = torch.zeros_like(a)
    bgrad2 = torch.zeros_like(b)
    for i in range(cgrad.shape[0]):
        mask = torch.zeros_like(b)
        mask[argmax[i], torch.arange(len(argmax[i]))] += 1
        mask[argmin[i], torch.arange(len(argmin[i]))] += 1
        agrad2[i] = cgrad[i]@(b*mask).T
        bgrad2 += (torch.unsqueeze(a[i],1)@torch.unsqueeze(cgrad[i],0))*mask
    return agrad1*beta + agrad2*(1-beta), bgrad1*beta + bgrad2*(1-beta)


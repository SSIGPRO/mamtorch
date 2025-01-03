import torch

def fullyconnected_reference(a, b, accblock_size=1):
    rows_to_pad = (accblock_size - (b.size(0) % accblock_size)) % accblock_size
    a_padded = torch.nn.functional.pad(a, (0, rows_to_pad, 0, 0))
    b_padded = torch.nn.functional.pad(b, (0, 0, 0, rows_to_pad))

    N = b_padded.shape[0]//accblock_size
    a_batched = a_padded.T.view(N, -1, a_padded.size(0)).transpose(1, 2)
    b_batched = b_padded.view(N, -1, b_padded.size(1))

    # Perform batched matrix multiplication: (N, A_rows, columns) @ (N, columns, rows_per_submatrix)
    v = torch.bmm(a_batched, b_batched)

    valmax, argmax = torch.max(v, axis=-3)
    valmin, argmin = torch.min(v, axis=-3)

    return valmax + valmin, argmax, argmin

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


import torch
import torch.nn.functional as F

def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

def cross_entropy(out1, out2, reduce=True):
    ent = -torch.sum(out1.log() * out2, dim=1)
    if reduce:
        return torch.mean(ent)
    else:
        return ent

def entropy_margin(p1, p2, value, margin=0.5):
    p1 = F.softmax(p1)
    p2 = F.softmax(p2)
    p = (p1 + p2)/2
    return -torch.mean(torch.clamp(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), min=margin))

def cross_entropy_margin(p1, p2, value, margin=0.5):
    p1 = torch.softmax(p1, dim=1)
    p2 = torch.softmax(p2, dim=1)

    crs = (cross_entropy(p1, p2, reduce=False) + cross_entropy(p2, p1, reduce=False))

    mask_known = crs < value - 1

    return -torch.mean(torch.clamp(torch.abs(crs - value), min=margin, max=2)[crs<2*value]), mask_known




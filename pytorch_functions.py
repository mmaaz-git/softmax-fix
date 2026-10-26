"""PyTorch baselines, Barron et al. (arXiv:2610.31642v1), and our log1p fix."""

import torch
from torch.nn import functional as F

# standard (PyTorch)

def standard_log_softmax(x):
    return F.log_softmax(x, dim=-1)

def standard_softmax(x):
    return F.softmax(x, dim=-1)

def standard_cross_entropy(z, t):
    loss = F.cross_entropy(z.reshape(-1, z.shape[-1]), t.reshape(-1), reduction="none")
    return loss.reshape(t.shape)

def standard_soft_target_cross_entropy(student, teacher):
    p = standard_softmax(teacher.detach())
    loss = F.cross_entropy(student.reshape(-1, student.shape[-1]), p.reshape(-1, p.shape[-1]), reduction="none")
    return loss.reshape(student.shape[:-1])


# barron et al.

def barron_log_softmax(x):
    i_max = x.argmax(dim=-1, keepdim=True)
    x_max = x.gather(-1, i_max)
    is_max = torch.zeros_like(x, dtype=torch.bool).scatter(-1, i_max, True)
    shifted = torch.where(is_max, 0.0, x - x_max)
    others = shifted.masked_fill(is_max, -torch.inf)
    m = -torch.logsumexp(others, dim=-1, keepdim=True)
    return shifted - torch.nn.functional.softplus(-m)

def barron_softmax(x):
    return barron_log_softmax(x).exp()

def barron_cross_entropy(z, t):
    z_t = z.gather(-1, t[..., None])
    is_t = torch.zeros_like(z, dtype=torch.bool).scatter(-1, t[..., None], True)
    others = (z - z_t).masked_fill(is_t, -torch.inf)
    m = -torch.logsumexp(others, dim=-1)
    return torch.nn.functional.softplus(-m)

def barron_soft_target_cross_entropy(student, teacher):
    # Soft-target cancellation fix from Section 3 of arXiv:2610.31642v1.
    log_q = barron_log_softmax(student)
    q = log_q.exp()
    p = barron_softmax(teacher.detach())
    delta = student - teacher.detach()
    d = delta - (p * delta).sum(dim=-1, keepdim=True)
    s = d - torch.log1p((p * torch.expm1(d)).sum(dim=-1, keepdim=True))
    u = p * torch.expm1(s)
    g = u - q * u.sum(dim=-1, keepdim=True)

    # Preserve the forward loss and supply the paper's first-order gradient.
    loss = -(p * log_q.detach()).sum(dim=-1)
    return loss + (g.detach() * (student - student.detach())).sum(dim=-1)


# new

def new_log_softmax(x):
    i_max = x.argmax(dim=-1, keepdim=True)
    x_max = x.gather(-1, i_max)
    is_max = torch.zeros_like(x, dtype=torch.bool).scatter(-1, i_max, True)
    shifted = torch.where(is_max, 0.0, x - x_max)
    others = shifted.masked_fill(is_max, -torch.inf)
    normalizer = torch.log1p(others.exp().sum(dim=-1, keepdim=True))
    return shifted - normalizer

def new_softmax(x):
    return new_log_softmax(x).exp()

def new_cross_entropy(z, t):
    return -new_log_softmax(z).gather(-1, t[..., None]).squeeze(-1)

def new_soft_target_cross_entropy(student, teacher):
    # Same cancellation fix, using the new log-softmax.
    log_q = new_log_softmax(student)
    q = log_q.exp()
    p = new_softmax(teacher.detach())
    delta = student - teacher.detach()
    d = delta - (p * delta).sum(dim=-1, keepdim=True)
    s = d - torch.log1p((p * torch.expm1(d)).sum(dim=-1, keepdim=True))
    u = p * torch.expm1(s)
    g = u - q * u.sum(dim=-1, keepdim=True)

    loss = -(p * log_q.detach()).sum(dim=-1)
    return loss + (g.detach() * (student - student.detach())).sum(dim=-1)

import torch
import fisher.between_bingham_fisher as bbf


def bingham_CE(VB1, LamB1, VB2, LamB2):
    """
    h(f1, f2)
    f1 is gt, f2 is prediction
    @param VB1, VB2: (b, 4, 4)
    @param LamB1, LamB2: (b, 4)
    """
    LamB1 = bbf.ensure_bingham_convention(LamB1)
    LamB2 = bbf.ensure_bingham_convention(LamB2)
    muF = VB1[:, :, LamB1.argmax()]       # (b, 3)


    VB1 = VB1[..., 1:]
    VB2 = VB2[..., 1:]
    LamB1 = LamB1[..., 1:]
    LamB2 = LamB2[..., 1:]

    first_term = torch.log(bingham_F(LamB2))
    second_term = 0
    A = VB1.transpose(1, 2) @ VB2       # (b, 3, 3)
    b = (muF[:, None, :] @ VB2).squeeze(1)      # (b, 3)
    for i in range(3):
        tmp1 = (A[:, i] ** 2 - b[:, i][:, None] ** 2) * (1 / bingham_F(LamB1)[:, None]) * bingham_dF(LamB1)   # (b, 3)
        tmp1 = tmp1.sum(1)      # (b, )
        second_term += LamB2[:, i] * (b[:, i] ** 2 + tmp1)
    cross_entropy = first_term - second_term        # (b, )
    assert not torch.isnan(cross_entropy).any() and not torch.isinf(cross_entropy).any()
    return cross_entropy


def bingham_entropy(LamB):
    """
    @param LamB: (b, 4)
    @return entropy: (b, )
    """
    LamB = bbf.ensure_bingham_convention(LamB)
    first_term = torch.log(bingham_F(LamB))
    second_term = - (LamB * bingham_dF(LamB) / bingham_F(LamB)[:, None]).sum(1)
    entropy = first_term + second_term
    return entropy


def bingham_F(LamB):
    """
    @param LamB: (b, 3) or (b, 4)
    @return F: (b, )
    """
    LamB = bbf.ensure_bingham_convention(LamB)
    c = LamB.sum(1) / 4
    S = bbf.LamB_to_S(LamB)
    F = bbf.constant_bingham_appr_fromS(S, c)
    return F


def bingham_dF(LamB):
    """
    @param LamB: (b, 3) or (b, 4)
    @return dF: (b, 3)
    """
    nelement = LamB.shape[1]
    LamB = bbf.ensure_bingham_convention(LamB)
    with torch.enable_grad():
        LamB.requires_grad_(True)
        F = bingham_F(LamB)
        dF = torch.autograd.grad(F, LamB, torch.ones_like(F))[0]
    if nelement == 3:
        return dF[:, 1:]
    else:
        return dF

def watson_A(kappa):

    lamB = torch.zeros(kappa.shape[0], 4).to(kappa)
    lamB[:, 0] = kappa.view(-1)
    lamB = lamB - kappa.view(-1, 1)
    F = bingham_F(lamB)
    dF = bingham_dF(lamB).detach()
    Sigma = dF / F.unsqueeze(-1)
    return Sigma[:, 0] + 1 - Sigma.sum(dim=-1)

def watson_KL_coef(kappa):

    # for d = 4 only
    A = watson_A(kappa)
    return kappa.view(-1) * (A - (1-A) / 3)

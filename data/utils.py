from openfold.utils import rigid_utils as ru
import torch

# Global map from chain characters to integers.
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


def to_numpy(x):
    return x.detach().cpu().numpy()

def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return ru.Rigid(rots=rots, trans=trans)


def compute_distance_matrix(cell, cart_coords, num_cells=1):
    pos = torch.arange(-num_cells, num_cells + 1, 1).to(cell.device)
    combos = (
        torch.stack(torch.meshgrid(pos, pos, pos, indexing="xy"))
        .permute(3, 2, 1, 0)
        .reshape(-1, 3)
        .to(cell.device)
    )
    shifts = torch.sum(cell.unsqueeze(0) * combos.unsqueeze(-1), dim=1)
    shifted = cart_coords.unsqueeze(1) + shifts.unsqueeze(0)
    dist = cart_coords.unsqueeze(1).unsqueeze(1) - shifted.unsqueeze(0)
    # +eps to avoid nan in differentiation
    dist = (dist.pow(2).sum(dim=-1) + 1e-32).sqrt()
    distance_matrix = dist.min(dim=-1)[0]
    return distance_matrix


def frac2cart(fcoord, cell):
    return fcoord @ cell

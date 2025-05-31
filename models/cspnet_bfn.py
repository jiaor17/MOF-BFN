import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from crysbfn.common.data_utils import lattice_params_to_matrix_torch, get_pbc_distances, radius_graph_pbc, frac_to_cart_coords, back2interval
from crysbfn.pl_modules.base_model import build_mlp

from models.bb_embedder_pyg import GaussianSmearing

MAX_ATOMIC_NUM=100

class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3, period = 1.):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = (2 * np.pi / period) * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        x = x.float()
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class CSPLayer(nn.Module):
    """ Message passing layer for cspnet."""
    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
        ip=True,
        edge_attr_dim=0,
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6 + self.dis_dim + edge_attr_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None, edge_attr = None):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            assert False
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        edges_input = torch.cat([hi, hj, lattices[edge2graph], frac_diff], dim=1)
        if edge_attr is not None:
            edges_input = torch.cat([edges_input, edge_attr], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(edge_features, edge_index[0], dim = 0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim = 1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None, edge_attr = None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff, edge_attr)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):

    def __init__(
        self,
        hidden_dim = 128,
        time_dim = 256,
        num_layers = 4,
        bb_emb_dim = 64,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 10,
        edge_style = 'fc',
        cutoff = 6.0,
        max_neighbors = 20,
        period = 1,
        ln = False,
        ip = True,
        pred_type = False,
        dtime_loss = True,
        cond_acc = False,
        so3_embedding_type = 'none',
    ):
        super(CSPNet, self).__init__()

        self.node_embedding = nn.Linear(bb_emb_dim, hidden_dim)
        self.dtime_loss = dtime_loss
        
        if not dtime_loss:
            time_dim = 1
        self.cond_acc = cond_acc


        self.so3_embedding_type = so3_embedding_type
        input_latent_dim = hidden_dim + 16
        acc_num = 6


        self.atom_latent_emb = nn.Linear(input_latent_dim + time_dim + acc_num * cond_acc, hidden_dim)
        self.period = eval(period)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs, period=self.period)
        elif dis_emb == 'none':
            self.dis_emb = None
        else:
            raise NotImplementedError
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip)
            )            
        self.num_layers = num_layers
        
        self.coord_out = nn.Linear(hidden_dim, 3, bias = False)
        self.lattice_out = nn.Linear(hidden_dim, 6, bias = False)
        self.so3_out = nn.Linear(hidden_dim, 4, bias = False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, bb_emb_dim)

    def back2interval(self, coords):
        if self.period == 1:
            return coords % 1.
        elif self.period == 2 * np.pi:
            return coords - 2 * np.pi * torch.round(coords / (2 * np.pi))
        elif self.period == 2:
            return coords - 2 * torch.round(coords / 2)
        else:
            assert NotImplementedError


    def gen_edges(self, num_atoms, frac_coords):
        if self.edge_style == 'fc':
            lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            if self.period == 1:
                return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.
            elif self.period == 2 * np.pi:
                diff = frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]
                diff = diff - 2 * np.pi * torch.round(diff / (2 * np.pi))
                return fc_edges, diff
            elif self.period == 2:
                diff = frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]
                diff = diff - 2 * torch.round(diff / 2)
                return fc_edges, diff
            else:
                assert NotImplementedError
        elif self.edge_style == 'knn':
            assert NotImplementedError
            
    
    def forward(self, t, bb_embs, frac_coords, so3_vecs, lattices, num_atoms, node2graph, log_acc=None, log_acc_so3=None, do_back2inter=True):
        # lattices = lattices.view(-1, 3, 3)
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(bb_embs)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        
        so3_features = so3_vecs.view(-1, 16)


        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, so3_features, t_per_atom], dim=1)
        # print(node_features.dtype) 

        if self.cond_acc:
            assert log_acc is not None, "log_acc is None"
            assert log_acc_so3 is not None, "log_acc_sphere is None"
            log_acc_so3 = log_acc_so3[:, 1:]
            node_features = torch.cat([node_features, log_acc, log_acc_so3], dim=1)

        

        # print(node_features.dtype)
        node_features = self.atom_latent_emb(node_features)


        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](node_features, 
                                                              frac_coords, lattices, edges, edge2graph, frac_diff = frac_diff)
        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        so3_out = self.so3_out(node_features)

        graph_features = scatter(node_features, node2graph, dim = 0, reduce = 'mean')
        lattice_out = self.lattice_out(graph_features)


        so3_out = so3_out / (so3_out.norm(dim=-1, keepdim=True))

        if self.pred_type:
            type_out = self.type_out(node_features)
            if do_back2inter:
                return lattice_out, self.back2interval(coord_out+frac_coords), so3_out, type_out
            else:
                return lattice_out, coord_out, so3_out, type_out

        if do_back2inter:
            return lattice_out, self.back2interval(coord_out+frac_coords), so3_out
        else:
            return lattice_out, coord_out, so3_out

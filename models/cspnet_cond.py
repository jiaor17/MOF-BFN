import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_scatter import scatter
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import hydra

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
    
class Condition(nn.Module):
    
    def __init__(self, input_dim = 1, process_func = 'identity', eps = 1e-7):

        super().__init__()
        self.process_func = process_func
        self.input_dim = input_dim
        self.eps = eps

    def forward(self, x):
        
        input_cond = x.to(torch.float32)

        if self.process_func == 'identity':
            input_cond = input_cond
        elif self.process_func == 'log':
            input_cond = torch.log(input_cond + self.eps)
        elif self.process_func == 'reverse':
            input_cond = 1. / (input_cond + self.eps)
        elif self.process_func == 'exp':
            input_cond = torch.exp(input_cond)
        else:
            input_cond = input_cond
        
        input_cond = input_cond.view(-1, self.input_dim)

        return input_cond

        

    



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

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None, edge_attr = None, condition_input = None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        if condition_input is not None:
            scale, shift = condition_input
            node_features = node_features * (1 + scale) + shift
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
        condition_conf = None,
        training_condition_ratio = 0.5
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

        # CFG from mattergen
        self.condition_conf = condition_conf
        self.conditions = nn.ModuleDict()
        for cond in self.condition_conf:
            self.conditions[cond.key] = hydra.utils.instantiate(cond.condition_embedding, _recursive_=False)

        self.cond_adapt_layers = nn.ModuleDict()
        self.cond_mixin_layers = nn.ModuleDict()

        for key in self.conditions.keys():
            adapt_layers = []
            mixin_layers = []

            for _ in range(self.num_layers):
                adapt_layers.append(
                    nn.Sequential(
                        nn.Linear(self.conditions[key].input_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                mixin_layers.append(nn.Linear(hidden_dim, hidden_dim * 2, bias=False))
                nn.init.zeros_(mixin_layers[-1].weight)

            self.cond_adapt_layers[key] = nn.ModuleList(adapt_layers)
            self.cond_mixin_layers[key] = nn.ModuleList(mixin_layers)

        self.training_condition_ratio = training_condition_ratio

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
    
    def forward(self, t, bb_embs, frac_coords, so3_vecs, lattices, num_atoms, node2graph, input_conditions, log_acc=None, log_acc_so3=None, do_back2inter=True, use_condition='random'):

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(bb_embs)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        
        so3_features = so3_vecs.view(-1, 16)


        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, so3_features, t_per_atom], dim=1)

        if self.cond_acc:
            assert log_acc is not None, "log_acc is None"
            assert log_acc_so3 is not None, "log_acc_sphere is None"
            if self.so3_embedding_type == 'aa' or self.so3_embedding_type == 'aa_vec':
                log_acc_ax, log_acc_ag = log_acc_so3
                log_acc_so3 = torch.cat([log_acc_ax.view(-1,1), log_acc_ag.view(-1,1)], dim = 1)
            elif self.so3_embedding_type in ['quat_bhm', 'quat_bhm_out', 'quat_bhm_main', 'quat_bhm_main_out', 'quat_bhm_A', 'quat1_bhm']:
                log_acc_so3 = log_acc_so3[:, 1:]
            else:
                log_acc_so3 = log_acc_so3.view(-1,1)
            node_features = torch.cat([node_features, log_acc, log_acc_so3], dim=1)

        # condition & mask
        batch_size = len(num_atoms)
        device = num_atoms.device
        conds, conds_mask = {}, {}
        for key in self.conditions.keys():
            conds[key] = self.conditions[key](input_conditions[key])
            if use_condition == 'random': # training
                mask = torch.rand(batch_size) <= self.training_condition_ratio
            elif use_condition == True:
                mask = torch.ones(batch_size)
            else:
                mask = torch.zeros(batch_size)
            conds_mask[key] = mask.to(device, torch.float32).unsqueeze(-1)


        node_features = self.atom_latent_emb(node_features)


            # self.cond_adapt_layers[key] = nn.ModuleList(adapt_layers)
            # self.cond_mixin_layers[key] = nn.ModuleList(mixin_layers)

        for i in range(0, self.num_layers):

            cond_tot = 0

            # condition injection
            for key in self.conditions.keys():
                cond_i = self.cond_adapt_layers[key][i](conds[key])
                cond_i = self.cond_mixin_layers[key][i](cond_i)
                cond_tot = cond_tot + conds_mask[key] * cond_i

            cond_tot = cond_tot.repeat_interleave(num_atoms, dim=0).chunk(2, dim=-1)

            node_features = self._modules["csp_layer_%d" % i](node_features, 
                                                              frac_coords, lattices, edges, edge2graph, frac_diff = frac_diff, condition_input = cond_tot)



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
        
    def antipodal(self, a, b):

        dot = (a * b).sum(dim=-1, keepdim=True)
        flip_mask = (dot < 0).float()
        b_aligned = b * (1 - 2 * flip_mask)
        return b_aligned
        
    def slerp(self, a, b, w):

        b = self.antipodal(a, b)
        dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        w = torch.full_like(theta, fill_value=w)

        sin_theta = torch.sin(theta)
        small_angle = sin_theta < 1e-6

        coef_a = torch.sin((1 - w) * theta) / (sin_theta + 1e-6)
        coef_b = torch.sin(w * theta) / (sin_theta + 1e-6)

        out = coef_a.unsqueeze(-1) * a + coef_b.unsqueeze(-1) * b
        out[small_angle] = F.normalize((1 - w[small_angle]).unsqueeze(-1) * a[small_angle] + w[small_angle].unsqueeze(-1) * b[small_angle], dim=-1)

        return F.normalize(out, dim=-1)





    def interpolant(self, t, bb_embs, frac_coords, so3_vecs, lattices, num_atoms, node2graph, input_conditions, log_acc=None, log_acc_so3=None, do_back2inter=True, weight=1.0):


        cond_lattice_out, cond_coord_out, cond_so3_out, cond_type_out = self.forward(t, bb_embs, frac_coords, so3_vecs, lattices, num_atoms, node2graph, input_conditions, log_acc, log_acc_so3, do_back2inter, use_condition=True)
        uncond_lattice_out, uncond_coord_out, uncond_so3_out, uncond_type_out = self.forward(t, bb_embs, frac_coords, so3_vecs, lattices, num_atoms, node2graph, input_conditions, log_acc, log_acc_so3, do_back2inter, use_condition=False)

        # lattice
        inter_lattice_out = uncond_lattice_out + weight * (cond_lattice_out - uncond_lattice_out)

        # type
        inter_type_out = uncond_type_out + weight * (cond_type_out - uncond_type_out)

        # coord

        delta_coord = self.back2interval(cond_coord_out - uncond_coord_out + self.period / 2) - self.period / 2

        inter_coord_out = self.back2interval(uncond_coord_out + weight * delta_coord)

        # SO3

        inter_so3_out = self.slerp(uncond_so3_out, cond_so3_out, weight)

        return inter_lattice_out, inter_coord_out, inter_so3_out, inter_type_out
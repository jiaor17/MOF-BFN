import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import i0e, i1e
from torch_scatter import scatter_mean, scatter_sum
from tqdm import tqdm
from torch.distributions import Normal
from .egnn.egnn_new import EGNN
import numpy as np
import hydra
from absl import logging
from crysbfn.common.data_utils import (
    EPSILON, _make_global_adjacency_matrix, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc, remove_mean)
from crysbfn.common.von_mises_fisher_utils import VonMisesFisher, ive
from crysbfn.common.bingham_utils import Bingham
import torch.distributions as D
import math

from fisher.bingham_utils import watson_KL_coef

def corrupt_t_pred(self, mu, t, gamma):
    # if t < self.t_min:
    #   return torch.zeros_like(mu)
    # else:
    # eps_pred = self.model()
    t = torch.clamp(t, min=self.t_min)
    # t = torch.ones((mu.size(0),1)).cuda() * t
    eps_pred = self.model(mu, t)
    x_pred = mu / gamma - torch.sqrt((1 - gamma) / gamma) * eps_pred
    return x_pred


class bfnBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(bfnBase, self).__init__(*args, **kwargs)

    # def zero_center_of_mass(self, x_pos, segment_ids):
    #     size = x_pos.size()
    #     assert len(size) == 2  # TODO check this
    #     seg_means = scatter_mean(x_pos, segment_ids, dim=0)
    #     mean_for_each_segment = seg_means.index_select(0, segment_ids)
    #     x = x_pos - mean_for_each_segment
    #     return x
    
    def zero_center_of_mass(self, x_pos, segment_ids):
        size = x_pos.size()
        assert len(size) == 2  # TODO check this
        seg_means = scatter_mean(x_pos, segment_ids, dim=0)
        mean_for_each_segment = seg_means.index_select(0, segment_ids)
        x = x_pos - mean_for_each_segment
        return x

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_r

    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        f_ = 0.5 * (1 + torch.erf((x - mu) / ((sigma) * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)
        return f_

    def continuous_var_bayesian_flow(self, t, sigma1, x, ret_eps=False, n_samples=1):
        """
        x: [N, D]
        """
        if n_samples == 1:
            gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
            eps = torch.randn_like(x)  # [B, D]
            mu = gamma * x + eps * torch.sqrt(gamma * (1 - gamma))
        else:
            t = t.unsqueeze(-1).repeat(1,1,n_samples)
            x = x.unsqueeze(-1).repeat(1,1,n_samples)
            gamma = 1 - torch.pow(sigma1, 2 * t)
            eps = torch.randn_like(x)
            mu = gamma * x + eps * torch.sqrt(gamma * (1 - gamma))
            mu = mu.mean(-1)
        if not ret_eps:
            return mu, gamma
        else:
            return mu, gamma, eps

    def discrete_var_bayesian_flow(self, t, beta1, x, K, ret_eps=False):
        """
        x: [N, K]
        """
        beta = beta1 * (t**2)  # (B,)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        if not ret_eps:
            return theta
        else:
            return theta, eps

    def ctime4continuous_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    # def ctime4continuous_loss(self, t, sigma1, x_pred, x, pbc_dist=None):
    #     if pbc_dist == None:
    #         loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
    #     else:
    #         loss = pbc_dist.view(x.shape[0], -1).abs().pow(2).sum(dim=1)
    #     return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))



    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, segment_ids=None, mult_constant=True, wn=False):
        # TODO not debuged yet
        if wn:
            steps = torch.arange(1,N+1).to(self.device)
            all_weights = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * steps / N))
            weight_norm = all_weights.mean().detach().clone()
        else:
            weight_norm = 1.
        if segment_ids is not None:
            weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
            loss = scatter_mean(weight.view(-1)*((x_pred - x)**2).mean(-1),segment_ids,dim=0)
        else:
            if mult_constant:
                loss =  N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * (x_pred - x).view(x.shape[0], -1).abs().pow(2)
            else:
                loss =  (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * (x_pred - x).view(x.shape[0], -1).abs().pow(2)
        # print(loss.shape)
        return loss.mean() / weight_norm
    
    def dtime4continuous_loss_cir(self, i, N, sigma1, x_pred, x, segment_ids=None, mult_constant=True, wn=False):
        freqs = torch.arange(-10,11).to(self.device).unsqueeze(0).unsqueeze(0)*np.pi*2
        tar_coord = x.unsqueeze(-1) + freqs
        coord_diff = (tar_coord - x_pred.unsqueeze(-1)).square().min(dim=-1).values
        if wn:
            steps = torch.arange(1,N+1).to(self.device)
            all_weights = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * steps / N))
            weight_norm = all_weights.mean().detach().clone()
        else:
            weight_norm = 1.
        if segment_ids is not None:
            weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
            loss = scatter_mean(weight.view(-1)*((x_pred - x)**2).mean(-1),segment_ids,dim=0)
        else:
            if mult_constant:
                loss =  N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * coord_diff.view(x.shape[0], -1).abs()
            else:
                loss =  (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * coord_diff.view(x.shape[0], -1).abs()
        # print(loss.shape)
        return loss.mean() / weight_norm

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K):
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity.mean()

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    # def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None, mult_constant=True):
    #     alpha = beta1 * (2 * i - 1) / (N**2)  # [N]
    #     e_x = one_hot_x  # [N, T, K]
    #     e_hat = p_0  # (N, T, K)
    #     if mult_constant:
    #         L_n = N * (K * alpha * (((e_x - e_hat) ** 2).sum(dim=-1)))
    #     else:
    #         L_n = (K * alpha * (((e_x - e_hat) ** 2).sum(dim=-1)))
    #     # L_n = N * (K * alpha * (((e_x - e_hat) ** 2).sum(dim=2).sum(dim=1))).mean()
    #     return L_n

    # def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None,mult_constant=True):
    #     if not mult_constant:
    #         N = 1
    #     # i in {1,n}
    #     # Algorithm 7 in BFN
    #     D = one_hot_x.size()[0]
    #     e_x = one_hot_x  # [D, K]
    #     e_hat = p_0  # (D, K)
    #     assert e_x.size() == e_hat.size()
    #     alpha = beta1 * (2 * i - 1) / N**2  # [D]
    #     mean_ = alpha * (K * e_x - 1)  # [D, K]
    #     std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
    #     eps = torch.randn_like(mean_)  # [D,K,]
    #     y_ = mean_ + std_ * eps  # [D, K]
    #     matrix_ek = torch.eye(K, K).to(e_x.device).unsqueeze(0).repeat(D,1,1)  # [D, K, K]
    #     mean_matrix = alpha.unsqueeze(-1) * (K * matrix_ek - 1)  # [K, K]
    #     std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
    #     LOG2PI = torch.log(torch.tensor(2 * np.pi))
    #     _log_gaussians = (  # [D, K]
    #         (-0.5 * LOG2PI - torch.log(std_matrix))
    #         - (y_.unsqueeze(1) - mean_matrix) ** 2 / (2 * std_matrix**2)
    #     ).sum(-1)
    #     _inner_log_likelihood = torch.log(torch.sum(e_hat * torch.exp(_log_gaussians), dim=-1))  # (D,)
    #     if segment_ids is not None:
    #         L_N = -scatter_mean(_inner_log_likelihood, segment_ids, dim=0)
    #     else:
    #         L_N = -_inner_log_likelihood.sum(dim=-1)  # [D]
    #     return N * L_N

    def dtime4discrete_loss_prob(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None, time_scheduler ="quad", beta_init=None
    ):
        # this is based on the official implementation of BFN.
        # import pdb
        # pdb.set_trace()
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)

        if time_scheduler == "quad":
            alpha = beta1 * (2 * i - 1) / (N**2)  # [N]
        elif time_scheduler == "linear":
            alpha = beta1 / N
        elif time_scheduler == "hybrid":
            assert beta_init is not None
            alpha = (beta1 - beta_init) * (2 * i - 1) / (N**2) + beta_init / N     
        else:
            raise NotImplementedError   
        
        alpha = alpha.view(-1, 1) # [D, 1]

        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = D.Independent(
            D.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = D.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = D.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        # pdb.set_trace()
        # print(receiver_dist.event_shape)

        sender_dist = D.Independent( D.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]

        y = sender_dist.sample(torch.Size([n_samples])) 

        # print(sender_dist.log_prob(y).size())
        # import pdb
        # pdb.set_trace()

        loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
        )

        # loss = (
        #         (sender_dist.log_prob(y) - receiver_dist.log_prob(y))
        #         .mean(0)
        #         .flatten(start_dim=1)
        #         .mean(1, keepdims=True)
        #     )
        # #
        return loss.mean()




    def dtime4discrete_loss(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None
    ):
        if K == 1:
            return torch.zeros_like(one_hot_x)
            n_samples = 5
        # this is based on the official implementation of BFN.
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)
        alpha = beta1 * (2 * i - 1) / (N**2)  # [D]
        alpha = alpha.view(-1, 1) # [D, 1]
        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = D.Independent(
            D.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = D.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = D.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        sender_dist = D.Independent( D.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]
        y = sender_dist.sample(torch.Size([n_samples])) 
        # if segment_ids != None:
        #     loss = scatter_mean(N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0), segment_ids, dim=0)
        # else:
        loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
        )
        
        return loss.mean()

    def dtime4circular_loss(self, i, N, alpha_i, x_pred, x, segment_ids=None, mult_constant=True, weight_norm=1, wn=False, mse_loss=True):
        if not mult_constant:
            N = 1
        freqs = torch.arange(-10,11).to(self.device).unsqueeze(0).unsqueeze(0)*np.pi*2
        tar_coord = x.unsqueeze(-1) + freqs
        coord_diff = (tar_coord - x_pred.unsqueeze(-1)).square().min(dim=-1).values
        weight = (i1e(alpha_i) / i0e(alpha_i)) * alpha_i   
        if mse_loss:
            loss = N * weight * coord_diff
        else:
            loss = N * weight * (1 - torch.cos(x_pred - x))
        if not wn:
            weight_norm = 1.
        return loss.mean() / weight_norm


    def ctime4circular_loss(self, t, beta1, x_pred, x, segment_ids,):
        alpha_t = beta1.pow(t) * (1 + t * torch.log(beta1))
        weight = i1e(alpha_t) * alpha_t / i0e(alpha_t)
        loss = 1 - torch.cos(x_pred - x)
        if segment_ids != None:
            return scatter_mean((weight * loss).sum(-1), segment_ids, dim=0)
        else:
            return (weight * loss).sum(-1)

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def dtime4sphere_loss(self, x, x_pred, alpha, p, ignore_weight = False):
        """
        x: [N, D]
        x_pred: [N, D]
        loss = alpha * (1 - x_pred^T x)
        """
        diff = 1 - torch.einsum("bi,bi->b", x, x_pred)
        # diff2 = 1 - torch.einsum("bi,bi->b", -x, x_pred)
        # diff = torch.min(diff, diff2)
        v = p / 2 - 1
        coef = alpha * ive(v+1,alpha) / ive(v,alpha)
        if ignore_weight:
            coef = 1.0
        loss = coef * diff
        return loss.mean()

    def sphere_var_bayesian_update(self, loc_prev, conc_prev, alpha_i, pred_x, beta1):
        scale = alpha_i.repeat(loc_prev.shape[0],1)
        y_i = VonMisesFisher(loc=pred_x,scale=scale).sample()
        theta = loc_prev * conc_prev.unsqueeze(-1) + scale * y_i
        conc = torch.linalg.norm(theta,dim=-1)    
        loc = theta / conc.unsqueeze(-1)
        conc = self.sphere_norm_log_conc((conc+self.epsilon).log(),beta1=beta1)
        return loc, conc


    def sphere_var_bayesian_flow_sim(self, x, t_index, beta1, N, epsilon=1e-7):
        # y,eps = self.vm_heler.sample(loc=x, concentration=beta_t, n_samples=1,ret_eps=True)
        # y = torch.tensor(scipy.stats.vonmises.rvs(loc=x.cpu().numpy(), kappa=beta_t.cpu().numpy())).cuda()
        # 使用torch自带的vonmises分布
        x = x.to(torch.float32)
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        n_batch, n_dim = x.shape
        t_index = t_index.long()
        t_index = t_index.unsqueeze(0).repeat(1,1,n_dim).view(-1,n_dim)
        device = x.device
        alpha_index = self.sphere_alpha_wrt_index(torch.arange(1, N+1).to(device).long(), N, beta1, self.sphere_dim) # [n_steps]
        alpha = alpha_index.unsqueeze(-1).unsqueeze(-1).repeat(1,n_batch,1) # （n_steps, n_batch, 1)
        sample_x = x.unsqueeze(0).repeat(alpha.shape[0],1,1).double() # (n_steps,n_batch,dim)
        torch_vmf = VonMisesFisher(loc=sample_x, scale=alpha.double().clip(float(epsilon)))
        y = torch_vmf.sample()
        prior = VonMisesFisher(loc=x.unsqueeze(0),scale=epsilon*torch.ones((1,n_batch,1),device=x.device)).sample()
        poster_cum = (y * alpha).cumsum(dim=0)
        posters = torch.cat([prior,poster_cum],dim=0) #[n_steps+1,n_batch,dim]
        selected_index = (t_index-1).unsqueeze(0)
        selected_poster = torch.gather(posters, dim=0, index=selected_index).squeeze(0)
        kappa = torch.functional.norm(selected_poster,dim=-1)
        mu = torch.nn.functional.normalize(selected_poster, p=2, dim=-1)
        log_kappa = (kappa+epsilon).log().float()
        log_kappa[(t_index==1)[:,0]] = torch.log(torch.tensor((epsilon))).to(device)
        normed_log_conc = self.sphere_norm_log_conc(log_kappa,beta1=beta1,epsilon=epsilon)
        return mu.view(input_shape).to(torch.float32), normed_log_conc.view(-1).to(torch.float32)

    def sphere_alpha_wrt_index(self, t_index, N, beta1, p):
        assert (t_index >= 1).all() and (t_index <= N).all()
        fname = f'./cache_files/sphere_schedules/linear_entropy_alphas_s{int(N)}_beta{int(beta1)}_dim{p}.pt'
        acc_schedule = torch.load(fname,map_location=t_index.device)
        return acc_schedule[t_index.long()-1]
    
    def sphere_denorm_conc(self, normed_log_conc, beta1, epsilon=1e-7):
        '''
        Denormalize logbeta 
        '''
        beta1 = torch.tensor(beta1)
        epsilon = torch.tensor(epsilon)
        log_conc = normed_log_conc * (torch.log(beta1) - torch.log(epsilon)) + torch.log(epsilon)
        return log_conc.exp()

    def sphere_norm_log_conc(self, log_kappa, beta1, epsilon=1e-7):
        '''
        Normalize logbeta to [0,1]
        '''
        return (log_kappa - math.log(epsilon))/(torch.log(beta1) - math.log(epsilon))

    def dtime4quat_loss(self, x, x_pred, alpha, p, ignore_weight = False):
        """
        x: [N, D]
        x_pred: [N, D]
        loss = alpha * (1 - x_pred^T x)
        """
        diff = 1 - torch.einsum("bi,bi->b", x, x_pred) ** 2
        # diff2 = 1 - torch.einsum("bi,bi->b", -x, x_pred)
        # diff = torch.min(diff, diff2)
        if ignore_weight:
            coef = 1.0
        else:
            coef = watson_KL_coef(alpha).unsqueeze(-1)
        loss = coef * diff
        return loss.mean()

    def quat_var_bayesian_update(self, eigvecs_prev, eigvals_prev, alpha_i, pred_x, beta1):
        scale = alpha_i.repeat(eigvals_prev.shape[0],1)
        watson_A = torch.einsum('bi,bj->bij', pred_x, pred_x) * scale.view(-1,1,1)
        y_i = Bingham(matrix=watson_A).sample()
        A_prev = torch.einsum('bij, bj, bkj', eigvecs_prev, eigvals_prev, eigvecs_prev)
        ayy_i = torch.einsum('bi,bj->bij', y_i, y_i) * scale.view(-1,1,1)
        A_post = A_prev + ayy_i
        bhm_post = Bingham(matrix=A_post)
        eigvecs_post, eigvals_post = bhm_post.eigvecs, bhm_post.eigvals
        # theta = loc_prev * conc_prev.unsqueeze(-1) + scale * y_i
        # conc = torch.linalg.norm(theta,dim=-1)    
        # loc = theta / conc.unsqueeze(-1)
        eigvals_post = self.quat_norm_log_eigvals((self.epsilon - eigvals_post).log(),beta1=beta1)
        return eigvecs_post, eigvals_post


    def quat_var_bayesian_flow_sim(self, x, t_index, beta1, N, epsilon=1e-7):
        # y,eps = self.vm_heler.sample(loc=x, concentration=beta_t, n_samples=1,ret_eps=True)
        # y = torch.tensor(scipy.stats.vonmises.rvs(loc=x.cpu().numpy(), kappa=beta_t.cpu().numpy())).cuda()
        # 使用torch自带的vonmises分布
        x = x.to(torch.float32)
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        n_batch, n_dim = x.shape
        t_index = t_index.long()
        t_index = t_index.unsqueeze(0).repeat(1,1,n_dim ** 2).view(-1,n_dim, n_dim)
        device = x.device
        alpha_index = self.quat_alpha_wrt_index(torch.arange(1, N+1).to(device).long(), N, beta1, self.sphere_dim) # [n_steps]
        alpha = alpha_index.unsqueeze(-1).unsqueeze(-1).repeat(1,n_batch,1) # （n_steps, n_batch, 1)
        sample_x = x.unsqueeze(0).repeat(alpha.shape[0],1,1).double() # (n_steps,n_batch,dim)
        bhm_matrix = torch.einsum('sbi, sbj -> sbij', sample_x, sample_x) * alpha.unsqueeze(-1)
        torch_bhm = Bingham(matrix = bhm_matrix.view(-1, n_dim, n_dim))
        y = torch_bhm.sample().view(-1, n_batch, n_dim)
        # prior = VonMisesFisher(loc=x.unsqueeze(0),scale=epsilon*torch.ones((1,n_batch,1),device=x.device)).sample()
        prior = torch.zeros(1, n_batch, n_dim, n_dim, device=device)
        poster_cum = (torch.einsum('sbi, sbj -> sbij', y, y) * alpha.unsqueeze(-1)).cumsum(dim=0)
        posters = torch.cat([prior,poster_cum],dim=0) #[n_steps+1,n_batch,dim, dim]
        selected_index = (t_index-1).unsqueeze(0)
        selected_poster = torch.gather(posters, dim=0, index=selected_index).squeeze(0)
        bhm_post = Bingham(matrix = selected_poster)
        eigvecs_post, eigvals_post = bhm_post.eigvecs, bhm_post.eigvals
        eigvals_post = self.quat_norm_log_eigvals((self.epsilon - eigvals_post).log(),beta1=beta1)
        return eigvecs_post.to(torch.float32), eigvals_post.to(torch.float32)

    def quat_alpha_wrt_index(self, t_index, N, beta1, p):
        assert (t_index >= 1).all() and (t_index <= N).all()
        fname = f'./cache_files/sphere_schedules/linear_entropy_bingham_alphas_s{int(N)}_beta{int(beta1)}_dim{p}.pt'
        acc_schedule = torch.load(fname,map_location=t_index.device)
        return acc_schedule[t_index.long()-1]
    
    def quat_denorm_eigvals(self, normed_log_eigvals, beta1, epsilon=1e-7):
        '''
        Denormalize logbeta 
        '''
        beta1 = torch.tensor(beta1)
        epsilon = torch.tensor(epsilon)
        log_conc = normed_log_eigvals * (torch.log(beta1) - torch.log(epsilon)) + torch.log(epsilon)
        return -log_conc.exp()

    def quat_norm_log_eigvals(self, log_kappa, beta1, epsilon=1e-7):
        '''
        Normalize logbeta to [0,1]
        '''
        return (log_kappa - math.log(epsilon))/(torch.log(beta1) - math.log(epsilon))


import time
import hydra
import omegaconf
from tqdm import tqdm
from crysbfn.common.data_utils import SinusoidalTimeEmbeddings, lattice_params_to_matrix_torch,back2interval, lattices_to_params_shape
from crysbfn.common.data_utils import PeriodHelper as p_helper
from crysbfn.pl_modules.bfn_base import bfnBase
from crysbfn.pl_modules.egnn.cspnet import CSPNet
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.special import i0e, i1e
from torch import FloatTensor, DoubleTensor, tensor
from torch import rad2deg
from overrides import overrides
from torch_scatter import scatter, scatter_mean
from crysbfn.common.von_mises_utils import VonMisesHelper
from crysbfn.common.utils import PROJECT_ROOT
from crysbfn.pl_modules.base_model import build_mlp
import scipy.stats as stats
from crysbfn.common.linear_acc_search import AccuracySchedule
import matplotlib.pyplot as plt
from torch.distributions import VonMises

class CrysBFN_CSP(bfnBase):
    def __init__(
        self,
        hparams,
        device="cuda",
        beta1_type=0.8,
        beta1_coord=1e6,
        sigma1_lattice=1e-3,
        K=88,
        t_min=0.0001,
        dtime_loss=True,
        dtime_loss_steps=1000,
        mult_constant= True,
        pred_mean = True,
        end_back=False,
        cond_acc = False,
        sch_type='exp',
        sim_cir_flow= True,
        **kwargs
    ):
        super(CrysBFN_CSP, self).__init__()
        self.hparams = hparams
        self.net:CSPNet = hydra.utils.instantiate(hparams.decoder, _recursive_=False, smooth = False)
        
        self.K = K
        self.t_min = t_min
        self.device = device
        self.beta1_type = tensor(beta1_type).to(device)
        self.beta1_coord = tensor(beta1_coord).to(device)
        self.dtime_loss = dtime_loss
        if dtime_loss:
            self.dtime_loss_steps = tensor(dtime_loss_steps).to(device)
        self.sigma1_lattice = tensor(sigma1_lattice).to(device)
        
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.t_min = t_min
        self.mult_constant = mult_constant
        self.atom_type_map = self.hparams.data.atom_type_map
        self.pred_mean = pred_mean
        self.end_back = end_back
        self.T_min = eval(str(self.hparams.T_min))
        self.T_max = eval(str(self.hparams.T_max))
        self.sim_cir_flow = sim_cir_flow
        self.cond_acc = cond_acc
        
        self.sch_type = sch_type
        if sch_type == 'linear':
            acc_schedule = AccuracySchedule(
                n_steps=self.dtime_loss_steps, beta1=self.beta1_coord, device=self.device)
            self.beta_schedule = acc_schedule.find_beta()
            self.beta_schedule = torch.tensor(
                                    [0.] + self.beta_schedule.cpu().numpy().tolist()).to(self.device)
            acc_sch = self.alpha_wrt_index(torch.arange(1, dtime_loss_steps+1).to(device).unsqueeze(-1),
                                      dtime_loss_steps, beta1_coord, sch_type='linear').squeeze(-1)
            self.acc_diff_mean = acc_schedule.analyze_acc_diff(acc_sch)
        elif sch_type == 'exp':
            acc_schedule = AccuracySchedule(
                n_steps=self.dtime_loss_steps, beta1=self.beta1_coord, device=self.device)
            self.beta_schedule = acc_schedule.find_diff_beta()
            self.beta_schedule = torch.tensor(
                                    [0.] + self.beta_schedule.cpu().numpy().tolist()).to(self.device)
        elif sch_type == 'add':
            steps = torch.range(0,self.dtime_loss_steps,1, device=self.device)
            t = steps / self.dtime_loss_steps
            self.beta_schedule = t * self.beta1_coord** t
        
        alphas = self.alpha_wrt_index(torch.arange(1, dtime_loss_steps+1).to(device).unsqueeze(-1),
                                      dtime_loss_steps, beta1_coord).squeeze(-1)
        self.alphas = torch.cat([torch.tensor([0.]), alphas.cpu()], dim=0).to(self.device)
        weights = self.dtime_loss_steps * (i1e(alphas) / i0e(alphas)) * alphas 
        self.cir_weight_norm =  weights.mean().detach()
        self.vm_helper:VonMisesHelper = VonMisesHelper(cache_sampling=False,
                                                      device=self.device,
                                                      sample_alphas=alphas,
                                                      num_vars=hparams.data.max_atoms*hparams.data.datamodule.batch_size.train*3,
                                                      )
        self.epsilon = torch.tensor(1e-7)
        self.norm_beta = False if not 'norm_beta' in self.hparams.keys() else hparams.norm_beta
        self.rej_samp = False if not 'rej_samp' in self.hparams.keys() else hparams.rej_samp
    
    def circular_var_bayesian_update(self, m, c, y, alpha):
        '''
        Compute (m_out, c_out) = h(m, c , y, α)
        according to 
        m_out = arctan((α sin(y) + c sin(m))/( α cos(y) + c cos(m))
        c_out = sqrt(α^2 + c^2 + 2αc cos(y-m))
        :param m: the previous mean, shape (D,)
        :param c: the previous concentration, shape (D,)
        return: m_out, c_out, shape (D,)
        '''
        m_out = torch.atan2(alpha * torch.sin(y) + c * torch.sin(m)+1e-6, 
                            alpha * torch.cos(y) + c * torch.cos(m)+1e-6)
        c_out = torch.sqrt(alpha**2 + c**2 + 2 * alpha * c * torch.cos(y - m))
        return m_out, c_out
    
    
    def beta_circ_wrt_t(self, t, beta1=None):
        '''
        Compute β(t) = t β_1^t
        :param t: index time, shape (N,)
        :param beta1: β_1, shape (1,)
        '''
        return self.beta_schedule[t.long()]
        # return  t * beta1.pow(t)
    
    def alpha_wrt_index(self, t_index_per_atom, N, beta1, sch_type='exp'):
        assert (t_index_per_atom >= 1).all() and (t_index_per_atom <= N).all()
        sch_type = self.hparams.BFN.sch_type
        if sch_type == 'exp' and not self.hparams.BFN.sim_cir_flow:
            assert len(t_index_per_atom.shape) == 2
            beta_i = self.beta_circ_wrt_t(t=t_index_per_atom)
            beta_prev = self.beta_circ_wrt_t(t=t_index_per_atom-1)
            alpha_i = beta_i - beta_prev
        elif sch_type == 'exp' and self.hparams.BFN.sim_cir_flow:
            fname = str(PROJECT_ROOT) + f'/cache_files/diff_sch_alphas_s{int(self.dtime_loss_steps)}_{self.beta1_coord}.pt'
            acc_schedule = torch.load(fname).to(self.device)
            return acc_schedule[t_index_per_atom.long()-1]
        elif sch_type == 'linear':
            fname = str(PROJECT_ROOT) + f'/cache_files/linear_entropy_alphas_s{int(self.dtime_loss_steps)}_{self.beta1_coord}.pt'
            acc_schedule = torch.load(fname).to(self.device)
            return acc_schedule[t_index_per_atom.long()-1]
        elif sch_type == 'add':
            t_i = t_index_per_atom 
            t_i_prev = t_index_per_atom - 1
            alpha_i = self.beta_circ_wrt_t(t=t_i, beta1=self.beta1_coord)\
                        - self.beta_circ_wrt_t(t=t_i_prev, beta1=self.beta1_coord)
        else:
            raise NotImplementedError
        return alpha_i
    
    def norm_logbeta(self, logbeta):
        '''
        Normalize logbeta to [0,1]
        '''
        if not self.norm_beta:
            return logbeta
        return (logbeta - torch.log(self.epsilon))/(torch.log(self.beta1_coord) - torch.log(self.epsilon))
    
    def denorm_logbeta(self, logbeta):
        '''
        Denormalize logbeta 
        '''
        if not self.norm_beta:
            return logbeta
        return logbeta * (torch.log(self.beta1_coord) - torch.log(self.epsilon)) + torch.log(self.epsilon)
    
    def circular_var_bayesian_flow(self, x, t, beta_t):
        '''
        Compute p(θ|x, t) = E_{vM(y|x,β(t))}δ(θ-h(θ, y, β(t)))
        the returned variable is in [-pi, pi)
        :param x: the input circular variable, shape (BxN, 3 or 1)
        :param t: time , shape (BxN,)
        :param beta_t: the concentration of the von Mises distribution, shape (BxN, 3 or 1)
        return: mu: the mean of the posterior, shape (BxN, 3 or 1)
                kappa: the concentration of the posterior, shape (BxN, 3 or 1)
        '''
        y = self.vm_helper.sample(loc=x, concentration=beta_t, n_samples=1)
        return y.detach().clone()
    
    @torch.no_grad()
    def circular_var_bayesian_flow_sim(self, x, t_index, beta1, n_samples=1, epsilon=1e-7):
        '''
        Compute p(θ|x, t) = E_{vM(y|x,β(t))}δ(θ-h(θ, y, β(t)))
        the returned variable is in [-pi, pi)
        :param x: the input circular variable, shape (BxN, 3 or 1)
        :param t: time , shape (BxN,)
        :param beta_t: the concentration of the von Mises distribution, shape (BxN, 3 or 1)
        return: mu: the mean of the posterior, shape (BxN, 3 or 1)
                kappa: the concentration of the posterior, shape (BxN, 3 or 1)
        '''
        t_index = t_index.long()
        alpha_index = self.alpha_wrt_index(torch.arange(1, self.dtime_loss_steps+1).to(self.device).unsqueeze(-1).long(), self.dtime_loss_steps, beta1).squeeze(-1)
        alpha = alpha_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,x.shape[0],x.shape[1],n_samples)
        # y = self.vm_helper.sample_cache(loc=x)
        y = self.vm_helper.sample(loc=x.unsqueeze(0).unsqueeze(-1).repeat(alpha.shape[0],1,1,n_samples), concentration=alpha, n_samples=1)
        # sample prior
        prior_mu = (2*torch.pi*torch.rand((1,x.shape[0],x.shape[1]))-torch.pi).to(self.device)
        prior_cos, prior_sin = prior_mu.cos(), prior_mu.sin()
        # sample posterior
        poster_cos_cum, poster_sin_cum = (alpha * y.cos()).cumsum(dim=0).mean(-1), (alpha * y.sin()).cumsum(dim=0).mean(-1)
        poster_cos_cum = torch.cat([prior_cos, poster_cos_cum], dim=0)
        poster_sin_cum = torch.cat([prior_sin, poster_sin_cum], dim=0)
        # concat prior and posterior
        poster_cos = torch.gather(poster_cos_cum, dim=0, 
                                  index=(t_index-1).unsqueeze(-1).unsqueeze(0).repeat(1,1,3)).squeeze(0)
        poster_sin = torch.gather(poster_sin_cum, dim=0, 
                                  index=(t_index-1).unsqueeze(-1).unsqueeze(0).repeat(1,1,3)).squeeze(0)
        poster_mu = torch.atan2(poster_sin, poster_cos)
        poster_kappa = (torch.sqrt(poster_cos**2 + poster_sin**2)+epsilon).log().float()
        # assign kappa = 0 for every t_index = 1
        poster_kappa[t_index==1] = torch.log(torch.tensor((epsilon))).to(self.device)
        # normalize log beta
        poster_kappa = self.norm_logbeta(poster_kappa)
        return poster_mu.detach(), poster_kappa.detach()
    
    @torch.no_grad()
    def circular_var_bayesian_flow_sim_sample(self, x, t_index, beta1, n_samples=1, epsilon=1e-7):
        '''
        Compute p(θ|x, t) = E_{vM(y|x,β(t))}δ(θ-h(θ, y, β(t)))
        the returned variable is in [-pi, pi)
        :param x: the input circular variable, shape (BxN, 3 or 1)
        :param t: time , shape (BxN,)
        :param beta_t: the concentration of the von Mises distribution, shape (BxN, 3 or 1)
        return: mu: the mean of the posterior, shape (BxN, 3 or 1)
                kappa: the concentration of the posterior, shape (BxN, 3 or 1)
        '''
        # assert 所有的 t_index 都是相同的
        assert (t_index == t_index[0]).all(), 't_index should be the same'
        idx = int(t_index[0].cpu())
        if idx == 1:
            mu = (2*torch.pi*torch.rand((x.shape[0],x.shape[1]))-torch.pi).to(self.device)
            kappa = torch.ones_like(mu) * torch.log(torch.tensor((self.epsilon))).to(self.device)
            return mu.detach(), kappa.detach()
        alpha_index = self.alpha_wrt_index(torch.arange(1, idx).to(self.device).long(), 
                                           self.dtime_loss_steps, beta1)
        alpha = alpha_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,x.shape[0],x.shape[1],n_samples) # (idx-1, n_atoms, 3, n_samples)
        y = self.vm_helper.sample(loc=x.unsqueeze(0).unsqueeze(-1).repeat(alpha.shape[0],1,1,n_samples), 
                                  concentration=alpha, n_samples=1)
        # sample posterior
        poster_cos_cum, poster_sin_cum = (alpha * y.cos()).cumsum(dim=0).mean(-1), (alpha * y.sin()).cumsum(dim=0).mean(-1)
        # poster_cos_cum, poster_sin_cum = (alpha * y.cos()).cumsum(dim=1), (alpha * y.sin()).cumsum(dim=1)
        poster_cos = poster_cos_cum[-1]
        poster_sin = poster_sin_cum[-1]
        poster_mu = torch.atan2(poster_sin, poster_cos)
        poster_kappa = (torch.sqrt(poster_cos**2 + poster_sin**2)+self.epsilon).log().float()
        poster_kappa = self.norm_logbeta(poster_kappa)
        return poster_mu.detach(), poster_kappa.detach()

    def back2interval(self, x):
        '''
        默认使用 [-pi, pi) 作为周期
        '''
        return back2interval(x)
        if self.pred_mean:
            return np.pi * torch.cos(np.pi*x)
        else:
            return back2interval(x)
    
    
    def interdependency_modeling(
            self,
            atom_types,
            t_index,
            mu_pos_t,
            segment_ids,
            gamma_lattices,
            num_atoms,
            mu_lattices_t=None,
            log_acc = None
            ):
        '''
        :param t_index: (num_molecules, ) shape
        :param mu_lattices_t: (num_molecules, 9) shape
        :param mu_pos_t: (num_atoms, 3) shape [T_min,T_max)
        :param mu_angles_t: (num_molecules, 3) shape (optional)
        '''
        mu_pos_t_in = mu_pos_t 
        mu_lattices_t_in = mu_lattices_t 
        # t_index must be (num_molecules, ) shape
        time_embed = self.time_embedding(t_index)
        lattice_final, coord_final, type_final = self.net.forward(
            time_embed, atom_types, mu_pos_t_in, mu_lattices_t_in, num_atoms, segment_ids, log_acc
        )
        # out for coord
        coord_pred = coord_final
        # out for lattice
        lattices_final = lattice_final.reshape(-1, 9)
        eps_lattices_pred = lattices_final # 暂时不用diff lattice
        mu_lattices_t = mu_lattices_t.reshape(-1, 9)
        # type out:
        # p_out_type = torch.nn.functional.softmax(type_final, dim=-1)
        if self.pred_mean:
            lattice_pred = eps_lattices_pred
        else:
            lattice_pred = (mu_lattices_t / gamma_lattices - torch.sqrt((1 - gamma_lattices) / gamma_lattices) * eps_lattices_pred)
        return coord_pred, lattice_pred
    
    def loss_one_step(self, 
                        t, 
                        atom_types, 
                        frac_coords,
                        lengths,
                        angles,
                        num_atoms,
                        segment_ids,
                        edge_index,
                      ):
        # sample time t 
        if t == None:
            # for receiver, t_{i-1} ~ U{0,1-1/N}
            t_per_mol = torch.randint(0, self.dtime_loss_steps, 
                                        size=num_atoms.shape, 
                                        device=self.device)/self.dtime_loss_steps
            t_index = t_per_mol * self.dtime_loss_steps + 1 # U{1,N}
        t_index_per_atom = t_index.repeat_interleave(num_atoms, dim=0)
        t_per_atom = t_per_mol.repeat_interleave(num_atoms, dim=0).unsqueeze(-1)
        t_per_mol = t_per_mol.unsqueeze(-1)
        # Bayesian flow for every modality to obtain input params
        # atom coord bayesian flow
        # [0,1) -> [T_min, T_max) 把frac_coords转为自定义的[T_min, T_max)比如fractional coordinate [0,1)
        pos = p_helper.frac2any(frac_coords % 1, self.T_min, self.T_max)
        # for anything related to Bayesian update, we need to transform to [-pi, pi)
        if self.hparams.BFN.sim_cir_flow:
            # time_start = time.time()
            cir_mu_pos_t, log_acc = self.circular_var_bayesian_flow_sim(
                                                x=p_helper.frac2circle(frac_coords%1), 
                                                t_index=t_index.repeat_interleave(num_atoms, dim=0), 
                                                beta1=self.beta1_coord,
                                                n_samples=int(self.hparams.n_samples)
                                                )
            # time_end = time.time()
            # print(f'torch_sample_time {time_end-time_start:.6f}s')
        else:
            # no simulation
            beta_t_pos = self.beta_circ_wrt_t(t_index_per_atom-1, self.beta1_coord).unsqueeze(-1).repeat(1,3)
            
            cir_mu_pos_t = self.circular_var_bayesian_flow(
                                                x=p_helper.frac2circle(frac_coords%1), 
                                                t=t_index_per_atom, 
                                                beta_t=beta_t_pos) 
            log_acc = None
        mu_pos_t = p_helper.circle2any(cir_mu_pos_t, self.T_min, self.T_max)
        # lattice bayesian flow
        lattices = lattice_params_to_matrix_torch(lengths, angles)
        lattices = lattices.reshape(-1, 9)
        lattices = (lattices - self.hparams.data.lattice_mean) / self.hparams.data.lattice_std
        mu_lattices_t, gamma_lattices = self.continuous_var_bayesian_flow(
                                            x=lattices, t=t_per_mol, 
                                            sigma1=self.sigma1_lattice)
        # reshape 为 (BxN, 3, 3)
        mu_lattices_t_33 = mu_lattices_t.reshape(-1, 3, 3)
        # coord pred is in [T_min, T_max)
        coord_pred, lattice_pred = self.interdependency_modeling(
            atom_types = atom_types,
            t_index = t_index,
            mu_pos_t=mu_pos_t,
            mu_lattices_t=mu_lattices_t_33,
            gamma_lattices=gamma_lattices,
            segment_ids=segment_ids,
            num_atoms=num_atoms,
            log_acc=log_acc
        )
        # discrete time loss
        t_index_per_atom = t_index.repeat_interleave(num_atoms, dim=0).unsqueeze(-1)
        lattice_loss = self.dtime4continuous_loss(
            i=t_index.unsqueeze(-1),
            N=self.dtime_loss_steps,
            sigma1=self.sigma1_lattice,
            x_pred=lattice_pred,
            x=lattices,
            segment_ids=None,
            mult_constant=self.mult_constant,
            wn = self.hparams.norm_weight
        )
        alpha_i = self.alpha_wrt_index(t_index_per_atom.long(),self.dtime_loss_steps,self.beta1_coord)
        coord_loss = self.dtime4circular_loss(
            i=t_index_per_atom,
            N=self.dtime_loss_steps,
            alpha_i=alpha_i,
            x_pred=p_helper.any2circle(coord_pred,self.T_min,self.T_max),
            x=p_helper.any2circle(pos,self.T_min,self.T_max),
            segment_ids=segment_ids,
            mult_constant=self.mult_constant,
            weight_norm=self.cir_weight_norm,
            wn=self.hparams.norm_weight
        )
        return lattice_loss.mean(), coord_loss.mean()
    
    
    @torch.no_grad()
    def init_params(self, num_atoms, segment_ids, batch, samp_acc_factor, start_idx, method = 'train'):
        if method == 'rand':
            # 随机初始化
            num_batch_atoms = num_atoms.sum()
            num_molecules = num_atoms.shape[0]
            # mu_pos_t = torch.zeros((num_batch_atoms, 3)).to(self.device)  # [N, 3] circular coordinates prior
            mu_pos_t = 2*np.pi*torch.rand((num_batch_atoms, 3)).to(self.device) - np.pi # [N, 3] circular coordinates prior
            mu_pos_t = p_helper.circle2any(mu_pos_t, self.T_min, self.T_max) # transform to [T_min, T_max)
            theta_type_t = torch.ones((num_batch_atoms, self.K)).to(self.device) / self.K  # [N, K] discrete prior
            # 把lattice视为9个连续变量
            mu_lattices_t = torch.zeros((num_molecules, 3, 3)).view(-1,9).to(self.device)  # [N, 9] continous lattice prior
            log_acc = self.norm_logbeta(
                            torch.log(torch.tensor((self.epsilon))) * torch.ones_like(mu_pos_t))
            rho_lattice = 1
            return num_molecules, mu_pos_t, theta_type_t, mu_lattices_t, log_acc, num_atoms, segment_ids, rho_lattice
        else:
            raise NotImplementedError

    def update_params(self, i, sample_steps, coord_pred, lattice_pred, mu_pos_t, log_acc, mu_lattices_t, rho_lattice, num_atoms, strategy='end_back'):
        num_molecules = lattice_pred.shape[0]
        if strategy == 'end_back':
            if i + 1 > sample_steps:
                return mu_pos_t, log_acc, mu_lattices_t, rho_lattice
            t_index = i * torch.ones((num_molecules, )).to(self.device)
            t_index_per_atom = t_index.repeat_interleave(num_atoms, dim=0).unsqueeze(-1)
            t_cts = torch.ones((num_molecules, 1)).to(self.device) * (i - 1) / sample_steps
            tplus1_per_mol = (t_cts + 1 / sample_steps).clamp(0, 1)
            tplus1_index_per_atom = t_index_per_atom + 1
            if not self.hparams.BFN.sim_cir_flow:
                beta_t = self.beta_circ_wrt_t(tplus1_index_per_atom-1, self.beta1_coord).repeat(1, 3)
                mu_pos_t = p_helper.circle2any(
                            self.circular_var_bayesian_flow(
                                x=p_helper.any2circle(coord_pred, self.T_min, self.T_max),
                                t=tplus1_index_per_atom-1,
                                beta_t=beta_t
                            ), 
                            self.T_min, self.T_max)
            else:
                cir_x = p_helper.any2circle(coord_pred, self.T_min, self.T_max)
                cir_mu_pos_t, log_acc = self.circular_var_bayesian_flow_sim_sample(
                    x=cir_x,
                    t_index=tplus1_index_per_atom.squeeze(-1),
                    beta1=self.beta1_coord,
                    n_samples=1
                )
                mu_pos_t = p_helper.circle2any(
                            cir_mu_pos_t, 
                            self.T_min, self.T_max)
            mu_lattices_t = self.continuous_var_bayesian_flow(
                x=lattice_pred,
                t=tplus1_per_mol,
                sigma1=self.sigma1_lattice,
                # n_samples=int(samp_acc_factor)
            )[0]
            alpha_lattice = torch.pow(self.sigma1_lattice, -2 * i / sample_steps) * (
                    1 - torch.pow(self.sigma1_lattice, 2 / sample_steps)
            )
            rho_lattice = rho_lattice + alpha_lattice
            return mu_pos_t, log_acc, mu_lattices_t, rho_lattice
        elif strategy=='vanilla':
            alpha_lattice = torch.pow(self.sigma1_lattice, -2 * i / sample_steps) * (
                    1 - torch.pow(self.sigma1_lattice, 2 / sample_steps)
            )
            y_lattice = lattice_pred + torch.randn_like(lattice_pred) * torch.sqrt(
                1 / alpha_lattice
            )       
            mu_lattices_t = (rho_lattice * mu_lattices_t + alpha_lattice * y_lattice) / (
                rho_lattice + alpha_lattice
            )
            rho_lattice = rho_lattice + alpha_lattice
            alpha_coord = self.alphas[i]
            y_coord = self.vm_helper.sample(loc=coord_pred, concentration=torch.ones_like(coord_pred)*alpha_coord, n_samples=1)
            mu_pos_t, acc_next = self.circular_var_bayesian_update(mu_pos_t,self.denorm_logbeta(log_acc).exp(),y_coord, alpha_coord)
            return  mu_pos_t, self.norm_logbeta((acc_next+self.epsilon).log()), mu_lattices_t, rho_lattice 
        else:
            raise NotImplementedError

    @torch.no_grad()
    def sample(
        self, 
        atom_types,
        num_atoms, 
        edge_index, 
        sample_steps=None, 
        edge_attr=None, 
        segment_ids=None,
        show_bar=False,
        return_traj=False,
        samp_acc_factor=1,
        batch = None,
        strategy = 'end_back',
        **kwargs
    ):
        # 随机初始化
        start_idx = 1
        traj = []
        # low noise sampling
        if 'n_samples' in self.hparams.keys():
            samp_acc_factor = int(self.hparams.n_samples) if int(samp_acc_factor) == 1 else samp_acc_factor
        
        rand_back = False
        back_sampling = False if 'back_sampling' not in kwargs.keys() else kwargs['back_sampling']
        sample_passes = 1
        
        print(f"Sampling with low noise with factor, {samp_acc_factor}, perform rejection sampling. {self.rej_samp}, ",
                        f"sample_passes {sample_passes}, rand_back {rand_back}, strategy {strategy} ")
        
        ret_coord_pred, ret_lattice_pred = None, None
        for sample_pass_idx in range(sample_passes):
            print(f"Sample pass {sample_pass_idx}\n")
            # num_molecules, mu_pos_t, theta_type_t, mu_lattices_t, log_acc, num_atoms, segment_ids
            num_molecules, mu_pos_t, _, mu_lattices_t, log_acc, num_atoms, segment_ids, rho_lattice = self.init_params(
                        num_atoms, segment_ids, batch, samp_acc_factor,start_idx=start_idx, method='rand')
            # sampling loop
            for i in tqdm(range(1,sample_steps+1),desc='Sampling',disable=not show_bar):
                t_index = i * torch.ones((num_molecules, )).to(self.device)
                t_index_per_atom = t_index.repeat_interleave(num_atoms, dim=0).unsqueeze(-1)
                t_cts = torch.ones((num_molecules, 1)).to(self.device) * (i - 1) / sample_steps
                t_cts_per_atom = t_cts.repeat_interleave(num_atoms, dim=0)
                # interdependency modeling
                gamma_lattices = 1 - torch.pow(self.sigma1_lattice, 2 * t_cts)
                coord_pred, lattice_pred = \
                        self.interdependency_modeling(
                        atom_types=atom_types,
                        t_index=t_index,
                        mu_pos_t=mu_pos_t,
                        segment_ids=segment_ids,
                        mu_lattices_t=mu_lattices_t,
                        gamma_lattices=gamma_lattices,
                        num_atoms=num_atoms,
                        log_acc=log_acc
                    )
                # update the parameters via end back
                if strategy == 'end_back':
                    mu_pos_t, log_acc, mu_lattices_t, rho_lattice = \
                        self.update_params(i, sample_steps, coord_pred, lattice_pred, 
                                           mu_pos_t, log_acc, mu_lattices_t, rho_lattice, num_atoms, strategy='end_back')                    
                elif strategy == 'vanilla':
                    mu_pos_t, log_acc, mu_lattices_t, rho_lattice = \
                        self.update_params(i, sample_steps, coord_pred, lattice_pred, 
                                           mu_pos_t, log_acc, mu_lattices_t, rho_lattice, num_atoms,'vanilla')
                elif strategy == 'mix':
                    if i < self.dtime_loss_steps * (2 / 3): # 600 for mp20
                        mu_pos_t, log_acc, mu_lattices_t, rho_lattice = \
                        self.update_params(i, sample_steps, coord_pred, lattice_pred, 
                                           mu_pos_t, log_acc, mu_lattices_t, rho_lattice, num_atoms, strategy='end_back')           
                    else:
                        mu_pos_t, log_acc, mu_lattices_t, rho_lattice = \
                        self.update_params(i, sample_steps, coord_pred, lattice_pred, 
                                           mu_pos_t, log_acc, mu_lattices_t, rho_lattice, num_atoms,'vanilla')
                else:
                    raise NotImplementedError
                # add trajectory
                if 'debug_mode' in self.hparams.logging.keys() and self.hparams.logging.debug_mode:
                    lengths_pred, angles_pred = lattices_to_params_shape(lattice_pred.reshape(-1,3,3)) 
                    inverse_map = {v: k for k, v in self.atom_type_map.items()}
                    traj.append({
                        'log_acc': log_acc.cpu(),
                        'frac_coords': p_helper.any2frac(coord_pred,eval(str(self.T_min)),eval(str(self.T_max))).cpu(),
                        'atom_types': torch.tensor([inverse_map[type.item()] for type in atom_types], device=self.device).cpu(),
                        'lengths': lengths_pred.cpu(),
                        'angles': angles_pred.cpu(),
                        'segment_ids': segment_ids.cpu(),
                        'num_atoms': num_atoms.cpu()
                    })
        
        ret_lattice_pred = lattice_pred if ret_lattice_pred is None else ret_lattice_pred
        ret_coord_pred = coord_pred if ret_coord_pred is None else ret_coord_pred

        if return_traj:
            return ret_coord_pred, ret_lattice_pred, traj
        return ret_coord_pred, ret_lattice_pred
    
@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    device = 'cuda'
    # device = 'cpu'
    batch = next(iter(datamodule.train_dataloader())).to(device)
    print(batch)
    vm_bfn = CrysBFN_CSP(device=device,hparams=cfg)
    vm_bfn.train_dataloader = datamodule.train_dataloader
    result_dict = vm_bfn.loss_one_step(
        t = None,
        atom_type = batch.atom_types,
        frac_coords = batch.frac_coords,
        lengths = batch.lengths,
        angles = batch.angles,
        num_atoms = batch.num_atoms,
        segment_ids= batch.batch,
        edge_index = batch.fully_connected_edge_index,
    )
    return result_dict    


        
if __name__ == '__main__':
    main()
        
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from crysbfn.common.bingham_utils import Bingham
# print(os.getcwd())

import torch
from tqdm import tqdm
from torch.special import i0e, i1e
from scipy.optimize import root_scalar
from scipy.stats import circvar,circstd
# from scipy.stats import circstd as circvar
from fisher.bingham_utils import bingham_F, bingham_dF, bingham_entropy

class AccuracySchedule(torch.nn.Module):
    def __init__(self, dim, n_steps, beta1, device='cuda:0'):
        super(AccuracySchedule, self).__init__()
        self.dim = dim
        self.n_steps = n_steps
        # self.vm_helper = VonMisesFisher(kappa1=1)
        self.device = device
        self.beta1 = torch.tensor(beta1,device=device)
    
    def linear_entropy(self, step):
        assert (step <= self.n_steps).all() and (step >= 0).all()
        t = step / self.n_steps
        # entropy1 = VonMisesFisher.entropy_wrt_kappa(self.dim, self.beta1)
        # entropy0 = VonMisesFisher.entropy_wrt_kappa(self.dim, torch.tensor(1.0e-6, device=self.device))
        # print(entropy0, entropy1)
        entropy0 = bingham_entropy(torch.tensor([[0., 0., 0., 0.]]))[0].to(self.device)
        entropy1 = bingham_entropy(torch.tensor([[0., -self.beta1, -self.beta1, -self.beta1]]))[0].to(self.device)
        slope = entropy1 - entropy0
        return entropy0 + slope * t
               
    
    
    def entropy_equation(self, tar_entropy):
        return lambda kappa: bingham_entropy(torch.tensor([[0, -kappa, -kappa, -kappa]]))[0] - tar_entropy
    
    def find_beta(self):
        steps = torch.range(1,self.n_steps,1, device=self.device).long()
        linear_entropies = self.linear_entropy(steps).unsqueeze(-1)
        # print(linear_entropies)
        betas = []
        for i in tqdm(range(len(linear_entropies))):
            tar_entropy = linear_entropies[i].cpu()
            root = root_scalar(self.entropy_equation(tar_entropy),bracket=[1.0e-6,self.beta1])
            if root.converged:
                betas.append(root.root)
            else:
                assert False, 'root not converged!'
        return torch.tensor(betas)
    


    def alpha_equation(self, prior_beta, tar_beta, n_samples=10000):
        def func(alpha):
            A = torch.tensor([[0, -alpha, -alpha, -alpha]])
            dF = bingham_dF(A)[0]
            F = bingham_F(A)[0]
            Sigma = dF / F
            delta_beta = 1 - 4 * Sigma[1]
            return tar_beta - prior_beta - alpha * delta_beta
        return func

    def alpha_equation_direct(self, prior_beta, tar_entropy, n_samples=10000):
        def func(alpha):
            # A = prior_beta.unsqueeze(0).repeat(n_samples, 1)
            # diagA = torch.diag_embed(A)
            # A_new = torch.tensor([[0, -alpha, -alpha, -alpha]]).repeat(n_samples, 1)
            # diagA_new = torch.diag_embed(A_new)
            # samples = Bingham(matrix=diagA_new).sample()
            # new_outer = torch.einsum('bi, bj-> bij', samples, samples) * alpha
            # A_post = A + new_outer
            # eigv_post = Bingham(matrix=A_post).eigvals.mean(dim=0)
            eigv_post = self.update_poster(prior_beta, alpha, n_samples=n_samples)
            entropy = bingham_entropy(eigv_post.unsqueeze(0))[0]
            return entropy - tar_entropy
        return func

    def update_poster(self, prior_beta, alpha, n_samples=10000):
        A = prior_beta.unsqueeze(0).repeat(n_samples, 1)
        diagA = torch.diag_embed(A)
        A_new = torch.tensor([[0, -alpha, -alpha, -alpha]]).repeat(n_samples, 1).to(prior_beta.device)
        diagA_new = torch.diag_embed(A_new)
        samples = Bingham(matrix=diagA_new).sample()
        new_outer = torch.einsum('bi, bj-> bij', samples, samples) * alpha
        A_post = diagA + new_outer
        eigv_post = Bingham(matrix=A_post).eigvals.mean(dim=0)
        return eigv_post



    @torch.no_grad()
    def find_linear(self, n_samples=100000):
        steps = torch.range(1,self.n_steps,1, device=self.device).long()
        linear_entropies = self.linear_entropy(steps).unsqueeze(-1)
        res_betas = self.find_beta()
        sender_alpha = [] # search sender alpha
        sender_alpha.append(res_betas[0])
        for i in tqdm(range(1,self.n_steps)):
            prior_beta = res_betas[i-1] #上一步达到的beta
            target_beta = res_betas[i] #目标beta
            root_alpha = root_scalar(self.alpha_equation(prior_beta, target_beta, n_samples=n_samples),
                                     bracket=[0.001, self.beta1])
            assert root_alpha.converged, 'alpha root not converged!'
            sender_alpha.append(torch.tensor(root_alpha.root))
        return torch.stack(sender_alpha) 

    @torch.no_grad()
    def find_linear_direct(self, n_samples=10000):
        steps = torch.range(1,self.n_steps,1, device=self.device).long()
        linear_entropies = self.linear_entropy(steps).unsqueeze(-1)
        prior_beta = torch.zeros(4, device=self.device)
        sender_alpha = []
        for i in tqdm(range(0, self.n_steps)):
            target_entropy = linear_entropies[i].cpu()
            root_alpha = root_scalar(self.alpha_equation_direct(prior_beta.cpu(), target_entropy, n_samples=n_samples),
                                     bracket=[0.001, self.beta1 * 10])   
            assert root_alpha.converged, 'alpha root not converged!'
            alpha_val = root_alpha.root
            sender_alpha.append(torch.tensor(root_alpha.root))   
            prior_beta = self.update_poster(prior_beta, alpha_val, n_samples=n_samples)
            print(prior_beta)
        return torch.stack(sender_alpha) 


    
    @torch.no_grad()
    def analyze_schedule(self, schedule=None, n_samples=100000):
        steps = torch.range(1,self.n_steps,1, device=self.device).long().to(self.device)
        if schedule == None:
            schedule = self.get_alpha(steps, schedule='add')
        assert schedule.shape == (self.n_steps,)
        schedule = schedule.to(self.device)
        # x = torch.tensor(uniform_direction.rvs(dim=3,size=(self.n_steps, n_samples)),device=self.device)
        # x = torch.tensor([np.sqrt(1/self.n_dim)]*self.n_dim,device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_steps,n_samples,1)
        # prior_mean = torch.tensor([np.sqrt(0.5),np.sqrt(0.5),0],device=self.device).unsqueeze(0).repeat(n_samples,1)
        # sender_alpha = schedule.unsqueeze(-1).repeat(1,n_samples).unsqueeze(-1).to(self.device)
        sender_alpha = schedule.repeat(n_samples).to(self.device)
        matrices = torch.zeros(self.n_steps * n_samples, 4, 4).to(self.device)
        matrices[:, 0, 0] = sender_alpha
        print(f"Start Sampling, {matrices.shape}")
        torch_bhm = Bingham(matrix=matrices)
        y = torch_bhm.sample().to(self.device)
        print(f"Sampled, {y.shape}")
        yy = torch.einsum('bi,bj->bij', y, y)
        ayy = yy * sender_alpha.unsqueeze(-1).unsqueeze(-1)
        ayy = ayy.view(n_samples, self.n_steps, 4, 4)
        poster = ayy.cumsum(dim=1)
        eigv = Bingham(matrix=poster.view(-1, 4, 4)).eigvals
        eigv = eigv.view(n_samples, self.n_steps, 4).mean(dim = 0)
        print("Begin Calc Entropy")
        entropy = bingham_entropy(eigv)
        print("After Calc Entropy")
        # entropy = vonmises_fisher.entropy(None,poster_acc)
        entropy = entropy.cpu().numpy()
        
        # 绘制entropy图
        plt.figure(dpi=300)
        plt.plot(entropy,label='simulated entropy')
        linear_entropy = self.linear_entropy(steps)
        plt.plot(linear_entropy.cpu(),label='linear entropy')
        plt.legend()
        plt.show()
        plt.savefig(f'cache_files/sphere_schedules/linear_entropy_bingham_alphas_s{self.n_steps}_beta{int(self.beta1)}_dim{self.dim}.png')
        return entropy    
    
    
if __name__ == '__main__':
    n_steps = 50
    beta1 = 2e2
    n_iters= 10000
    min_loss = 1e6
    dim = 4
    # fname = f'./cache_files/sphere_schedules/linear_entropy_bingham_alphas_s{n_steps}_beta{int(beta1)}_dim{dim}_direct.pt'
    fname = f'./cache_files/sphere_schedules/linear_entropy_bingham_alphas_s{n_steps}_beta{int(beta1)}_dim{dim}.pt'
    find_linear = True

    acc_schedule = AccuracySchedule(n_steps=n_steps,beta1=beta1,dim=dim)
    t = torch.range(0, n_steps-1, 1) / n_steps
    if find_linear:
        sender_alphas = acc_schedule.find_linear(n_samples=10000)
        torch.save(sender_alphas, fname)
    else:
        if os.path.exists(fname):
            sender_alphas = torch.load(fname)
        else:
            raise FileNotFoundError
    linear_entropy = acc_schedule.analyze_schedule(sender_alphas, n_samples=10000)
    print('designed beta:', beta1)
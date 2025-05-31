import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from crysbfn.common.von_mises_utils import VonMisesHelper
# print(os.getcwd())

import torch
from tqdm import tqdm
from torch.special import i0e, i1e
from scipy.optimize import root_scalar
from scipy.stats import circvar,circstd
# from scipy.stats import circstd as circvar

class AccuracySchedule(torch.nn.Module):
    def __init__(self, n_steps, beta1, device='cuda:0'):
        super(AccuracySchedule, self).__init__()
        self.n_steps = n_steps
        self.vm_helper = VonMisesHelper(kappa1=1)
        self.device = device
        self.beta1 = torch.tensor(beta1,device=device)
        self.alpha_schedule = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # torch.nn.functional.
        ).to(device)
        
    
    def beta_schedule(self, t):
        return t * self.beta1** t
    
    def linear_entropy(self, step):
        assert (step <= self.n_steps).all() and (step >= 0).all()
        t = step / self.n_steps
        entropy1 = self.vm_helper.entropy_wrt_kappa(self.beta1)
        entropy0 = self.vm_helper.entropy_wrt_kappa(torch.tensor(0.))
        slope = entropy1 - entropy0
        return entropy0 + slope * t
    
    def get_beta(self, step:torch.tensor):
        assert (step <= self.n_steps).all() and (step >= 0).all()
        t = step / self.n_steps
        return self.beta_schedule(t)              
    
    def get_alpha(self, step:torch.tensor, schedule='add'):
        assert (step <= self.n_steps).all() and (step >= 1).all()
        if schedule == 'exp':
            sch = torch.FloatTensor(
                np.exp(np.linspace(np.log(0.25), np.log(0.01), self.n_steps))).to(self.device)
            alpha = (1/sch[step.long()-1]).square()
        elif schedule == 'mlp':
            # print(step)
            # return 
            step = step / self.n_steps
            alpha = self.alpha_schedule(step.unsqueeze(-1)).square().squeeze(-1).exp()
        elif schedule == 'add':
            step_prev = step - 1
            alpha = self.get_beta(step) - self.get_beta(step_prev)
        else:
            raise NotImplementedError
        return alpha
    
    def simulate(self, steps=None, n_samples=10000, ret_acc=False, sim_schedule='exp'):
        # alphas = self.get_alpha(steps)
        if steps == None:
            steps = torch.range(1,self.n_steps,1, device=self.device)
        alphas = self.get_alpha(steps, schedule=sim_schedule)
        x = torch.zeros_like(steps) + 0.5
        y = self.vm_helper.sample(loc=x, concentration=alphas,n_samples=n_samples).detach().clone()
        theta_x = (y.cos() * alphas).cumsum(dim=-1)
        theta_y = (y.sin() * alphas).cumsum(dim=-1)
        acc = torch.sqrt(theta_x**2 + theta_y**2)
        acc_mean = acc.mean(dim=0)
        if sim_schedule == 'mlp':
            return self.vm_helper.entropy_wrt_kappa(acc_mean)
        entropy = self.vm_helper.entropy_wrt_kappa(torch.tensor([0.]+acc_mean.cpu().numpy().tolist()))
        if ret_acc:
            return entropy, acc_mean
        return entropy
    
    def forward(self):
        steps = torch.range(1,self.n_steps,1, device=self.device).requires_grad_(True)
        acc_mean = self.simulate(steps, sim_schedule='mlp')
        entropy = self.vm_helper.entropy_wrt_kappa(acc_mean)
        linear_entropy = self.linear_entropy(steps)
        return (entropy - linear_entropy).square().mean()
    
    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        for i in range(n_iters):
            loss = self.forward()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < min_loss:
                acc_schedule_best = self.alpha_schedule.parameters()
            print(f'iter {i} loss {loss.item()}')
        return acc_schedule_best
    
    @torch.no_grad()
    def plot_acc(self,acc):
        acc_mean = acc.mean(dim=0)
        import matplotlib.pyplot as plt
        plt.figure(dpi=300)
        plt.plot(acc_mean.log().cpu().numpy(),label='log beta(t) mean')
        plt.fill_between(range(len(acc_mean)),(acc_mean-acc.std(dim=0)).log().cpu().numpy(),(acc_mean+acc.std(dim=0)).log().cpu().numpy(),alpha=0.5,label='log beta(t) std')
        plt.scatter(range(len(acc_mean)),acc.min(dim=0).values.log().cpu().numpy(),label='log beta(t) min',s=1)
        plt.scatter(range(len(acc_mean)),acc.max(dim=0).values.log().cpu().numpy(),label='log beta(t) max',s=1)
        # plt.plot(acc_var.cpu().numpy(),label='log beta(t) variance')
        plt.legend()
        plt.xlabel('step')
        plt.ylabel('log beta(t)')
        plt.show()
        plt.savefig('./cache_files/log_beta_variance.png')
        
        
        plt.figure(dpi=300)
        plt.plot(acc_mean.cpu().numpy(),label='beta(t) mean')
        plt.fill_between(range(len(acc_mean)),(acc_mean-acc.std(dim=0)).cpu().numpy(),(acc_mean+acc.std(dim=0)).cpu().numpy(),alpha=0.5,label='beta(t) std')
        plt.scatter(range(len(acc_mean)),acc.min(dim=0).values.cpu().numpy(),label='beta(t) min',s=1)
        plt.scatter(range(len(acc_mean)),acc.max(dim=0).values.cpu().numpy(),label='beta(t) max',s=1)
        # plt.plot(acc_var.cpu().numpy(),label='log beta(t) variance')
        plt.legend()
        plt.xlabel('step')
        plt.ylabel('beta(t)')
        plt.show()
        plt.savefig('./cache_files/beta_variance.png')
        
        plt.figure(dpi=300)
        plt.plot(acc[0].cpu(),label='beta(t) sample1')
        plt.plot(acc[1].cpu(),label='beta(t) sample2')
        plt.plot(acc[2].cpu(),label='beta(t) sample3')
        plt.legend()
        plt.show()
        plt.savefig('./cache_files/beta_samples.png')
        plt.figure(dpi=300)
        plt.plot(acc[0].cpu().log(),label='log beta(t) sample1')
        plt.plot(acc[1].cpu().log(),label='log beta(t) sample2')
        plt.plot(acc[2].cpu().log(),label='log beta(t) sample3')
        plt.legend()
        plt.show()
        plt.savefig('./cache_files/log_beta_samples.png')
    
    @torch.no_grad()
    def analyze_acc_diff(self,schedule):
        steps = torch.range(1,self.n_steps,1, device=self.device).long().to(self.device)
        schedule = schedule.to(self.device)
        alphas = schedule[steps-1]
        x = torch.zeros_like(steps) - np.pi / 3
        x = x.to(self.device)
        y = self.vm_helper.sample(loc=x, concentration=alphas,n_samples=10000).detach().clone()
        theta_x = (y.cos() * alphas).cumsum(dim=-1)
        theta_y = (y.sin() * alphas).cumsum(dim=-1)
        acc = torch.sqrt(theta_x**2 + theta_y**2)
        acc = torch.cat([torch.zeros_like(acc[...,0]).unsqueeze(1),acc],dim=1)
        acc_diff = acc[...,1:] - acc[...,:-1]
        # acc_diff_mean = acc_diff.mean(dim=0)
        acc_diff_std = acc_diff.std(dim=0)
        # plt.plot(acc_diff_mean.cpu().numpy(),label='beta(t) mean')
        # plt.fill_between(range(len(acc_diff_mean)),(acc_diff_mean-acc_diff_std).cpu().numpy(),(acc_diff_mean+acc_diff_std).cpu().numpy(),alpha=0.5,label='beta(t) std')
        # plt.savefig('./cache_files/beta_diff_variance.png')
        return acc_diff.mean(dim=0)
        
    
    @torch.no_grad()
    def analyze_schedule(self, schedule=None, n_samples=100000):
        steps = torch.range(1,self.n_steps,1, device=self.device).long().to(self.device)
        if schedule == None:
            schedule = self.get_alpha(steps, schedule='add')
        assert schedule.shape == (self.n_steps,)
        schedule = schedule.to(self.device)
        alphas = schedule[steps-1]
        x = torch.zeros_like(steps) - np.pi / 3
        x = x.to(self.device)
        y = self.vm_helper.sample(loc=x, concentration=alphas,n_samples=n_samples).detach().clone()
        theta_x = (y.cos() * alphas).cumsum(dim=-1)
        theta_y = (y.sin() * alphas).cumsum(dim=-1)
        acc = torch.sqrt(theta_x**2 + theta_y**2)
        anaylyze_res = self.analyze_acc(acc)
        acc_mean = acc.mean(dim=0)
        entropy = self.vm_helper.entropy_wrt_kappa(acc_mean) # 关心 entropy
        mu = torch.atan2(theta_y,theta_x)
        prior_mu = torch.rand_like(mu[:,0]) * 2 * np.pi - np.pi
        final_acc = acc_mean[-1] # 关心 beta_1
        mu = torch.cat([prior_mu.unsqueeze(-1),mu],dim=-1)
        input_var = circvar(mu.cpu().numpy(),low=-np.pi,high=np.pi,axis=0,) # 关心 circ input variance
        print('the final acc is',final_acc)
        weights = alphas * i1e(alphas) / i0e(alphas)
        print(f'the mean of weights is {weights.mean().item()}')
        return final_acc, entropy, torch.tensor(input_var[:-1]), (acc+1e-10).log().var(dim=0)
    
    def entropy_equation(self, tar_entropy):
        return lambda kappa: self.vm_helper.entropy_wrt_kappa(torch.tensor(kappa)) - tar_entropy
    
    def find_beta(self):
        steps = torch.range(1,self.n_steps,1, device=self.device).long()
        linear_entropies = self.linear_entropy(steps).unsqueeze(-1)
        betas = []
        for i in tqdm(range(len(linear_entropies))):
            tar_entropy = linear_entropies[i].cpu()
            root = root_scalar(self.entropy_equation(tar_entropy),bracket=[0,self.beta1])
            if root.converged:
                betas.append(root.root)
            else:
                assert False, 'root not converged!'
        return torch.tensor(betas)
    
    def cirvar_equation(self, tar_cirvar):
        return lambda kappa: ((1-i1e(torch.tensor(kappa))/i0e(torch.tensor(kappa))) - tar_cirvar)
    
    @torch.no_grad()
    def find_diff_beta(self):
        sigmas = torch.linspace(np.log(0.5*2*np.pi),np.log(0.005*2*np.pi),self.n_steps).exp().to(self.device)
        diff_cirvar = 1 - torch.exp(-sigmas ** 2 / 2)
        betas = []
        for i in tqdm(range(len(diff_cirvar))):
            tar_cirvar = diff_cirvar[i]
            root = root_scalar(self.cirvar_equation(tar_cirvar),bracket=[0,1e6])
            if root.converged:
                betas.append(root.root)
            else:
                assert False, 'root not converged!'
        return torch.tensor(betas)

    def alpha_equation(self, prior_beta, tar_beta, n_samples=10000):
        def func(alpha):
            prior_mean = torch.ones(size=(n_samples,)) * (torch.rand(size=(1,))[0])
            sender_alpha = alpha * torch.ones_like(prior_mean)
            y = self.vm_helper.sample(loc=prior_mean,
                                      concentration=sender_alpha, 
                                      n_samples=1)
            poster_cos = prior_beta * torch.cos(prior_mean) + sender_alpha * torch.cos(y)
            poster_sin = prior_beta * torch.sin(prior_mean) + sender_alpha * torch.sin(y)
            poster_acc = (torch.sqrt(poster_cos**2 + poster_sin**2)).mean(dim=0)
            return poster_acc - tar_beta
        return func

    @torch.no_grad()
    def find_diff_alpha(self, n_samples=100000):
        res_betas = self.find_diff_beta()
        sender_alpha = [] # search sender alpha
        sender_alpha.append(res_betas[0])
        for i in tqdm(range(1,self.n_steps)):
            prior_beta = res_betas[i-1] #上一步达到的beta
            target_beta = res_betas[i] #目标beta
            root_alpha = root_scalar(self.alpha_equation(prior_beta, target_beta, n_samples=n_samples),
                                     bracket=[target_beta-prior_beta, self.beta1])
            assert root_alpha.converged, 'alpha root not converged!'
            sender_alpha.append(torch.tensor(root_alpha.root))
        return torch.stack(sender_alpha) 


    @torch.no_grad()
    def find_linear(self, n_samples=100000):
        res_betas = self.find_beta()
        sender_alpha = [] # search sender alpha
        sender_alpha.append(res_betas[0])
        for i in tqdm(range(1,self.n_steps)):
            prior_beta = res_betas[i-1] #上一步达到的beta
            target_beta = res_betas[i] #目标beta
            root_alpha = root_scalar(self.alpha_equation(prior_beta, target_beta, n_samples=n_samples),
                                     bracket=[target_beta-prior_beta, self.beta1])
            assert root_alpha.converged, 'alpha root not converged!'
            sender_alpha.append(torch.tensor(root_alpha.root))
        return torch.stack(sender_alpha)          
    
    def log_phi_q(self, q, max_k=200):
        assert (q > -1).all() and (q < 1).all()
        ks = torch.range(1,max_k,1,device=self.device).unsqueeze(-1).repeat(1,len(q))
        return (1-q.unsqueeze(0).repeat(max_k,1).pow(ks)).log().sum(0)
         
    
    def analyze_diff(self, n_samps=100000,n_steps=1000, start=0, max_k=200):
        steps = torch.range(1,n_steps,1, device=self.device).long()
        x = torch.zeros_like(steps) + start
        x = x.unsqueeze(0).repeat(n_samps,1)
        sigmas = torch.linspace(np.log(0.5),np.log(0.005),n_steps).exp().unsqueeze(0).repeat(n_samps,1).to(self.device)
        y = (torch.randn(size=(n_samps,n_steps),device=self.device) * sigmas + x) % 1
        input_circvar = circvar(y.cpu().numpy(),low=0,high=1,axis=0,)
        return input_circvar
    
    
    
if __name__ == '__main__':
    n_steps = 10
    beta1 = 1e3
    n_iters= 10000
    min_loss = 1e6
    fname = f'./cache_files/linear_entropy_alphas_s{n_steps}_{beta1}.pt'
    find_linear = True

    acc_schedule = AccuracySchedule(n_steps=n_steps,beta1=beta1)
    t = torch.range(0, n_steps-1, 1) / n_steps
    if find_linear:
        sender_alphas = acc_schedule.find_linear()
        torch.save(sender_alphas, fname)
        exit()
    else:
        if os.path.exists(fname):
            sender_alphas = torch.load(fname)
        else:
            raise FileNotFoundError
    final_acc, linear_entropy, linear_entropy_input_var, acc_var = acc_schedule.analyze_schedule(sender_alphas)
    
    diff_betas = acc_schedule.find_diff_beta()
    diff_alphas = diff_betas - torch.cat([torch.tensor([0.]),diff_betas[:-1]])
    diff_sch_final_acc, diff_sch_linear_entropy, diff_sch_input_var, diff_sch_acc_var = acc_schedule.analyze_schedule(diff_alphas)
    # compare two schedules
    fname2 = f'./cache_files/linear_entropy_alphas_s{n_steps}_{beta1/10}.pt'
    sender_alphas2 = torch.load(fname2)
    final_acc2, linear_entropy2, input_var2, acc_var2 = acc_schedule.analyze_schedule(sender_alphas2)
    # plt.scatter(t, linear_entropy_input_var.cpu(), s=1, label=fname)
    # plt.scatter(t, input_var2.cpu(), s=1, label=fname2)
    # plt.legend()
    # plt.savefig('./cache_files/input_var.png')
    
    # compare diffusion schedule and bfn schedule
    plt.figure()
    diff_input_var = acc_schedule.analyze_diff(n_steps=n_steps)
    plt.scatter(t, diff_input_var, s=1, label='diffusion')
    plt.scatter(t, linear_entropy_input_var.cpu(), s=1, label='bfn')
    plt.legend()
    plt.savefig('./cache_files/diff_vs_bfn_input_var.png')

    # compare linear schedule and previous schedule
    plt.figure(dpi=300)
    _, add_entropy, add_input_var, _ = acc_schedule.analyze_schedule()
    plt.scatter(t, add_input_var.cpu(),  label='t*beta1**t',s=1)
    plt.scatter(t, linear_entropy_input_var.cpu(),  label=f'linear entropy {beta1}',s=1)
    plt.scatter(t, input_var2.cpu(),  label=f'linear entropy {beta1/10}',s=1)
    # plt.scatter(t, diff_input_var, label='diffusion',s=1)
    plt.scatter(t, 1 - i1e(diff_betas)/i0e(diff_betas), label='diff_sch',s=1)
    plt.legend()
    plt.title(f'n_steps {n_steps} beta1 {beta1}')
    plt.savefig('./cache_files/input_variance_comparison.png')
    
    # plot the variance of log alpha
    plt.figure()
    plt.scatter(t, acc_var.cpu(), s=1, label=fname)
    # plt.scatter(t, acc_var2.cpu(), s=1, label=fname2)
    plt.title('variance of log beta')
    plt.legend()
    plt.savefig('./cache_files/log_beta_var_comparison.png')

    
    # compare entropy
    plt.figure()
    gt_linear_entropy = acc_schedule.linear_entropy((t * n_steps).to(acc_schedule.device))
    plt.plot(t, gt_linear_entropy.cpu(),label='ground truth linear entropy')
    plt.scatter(t, linear_entropy.cpu(), s=1, label='approximated linear entropy')
    plt.scatter(t, add_entropy.cpu(), s=1, label='previous schedule (t*beta**t) entropy')
    plt.legend()
    plt.savefig('./cache_files/entropy_comparison.png')
    
    
    

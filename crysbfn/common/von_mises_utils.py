from torch.distributions import Normal
from torch.distributions.von_mises import VonMises
from torch.distributions.von_mises import _log_modified_bessel_fn as log_bessel_fn
from torch.special import i0e, i1e
import torch
import numpy as np
import ray
import crysbfn
    
class VonMisesHelper:
    def __init__(self, kappa1=1e3, n_steps=10, device='cuda', cache_sampling=False, **kwargs):
        self.kappa1 = torch.tensor(kappa1, dtype=torch.float64)
        self.kappa0 = torch.tensor(0.0, dtype=torch.float64)
        self.entropy0 = VonMisesHelper.entropy_wrt_kappa(self.kappa0)
        self.entropy1 = VonMisesHelper.entropy_wrt_kappa(self.kappa1)
        
        self.n_steps = n_steps
        self.device = device
        
        self.cache_sampling = cache_sampling

    @staticmethod
    def entropy_wrt_kappa(kappa: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Compute the entropy of a von Mises distribution with respect to a given kappa.
        :param kappa: the kappa
        :return: the entropy of the von Mises distribution with respect to time
        """
        kappa  = kappa.double()
        I0e_kappa = i0e(kappa) # exp(-|kappa|)*I0(kappa)
        I1e_kappa = i1e(kappa)
        return torch.log(2 * torch.tensor(torch.pi)) + torch.log(I0e_kappa) + kappa.abs() - (kappa * I1e_kappa / I0e_kappa)
    
    def entropy_wrt_t(self, t: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Use the linear interpolation of H(1) and H(0) to compute
        the entropy of a von Mises distribution with respect to a given time.
        :param t: the time
        :return: the entropy of the von Mises distribution with respect to time
        """
        assert 0 <= t <= 1, f"t must be in [0, 1] but is {t}"
        return (1 - t) * self.entropy0 + t * self.entropy1
    
    @staticmethod
    def bayesian_update_function(m, c, y, alpha):
        '''
        Compute (m_out, c_out) = h(m, c , y, α)
        according to 
        m_out = arctan((α sin(y) + c sin(m))/( α cos(y) + c cos(m))
        c_out = sqrt(α^2 + c^2 + 2αc cos(y-m))
        :param m: the previous mean, shape (D,)
        :param c: the previous concentration, shape (D,)
        return: m_out, c_out, shape (D,)
        '''
        m_out = torch.atan2(alpha * torch.sin(y) + c * torch.sin(m), 
                            alpha * torch.cos(y) + c * torch.cos(m))
        c_out = torch.sqrt(alpha**2 + c**2 + 2 * alpha * c * torch.cos(y - m))
        return m_out, c_out
    
    @staticmethod
    def kld_von_mises(mu1, kappa1, mu2, kappa2):
        '''
        according to https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/von_mises.py#L489
        Compute the Kullback-Leibler divergence between prior and posterior
        :param mu1: the prior/known mean, shape (D,)
        :param kappa1: the prior/known concentration, shape (D,)
        :param mu2: the posterior/matching mean, shape (D,)
        :param kappa2: the posterior/matchng concentration, shape (D,)
        '''
        # first term is always zero for d = 2
        second_term = torch.log(i0e(kappa2) / i0e(kappa1)) + (kappa2 - kappa1)
        third_term = i1e(kappa1) / i0e(kappa1) * (kappa1 - kappa2 * torch.cos(mu1 - mu2)) 
        return second_term + third_term 
    
    def sample(self, loc, concentration, n_samples, 
               epsilon = 1e-6, dtype=torch.float64, 
               loop_patience=1000, device='cuda',ret_eps=False):
        '''
        :param loc: shape (batch_size, )
        :param concentration: shape (batch_size, )
        :param n_samples: the number of samples to draw
        '''
        assert loc.shape == concentration.shape
        torch_vm = VonMises(loc=loc.double(),concentration=concentration.double().clip(epsilon))
        if n_samples == 1:
            samples = torch_vm.sample().float().detach()
            samples = samples - 2. * torch.pi * torch.round(samples / (2. * torch.pi))
        else:
            samples = torch_vm.sample((n_samples,)).float().detach()
            samples = samples - 2. * torch.pi * torch.round(samples / (2. * torch.pi))
        return samples
    
        

    
    
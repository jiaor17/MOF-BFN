import math
import torch
from torch.distributions.kl import register_kl
from torch.distributions.multivariate_normal import MultivariateNormal as MN
from fisher.bingham_utils import bingham_F, bingham_dF, bingham_entropy


class Bingham(torch.distributions.Distribution):

    arg_constraints = {
        "matrix": torch.distributions.constraints.real,
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, matrix, validate_args = None):
        sym_matrix, eigvals, eigvecs = self.standardize_matrices(matrix)
        self.dtype = sym_matrix.dtype
        self.matrix = sym_matrix
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.device = sym_matrix.device
        self.dim = sym_matrix.shape[-1]

        super().__init__(self.eigvals.size(), validate_args=validate_args)

    def standardize_matrices(self, A):

        A_symm = (A + A.transpose(-2, -1)) / 2
        eigvals, eigvecs = torch.linalg.eigh(A_symm)

        eigvals_sorted, indices = torch.sort(eigvals, dim=-1, descending=True)
        eigvecs_sorted = torch.gather(eigvecs, -1, indices.unsqueeze(-2).expand(*A.shape))

        max_eig = eigvals_sorted[..., 0, None]
        eigvals_modified = eigvals_sorted - max_eig

        A_prime = torch.matmul(eigvecs_sorted, torch.matmul(torch.diag_embed(eigvals_modified), eigvecs_sorted.transpose(-2, -1)))

        return A_prime, eigvals_modified, eigvecs_sorted                



    def sample(self):
        with torch.no_grad():
            return self.rsample()

    def __angular_central_sample(self, mask = None):

        eigv = self.eigvals
        if mask is not None:
            eigv = eigv[mask]
        # prec = torch.diag_embed(-2 * eigv + 1)
        # m = MN(loc=torch.zeros_like(eigv), precision_matrix = prec).sample()
        m = torch.randn_like(eigv) / torch.sqrt(1 - 2 * eigv)
        m_uni = m / m.norm(dim=-1, keepdim=True)
        return m_uni

    def __angular_central_rel_prop(self, v, mask = None):

        eigv = self.eigvals
        if mask is not None:
            eigv = eigv[mask]
        prec = torch.diag_embed(-2 * eigv + 1)
        d = eigv.shape[-1]
        pdf = torch.einsum('bi, bij, bj -> b', v, prec, v) ** (-d / 2)
        return pdf

    def __bingham_rel_prop(self, v, mask = None):

        eigv = self.eigvals
        if mask is not None:
            eigv = eigv[mask]
        diag_eig = torch.diag_embed(eigv)
        return torch.exp(torch.einsum('bi, bij, bj -> b', v, diag_eig, v))

    def unnormalized_pdf(self, v):

        vr = torch.einsum('bji, bj -> bi', self.eigvecs, v)
        return self.__bingham_rel_prop(vr)

    def rsample(self):

        samples = torch.zeros_like(self.eigvals)
        accepted = torch.zeros(self.eigvals.shape[:-1], dtype=torch.bool, device=self.device)
        d = samples.shape[-1]

        M = math.exp(-1.5) * (4 ** (d / 2))

        # i = 0 

        while not accepted.all():
            candidate = self.__angular_central_sample(mask = ~accepted)
            acceptence_ratio = self.__bingham_rel_prop(candidate, mask = ~accepted) / (M * self.__angular_central_rel_prop(candidate, mask = ~accepted))

            accept = torch.rand_like(acceptence_ratio) < acceptence_ratio
            accepted_indices = torch.nonzero(~accepted, as_tuple=True)[0][accept]
            samples[accepted_indices] = candidate[accept]
            accepted[accepted_indices] = True
        z = torch.einsum('bij, bj -> bi', self.eigvecs, samples)

        return z.type(self.dtype)

    def entropy(self):

        return bingham_entropy(self.eigvals)

    def F(self):

        return bingham_F(self.eigvals)

    def dF(self):

        return bingham_dF(self.eigvals)

    def expected_moments(self):

        Sigma = self.dF() / self.F().unsqueeze(-1)
        first_term = torch.einsum('bij, bj, bkj -> bik', self.eigvecs, Sigma, self.eigvecs)
        second_term = (1 - Sigma.sum(dim=-1)).unsqueeze(-1).unsqueeze(-1) * torch.einsum('bi, bj -> bij', self.eigvecs[:, 0], self.eigvecs[:, 0])
        return first_term + second_term




if __name__ == '__main__':
    import torch
    import time

    # 定义一个函数来生成 vMF 样本
    def generate_bingham_samples(num_samples, dim):
        # 定义 vMF 分布
        # loc = torch.zeros((num_samples,dim),device='cuda:0')
        # concentration = torch.rand((num_samples,1),device='cuda:0').abs().clip(1e-7)*1000
        ev = torch.rand(num_samples, dim) * 10
        A = torch.diag_embed(ev)
        bhm = Bingham(matrix = A)
        # 采样
        samples = bhm.sample()
        return samples

    # 测试函数执行时间
    def test_execution_time(num_samples, dim):
        start_time = time.time()
        samples = generate_bingham_samples(num_samples, dim)
        end_time = time.time()
        return end_time - start_time

    # 设置参数
    num_samples = 1000000
    dim = 4

    # 测试采样时间
    execution_time = test_execution_time(num_samples, dim)
    print(f"Execution time for sampling {num_samples} Bingham samples with dim={dim}: {execution_time:.6f} seconds")
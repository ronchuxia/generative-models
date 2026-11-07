"""Check: the SD3 shift (no logit-normal) as a Gaussian path in the MIT 6.S184 notation, p_t(x|z) = N(alpha_t z, beta_t^2 I),
t = 0 noise, t = 1 data, t ~ U(0, 1).
Claim: alpha_t = t / (s - (s - 1) t), beta_t = 1 - alpha_t = s (1 - t) / (s - (s - 1) t).
1. beta_t with t ~ U(0, 1) has the same distribution as the SD3 noise level sigma = s u / (1 + (s - 1) u), u ~ U(0, 1).
2. d/dt alpha_t = s / (s - (s - 1) t)^2 (autograd vs formula).
3. The lab's conditional vector field (alpha' - beta'/beta alpha) z + beta'/beta x equals alpha'_t (z - eps).
"""
import torch

s, n = 3.0, 1_000_000
torch.manual_seed(0)


def alpha(t):
    return t / (s - (s - 1) * t)


def beta(t):
    return 1 - alpha(t)


# 1. Same noise-level distribution
t = torch.rand(n, dtype=torch.float64)
u = torch.rand(n, dtype=torch.float64)
sigma_sd3 = s * u / (1 + (s - 1) * u)
q = torch.linspace(0.01, 0.99, 99, dtype=torch.float64)
print("1. max quantile difference, beta_t vs SD3 sigma:", (beta(t).quantile(q) - sigma_sd3.quantile(q)).abs().max().item())
print("   alpha_t + beta_t = 1:", torch.allclose(alpha(t) + beta(t), torch.ones_like(t)))

# 2. Derivative of alpha_t
tg = torch.linspace(0, 1, 11, dtype=torch.float64, requires_grad=True)
(da,) = torch.autograd.grad(alpha(tg).sum(), tg)
print("2. max |autograd - formula| for d/dt alpha_t:", (da - s / (s - (s - 1) * tg.detach()) ** 2).abs().max().item())
print("   d/dt alpha_t at t = 0, 0.5, 1:", [round(v, 4) for v in da[[0, 5, 10]].tolist()])

# 3. Conditional vector field
tt = torch.rand(1000, 1, dtype=torch.float64, requires_grad=True)
z, eps = torch.randn(1000, 2, dtype=torch.float64), torch.randn(1000, 2, dtype=torch.float64)
(a_dot,) = torch.autograd.grad(alpha(tt).sum(), tt)
b_dot = -a_dot
tt = tt.detach()
x = alpha(tt) * z + beta(tt) * eps
u_lab = (a_dot - b_dot / beta(tt) * alpha(tt)) * z + b_dot / beta(tt) * x
print("3. max |lab field - alpha'_t (z - eps)|:", (u_lab - a_dot * (z - eps)).abs().max().item())

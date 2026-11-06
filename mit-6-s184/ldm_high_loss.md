# Latent DiT: Why the Loss Is High but the Digits Look Good

LDM = Latent Diffusion Model. DiT = Diffusion Transformer. VAE = Variational Autoencoder. CFG = Classifier-Free Guidance. ODE = Ordinary Differential Equation.

## Run and data

| Item | Value |
|---|---|
| LDM run | `runs/autumn-durian-cf0ae748`, batch size 64, 10000 steps |
| VAE run | `runs/white-watermelon-e77710b0`, beta = 10, checkpoint step 4750 |
| DiT checkpoint | step 9500 |
| Training log | loss 4242 at step 0, loss 3999 at step 9999 |
| Notebook | [MIT 6.S184 lab_three.ipynb](MIT%206.S184%20lab_three.ipynb), cells 13, 16, 28, 46, 63, 69, 77, 78 |

The LDM used the VAE from step 4999. That VAE was only in memory, thus I used the checkpoint from step 4750. My measured DiT loss (4003 to 4008) agrees with the training log (3999). Thus this difference has a very small effect.

All values come from these scripts (CPU):

- [ldm_loss_floor.py](../agent-workspace/ldm_loss_floor.py): batch size test, latent statistics, lower and upper bounds.
- [ldm_output_rank.py](../agent-workspace/ldm_output_rank.py): rank of the DiT output, loss of the linear formula.
- [ldm_noise_directions.py](../agent-workspace/ldm_noise_directions.py): eigenvalues, the 1888 fixed directions, loss in each part.

The samples at step 9500 show clear digits:

![LDM samples at step 9500](runs/autumn-durian-cf0ae748/step_%20%209500_output.png)

# Summary

1. The batch size does not change the mean loss.
2. The latent is 97% encoder noise. No model can predict this noise. Thus the loss has a minimum between **3172 and 3207**.
3. The DiT stays at approximately **4000**, not at the minimum. The cause is `final_dim = 10` in the `Depatchifier`. The DiT output can change in only **160** of the 2048 latent directions.
4. In the other **1888** directions, the DiT output is a constant near 0. There, the loss is the same as for $u = 0$.
5. These 1888 directions are almost all noise directions. In these directions, the start noise already has approximately the correct distribution. With $u \approx 0$, the ODE keeps the start noise. Thus the digits look good.

# 1. The batch size has no effect on the mean loss

The loss is divided by `batch_size` (cell 77):

```python
loss = torch.sum((u_theta - u_ref)**2) / batch_size
```

Thus the loss is the sum over the $128 \times 4 \times 4 = 2048$ latent elements, with the mean over the samples.

| Batch size | Mean loss | Std over batches | Batches |
|---|---|---|---|
| 64 | 4005 | 17 | 64 |
| 256 | 4004 | 7 | 16 |

A smaller batch only makes the loss more noisy.

# 2. The latent is mostly encoder noise

## The code

The LDM trainer makes each latent with the VAE encoder (cell 77):

```python
z_mean, z_logvar = self.vae.encode(z)
z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
```

The encoder (cell 63) gives a `z_mean` for each image. But `logvar` is one scalar for all elements and all images:

```python
self.logvar = nn.Parameter(torch.zeros(()))
```

## The model of the latent

Each latent element is

$$z_j = z_{\text{mean},j}(\text{image}) + \sigma\, n_j, \qquad \sigma = e^{\frac{1}{2} z_{\text{logvar}}}, \qquad n_j \sim \mathcal{N}(0, 1).$$

The noise $n$ does not depend on the image. The law of total variance gives

$$\operatorname{Var}(z_j) = \operatorname{Var}_{\text{images}}(z_{\text{mean},j}) + \sigma^2.$$

## The values

In the checkpoint, $z_{\text{logvar}} = -0.0283$. Thus $\sigma^2 = e^{-0.0283} = 0.9721$ and $\sigma = 0.986$.

I encoded 4096 MNIST images. Then I calculated the variance of each of the 2048 elements over the images:

```python
var, mu = zm.var(0), zm.mean(0)   # zm: (4096, 2048)
```

| $\operatorname{Var}(z_{\text{mean},j})$ over the 2048 elements | Value |
|---|---|
| Mean | 0.0293 |
| 10% / 50% / 90% / 99% quantile | 0.0025 / 0.0125 / 0.0725 / 0.257 |
| Maximum | 0.414 |
| Sum | 59.9 |
| Elements with value > 0.01 | 1165 |

In every element, the image part is smaller than the noise part (0.414 < 0.972). Over all elements, the image part is 59.9. The noise part is $2048 \times 0.972 = 1991$.

The mean of the latent is near 0. The sum of $\mathbb{E}[z_{\text{mean},j}]^2$ over all elements is 0.549. The largest $|\mathbb{E}[z_{\text{mean},j}]|$ is 0.074.

The VAE loss (cell 69) agrees with these results. With $\beta = 10$, the term `0.5 * self.beta * torch.sum(z_mean**2) / b` pushes `z_mean` to 0. The term `exp(z_logvar) - z_logvar - 1` has its minimum at $z_{\text{logvar}} = 0$, that is, $\sigma^2 = 1$.

# 3. The minimum loss

## Step 1: the target

Cell 13 with `LinearAlpha` ($\alpha_t = t$) and `LinearBeta` ($\beta_t = 1 - t$):

```python
return alpha_t * z + beta_t * torch.randn_like(z)                                   # sample_conditional_path
return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x      # conditional_vector_field
```

Thus $x_t = t\,z + (1-t)\,\varepsilon$. With $\dot\alpha_t = 1$ and $\dot\beta_t = -1$:

$$u_{\text{ref}} = \Big(1 + \frac{t}{1-t}\Big) z - \frac{x_t}{1-t} = \frac{z - x_t}{1-t} = z - \varepsilon.$$

## Step 2: the minimum of a squared error

Let $w = z - \varepsilon$. For any model $u(x_t, t, y)$:

$$\mathbb{E}\|u - w\|^2 = \mathbb{E}\big\|u - \mathbb{E}[w \mid x_t, t, y]\big\|^2 + \mathbb{E}\big\|w - \mathbb{E}[w \mid x_t, t, y]\big\|^2.$$

The cross term is 0, because $\mathbb{E}\big[w - \mathbb{E}[w \mid \cdot] \,\big|\, \cdot\big] = 0$. The second term does not depend on the model. Thus the minimum loss is

$$\mathbb{E}\Big[\operatorname{tr} \operatorname{Var}(w \mid x_t, t, y)\Big].$$

## Step 3: one Gaussian element

Let $z \sim \mathcal{N}(m, s^2)$ and $\varepsilon \sim \mathcal{N}(0, 1)$, independent. Then $z$ and $x_t$ are jointly Gaussian. The Gaussian conditioning formula gives

$$\operatorname{Var}(z \mid x_t) = s^2 - \frac{\operatorname{Cov}(z, x_t)^2}{\operatorname{Var}(x_t)} = s^2 - \frac{t^2 s^4}{t^2 s^2 + (1-t)^2} = \frac{s^2 (1-t)^2}{t^2 s^2 + (1-t)^2}.$$

$w = (z - x_t)/(1-t)$, and $x_t$ is known. Thus

$$\operatorname{Var}(w \mid x_t) = \frac{\operatorname{Var}(z \mid x_t)}{(1-t)^2} = \frac{s^2}{t^2 s^2 + (1-t)^2}.$$

The mean $m$ has no effect.

## Step 4: the average over t

Cell 77 uses `t = torch.rand(...)`, thus $t \sim \mathcal{U}(0, 1)$. The denominator is $a t^2 - 2t + 1$ with $a = s^2 + 1$. An antiderivative is

$$\frac{d}{dt}\Big[\frac{1}{s}\arctan\frac{a t - 1}{s}\Big] = \frac{a}{s^2 + (a t - 1)^2} = \frac{1}{a t^2 - 2t + 1},$$

because $s^2 + (a t - 1)^2 = a\,(a t^2 - 2t + 1)$. At $t = 1$ the value is $\arctan s$. At $t = 0$ the value is $-\arctan(1/s)$. For $s > 0$, $\arctan s + \arctan(1/s) = \pi/2$. Thus

$$\int_0^1 \frac{s^2}{t^2 s^2 + (1-t)^2}\,dt = s^2 \cdot \frac{1}{s} \cdot \frac{\pi}{2} = \frac{\pi}{2}\, s.$$

Numerical check (trapezoid rule, 200001 points):

| $s$ | Integral | $\frac{\pi}{2} s$ |
|---|---|---|
| 0.5 | 0.7854 | 0.7854 |
| 1.0 | 1.5708 | 1.5708 |
| 1.28 | 2.0106 | 2.0106 |

## Step 5: lower bound for this latent

More information cannot make the minimum larger, because

$$\mathbb{E}\big[\operatorname{Var}(w \mid A, B)\big] \le \mathbb{E}\big[\operatorname{Var}(w \mid A)\big].$$

Thus give the model the image too. Then the model also knows $z_{\text{mean}}$ and $y$. Only $n$ and $\varepsilon$ are unknown. For each element:

$$x_t - t\, z_{\text{mean}} = t\,\sigma n + (1-t)\,\varepsilon, \qquad w - z_{\text{mean}} = \sigma n - \varepsilon.$$

This is Step 3 with $s = \sigma$ and $m = 0$. $n$ and $\varepsilon$ are independent over the 2048 elements. Thus the minimum with the extra information is

$$2048 \cdot \frac{\pi}{2} \cdot 0.986 = 3172.$$

The real model has less information. Thus its minimum is at least **3172**.

# 4. The upper bound

## Step 1: affine models

Use only affine models $u = a(t) + B(t)\, x_t$. They are a subset of all models, and they do not use $y$. Thus their best loss is at least the true minimum.

## Step 2: only the mean and the covariance are important

The best affine model and its error depend only on the mean and the covariance of $(w, x_t)$:

$$\text{error}(t) = \operatorname{tr}\Big(\Sigma_{ww} - \Sigma_{wx}\,\Sigma_{xx}^{-1}\,\Sigma_{xw}\Big).$$

Thus a Gaussian $z$ with the same mean $\mu$ and covariance $C$ gives the same error.

## Step 3: for a Gaussian, affine is the best

For a Gaussian $z$, $\mathbb{E}[w \mid x_t]$ is affine. Thus for a Gaussian, the best affine model is also the best model of all. Its error is the Gaussian minimum.

## Step 4: the Gaussian minimum

Write $C = V \Lambda V^\top$ and rotate all vectors with $V^\top$. Then $V^\top \varepsilon \sim \mathcal{N}(0, I)$ again, and the squared norm does not change. In the rotated coordinates, the elements are independent with variances $\Lambda_{ii}$. Step 4 of section 3 applies to each element:

$$\text{minimum} = \sum_i \frac{\pi}{2}\sqrt{\Lambda_{ii}} = \frac{\pi}{2}\operatorname{tr}\big(C^{1/2}\big).$$

## Step 5: the value for this latent

From section 2, $C = \operatorname{Cov}(z_{\text{mean}}) + \sigma^2 I$:

```python
cov = torch.cov(z_mean.T.double()) + sigma**2 * torch.eye(k, dtype=torch.float64)
eig = torch.linalg.eigvalsh(cov).clamp(min=0)
```

The result is **3207**. $C$ is an estimate from 4096 images, thus this value has a small estimation error.

## Check with a diagonal model

If $B(t)$ is diagonal, each element uses only its own $x_t$. The error is then $\frac{\pi}{2}\sum_j \sqrt{\operatorname{Var}(z_j)} = 3218$. This is the "linear formula" in the tables below:

```python
u_lin = mu + (tt * s2 - (1 - tt)) / (tt ** 2 * s2 + (1 - tt) ** 2) * (xx - tt * mu)
```

Its measured loss was 3227 and 3245 in two runs with different samples. The differences from 3218 are in the range of the sampling noise.

Thus the minimum loss is between **3172 and 3207**.

# 5. Noise directions

## Definition

A unit vector $v$ is a noise direction if $\operatorname{Var}(v^\top z_{\text{mean}}) = v^\top \operatorname{Cov}(z_{\text{mean}})\, v$ is small compared with $\sigma^2$.

In any direction $v$:

$$v^\top z = v^\top z_{\text{mean}} + \sigma\, v^\top n.$$

$n \sim \mathcal{N}(0, I)$ is isotropic. Thus $v^\top n \sim \mathcal{N}(0, 1)$ exactly. Thus $v^\top z$ is $\mathcal{N}(0, 0.972)$ plus an independent part with mean $v^\top \mu$ and variance $v^\top C v$.

## The eigenvalues of Cov(z_mean)

| | Value |
|---|---|
| Top 5 eigenvalues | 4.11, 3.83, 3.63, 3.45, 3.33 |
| Number of eigenvalues > 1 / > 0.1 / > 0.01 | 22 / 45 / 70 |
| Part of $\operatorname{tr}\operatorname{Cov}(z_{\text{mean}}) = 59.9$ in the top 10 / 50 / 160 directions | 52% / 99% / 100% |

Thus the image information is in approximately 50 directions. These are eigen directions, not single elements. In the other approximately 1978 directions, $v^\top C v < 0.01$. There

$$v^\top z \sim \mathcal{N}\big(v^\top \mu,\; 0.972 + (\text{less than } 0.01)\big).$$

The mean is small: $\|\mu\| = \sqrt{0.549} = 0.74$ over all 2048 dimensions.

# 6. The DiT output has rank 160

## The code

Cell 46, with `patch_size = 1` and `final_dim = 10` from cell 78:

```python
self.linear = nn.Linear(dim, final_dim*patch_size*patch_size)   # 256 -> 10 values per token
self.conv = nn.Conv2d(final_dim, c_out, 3, 1, 1)                # 10 -> 128 channels, linear
```

## The problem

All 2048 outputs ($128 \times 4 \times 4$) are a linear function of 160 values ($10 \times 4 \times 4$), plus a bias:

$$u = W v + b, \qquad W \in \mathbb{R}^{2048 \times 160}.$$

Here $v$ is the output of `self.linear`, $W$ is the 3×3 conv without its bias, and $b$ is the conv bias. I made $W$ with 160 unit vectors through the conv:

```python
W = F.conv2d(torch.eye(160).view(160, 10, 4, 4), conv.weight, None, 1, 1).flatten(1).T
```

Measured results:

- $\operatorname{rank}(W) = 160$.
- Singular values of the DiT output over 2048 samples: number 160 is 20.2, number 161 is $4.65 \times 10^{-6}$.

This setting comes from the pixel DiT. There, `c = 1` is less than `final_dim = 10`, thus the output is not limited. In the latent DiT, `c = 128` is much more than 10.

# 7. In the 1888 other directions, u is a constant near 0

## The output is constant (because of the architecture)

Let $Q_\perp$ be an orthonormal basis of the 1888 directions orthogonal to $\operatorname{range}(W)$. Then for all inputs:

$$Q_\perp^\top u = Q_\perp^\top W v + Q_\perp^\top b = Q_\perp^\top b.$$

Measured on 1024 samples:

- Maximum std of $Q_\perp^\top u$ over the samples: $8.6 \times 10^{-8}$ (float32 round-off).
- Maximum difference between the mean and $Q_\perp^\top b$: $8.7 \times 10^{-9}$.

## The constant is near 0 (because of the training)

In these directions, the loss is $\mathbb{E}\|Q_\perp^\top b - Q_\perp^\top w\|^2$. Its minimum is at

$$Q_\perp^\top b = Q_\perp^\top\, \mathbb{E}[w] = Q_\perp^\top\, \mathbb{E}[z_{\text{mean}}],$$

because $\mathbb{E}[\varepsilon] = 0$. From section 2, $\mathbb{E}[z_{\text{mean}}]$ is near 0.

| | Norm |
|---|---|
| $Q_\perp^\top b$ (the DiT output) | 0.46 (RMS 0.011 per direction) |
| $Q_\perp^\top\, \mathbb{E}[z_{\text{mean}}]$ (the optimal constant) | 0.49 |
| The target $w$ in these directions, approximately $\sqrt{1888 \times 1.97}$ | 61 |

Thus $u \approx 0$ in these directions.

## These directions are almost all noise directions

| | Image variance $\operatorname{tr}\operatorname{Cov}(z_{\text{mean}})$ |
|---|---|
| In $\operatorname{range}(W)$, 160 directions | 56.2 |
| In the 1888 other directions | 3.7 |
| In a random 160-dimensional subspace (average), $160/2048 \times 59.9$ | 4.7 |

Thus the training moved $\operatorname{range}(W)$ onto the image directions. In the 1888 other directions, $\operatorname{Var}(z_{\text{mean}})$ is 0.0020 per direction on average (maximum 0.323). Thus there

$$z \approx \mathcal{N}(0,\; 0.974\, I).$$

# 8. The loss in each part

Measured on 1024 samples:

| Model | $\operatorname{range}(W)$, 160 directions | Other 1888 directions | Total |
|---|---|---|---|
| $u = 0$ | 375 | 3728 | 4103 |
| DiT, step 9500 | **280** | **3728** | 4008 |
| Linear formula (no trained parameters) | 295 | 2949 | 3245 |

- In $\operatorname{range}(W)$, the DiT is better than the linear formula (280 < 295). The DiT can use the label and nonlinear functions. The linear formula cannot.
- In the 1888 other directions, the DiT has exactly the loss of $u = 0$ (3728).
- All of the gap of 779 is in the 1888 other directions.

# 9. Why u = 0 is not the best field in a noise direction

In a noise direction with variance $s^2$ and mean 0, the best field is $\mathbb{E}[w \mid x_t]$ from Step 3 of section 3:

$$u^*(x, t) = \frac{t s^2 - (1-t)}{t^2 s^2 + (1-t)^2}\, x.$$

For $s = 1$:

| $t$ | $u^*(x, t)$ |
|---|---|
| 0 | $-x$ |
| 0.5 | 0 |
| 1 | $+x$ |

This field follows the path

$$p_t = \mathcal{N}\big(0,\; t^2 s^2 + (1-t)^2\big).$$

For $s = 1$, the variance first decreases to 0.5 at $t = 0.5$. Then it increases to 1 at $t = 1$.

The loss measures the distance to this field, not only the final distribution. $u = 0$ is far from $u^*$, except near $t = 0.5$. Per direction:

| Model | Loss per direction |
|---|---|
| $u = 0$ | $3728 / 1888 = 1.97$ |
| Linear formula | $2949 / 1888 = 1.56 \approx \frac{\pi}{2} \cdot 0.987$ |

# 10. Why the digits look good

## The sampler

The Euler step (cell 16):

```python
return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h
```

With CFG (cell 28), the drift is

$$(1 - w)\, u(x, t, \varnothing) + w\, u(x, t, y).$$

In the 1888 directions, both terms are $Q_\perp^\top b$. Thus the drift is $Q_\perp^\top b$ for all guidance scales $w$. `visualize_samples` (cell 77) uses `ts = linspace(0, 0.999, 100)`, thus $\sum h = 0.999$. The start noise is $x_0 \sim \mathcal{N}(0, I)$ (`IsotropicGaussian` with `std = 1.0`). Thus

$$Q_\perp^\top x_1 = Q_\perp^\top x_0 + 0.999\, Q_\perp^\top b \approx Q_\perp^\top x_0 \sim \mathcal{N}(0, I).$$

## The comparison with the target

| Distribution in the 1888 directions | Std per direction |
|---|---|
| Samples, $\mathcal{N}(0, I)$ | 1.000 |
| Target, $\mathcal{N}(0, 0.974\, I)$ (section 7) | 0.987 |

The std is wrong by approximately 1.3%.

## The result

- $u = 0$ and $u^*$ both move $\mathcal{N}(0, I)$ to approximately the correct distribution. They use different paths.
- The loss of $u = 0$ is high, because the loss measures the path. The final samples are approximately correct.
- The small image variance in the 1888 directions (3.7 of 59.9) is replaced by noise.
- The 160 directions that the DiT can change hold 94% of the image variance (56.2 of 59.9). This agrees with the good digits.

# How to remove the limit

If `final_dim` is 128 or more in cell 78, then `self.linear` gives at least 128 values per token, and the conv can give an output with full rank 2048. Then the loss can go below 4000, but not below 3172 (section 3). I did not change the notebook.

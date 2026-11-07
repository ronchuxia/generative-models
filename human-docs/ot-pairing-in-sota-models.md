# Do State-of-the-Art Models Use OT Pairs in Training?

No. All five large image and video models that we checked use independent noise. Each training sample gets its own new Gaussian noise. None of them uses optimal transport (OT) pairs.

## 1. Coupling in Large Models

| Model | Coupling | Quote from the paper |
|---|---|---|
| Stable Diffusion 3 (Esser et al., 2024) | Independent | Noise "ϵ∼𝒩(0,I)". The paper cites Pooladian et al. (minibatch OT), but it does not use their method. |
| FLUX.1 Kontext (Black Forest Labs, 2025) | Independent | "noise ε∼𝒩(0,1)" |
| Movie Gen (Meta, 2024) | Independent | "we sample a 'noise' sample 𝐗₀∼𝒩(0,1)". The "optimal transport path" in this paper is the straight path. |
| HunyuanVideo (Tencent, 2024) | Independent | "initialize a noise 𝐱₀∼𝒩(𝟎,𝐈)" |
| Wan (Alibaba, 2025) | Independent | "a random noise x₀∼𝒩(0,I)" |

## 2. Why Large Models Do Not Use OT Pairs

### 2.1 The gain is mostly at very few sampling steps

Tong et al. (2024, Section 5.3, CIFAR-10) say: "After a long training time, all methods achieve similar performance at a high number of function evaluations", but OT pairs perform "significantly better with a small number of function evaluations".

### 2.2 With a condition, plain OT pairs make results worse

Cheng and Schwing (2025) say that OT "disregards the conditions when computing the coupling". Thus the noise in training is "skewed by the condition", but the noise at inference is not.

Their results on ImageNet 256×256 (latent space, caption-conditioned). The values are FID (Fréchet Inception Distance). Lower is better.

| Euler steps | 2 | 10 | 100 |
|---|---|---|---|
| Independent noise | 202.5 | 10.0 | 3.48 |
| OT pairs | 193.4 | 15.6 | 5.36 |
| C2OT (their fix) | 200.8 | 10.0 | 3.32 |

Most current models take text as a condition. Thus this problem applies to them.

### 2.3 The cost increases fast with the OT batch size

We measured the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on the 2D moons and circles data:

| OT batch size | 250 | 500 | 1000 |
|---|---|---|---|
| Time for one batch | 0.008 s | 0.049 s | 0.30 s |

## 3. Limits

- We checked only models that have a public paper. Closed models do not publish their training details.
- FLUX.1 [dev] has no paper. Thus we used the FLUX.1 Kontext paper.

## 4. Relation to the Toy Test

This agrees with the moons-to-circles test (`human-docs/ot-flow-matching-circles.md`). With random pairs, cosine decay, a wider model and 1000 Euler steps gave rings close to the best possible value. The main gain from OT pairs was that 100 steps gave the same result as 1000 steps.

## References

- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3), 2024. https://arxiv.org/abs/2403.03206
- Black Forest Labs, *FLUX.1 Kontext*, 2025. https://arxiv.org/abs/2506.15742
- Polyak et al., *Movie Gen: A Cast of Media Foundation Models*, 2024. https://arxiv.org/abs/2410.13720
- Kong et al., *HunyuanVideo: A Systematic Framework for Large Video Generative Models*, 2024. https://arxiv.org/abs/2412.03603
- Wan Team, *Wan: Open and Advanced Large-Scale Video Generative Models*, 2025. https://arxiv.org/abs/2503.20314
- Tong et al., *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*, TMLR 2024. https://arxiv.org/abs/2302.00482
- Cheng and Schwing, *The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation*, 2025. https://arxiv.org/abs/2503.10636
- Pooladian et al., *Multisample Flow Matching: Straightening Flows with Minibatch Couplings*, 2023. https://arxiv.org/abs/2304.14772

# The Sine-Cosine Conditional Path in FlowMatchingTrainer

This document tells if the conditional path in `notebooks/flow-matching-with-scaled-sigma.ipynb` is curved. It also tells if `FlowMatchingTrainer` trains the model on this path. The script `agent-workspace/check_sincos_path.py` gives all the numbers.

## 1. Setup

- `FlowMatchingTrainer` (`src/generative_models/trainer/flow_matching.py`) makes the point $x_t = \alpha(t)\, z + \beta(t)\, x$.
- $x$ is a source point (moons). $z$ is a target point (circles). The dataset pairs them at random.
- The notebook uses $\alpha(t) = \sin(\pi t / 2)$ and $\beta(t) = \cos(\pi t / 2)$.
- The regression target in the trainer is $z - x$.

Write $\theta = \pi t / 2$. Then:

$$
x_t = \cos\theta\, x + \sin\theta\, z, \qquad \theta \in [0, \pi/2]
$$

## 2. The Path Is Curved

For one pair $(x, z)$, the path is an arc of an ellipse. The center of the ellipse is the origin. The arc starts at $x$ ($t = 0$) and stops at $z$ ($t = 1$).

The general rule: the path is straight only if all points $(\alpha(t), \beta(t))$ are on one straight line. For example, $\alpha(t) = t$ and $\beta(t) = 1 - t$ give a straight path. The sine-cosine points are on the unit circle. Thus the path is curved.

At $t = 0.5$:

$$
x_{0.5} = \frac{x + z}{\sqrt{2}} = \sqrt{2} \cdot \frac{x + z}{2}
$$

Thus the middle point of the path is $\sqrt{2}$ times the middle point of the straight segment. The path goes away from the origin.

There is one exception. If $x$ and $z$ are on one line through the origin, the path stays on that line. But it can go past $z$ and then come back. Example: $x = (1, 0)$, $z = (2, 0)$. The path goes to $\sqrt{5} \approx 2.236$, and then back to $2$.

Measurement on 10000 moons-circles pairs:

| Value | Mean |
|---|---|
| Length of the straight segment from $x$ to $z$ | 1.318 |
| Largest distance from the path to the straight segment | 0.216 |

## 3. The Trainer Target Is Not the Velocity of This Path

The velocity of the path is its time derivative:

$$
\frac{d x_t}{dt} = \alpha'(t)\, z + \beta'(t)\, x = \frac{\pi}{2} \left( \cos\theta\, z - \sin\theta\, x \right)
$$

The trainer uses $z - x$. This is the velocity of the straight path only.

Measurement on 10000 pairs with random $t$:

| Value | Result |
|---|---|
| Largest difference between the finite-difference velocity and the formula above | 0.0015 (float32 rounding) |
| Mean distance between the finite-difference velocity and $z - x$ | 0.570 |

Thus the trainer takes $x_t$ from the curved path, but it uses the straight-path velocity as the target. The model does not learn the flow along the curved path.

Theorem 3.1 of Tong et al. (2024) shows that the marginal field moves the source distribution to the target distribution. But the theorem needs a conditional target that is the true velocity of the conditional path. The target $z - x$ does not satisfy this condition.

## 4. Example: the Flow Does Not Get to the Target

This example has a closed-form best model.

- 1D data: $x \sim \mathcal{N}(0, 1)$ and $z \sim \mathcal{N}(0, 4)$, independent.
- All variables are Gaussian. Thus the best model at time $t$ is linear: $u(y, t) = k(t)\, y$.
- With $\sigma_x^2 = 1$ and $\sigma_z^2 = 4$:

$$
k_{\text{trainer}}(t) = \frac{\sin\theta\, \sigma_z^2 - \cos\theta\, \sigma_x^2}{\sin^2\theta\, \sigma_z^2 + \cos^2\theta\, \sigma_x^2}
\qquad
k_{\text{true}}(t) = \frac{\pi}{2} \cdot \frac{\sin\theta \cos\theta\, (\sigma_z^2 - \sigma_x^2)}{\sin^2\theta\, \sigma_z^2 + \cos^2\theta\, \sigma_x^2}
$$

- The ODE (ordinary differential equation) $dy/dt = k(t)\, y$ gives a final standard deviation of $\exp\left(\int_0^1 k(t)\, dt\right)$.

| Target | Final standard deviation |
|---|---|
| Correct value | 2.000 |
| $z - x$ (the trainer now) | 1.792 |
| $\alpha'(t)\, z + \beta'(t)\, x$ | 2.000 |

A Monte Carlo test with 1000000 samples gives $k_{\text{trainer}}(0.3) = 0.5713$. The formula gives 0.5716.

## 5. Conclusion

- The sine-cosine path is curved.
- The trainer does not train the model on this path, because the target $z - x$ is the velocity of the straight path.
- The correct target for this path is $\alpha'(t)\, z + \beta'(t)\, x$.

## References

- Tong, A., et al. (2024). Improving and generalizing flow-based generative models with minibatch optimal transport. TMLR.

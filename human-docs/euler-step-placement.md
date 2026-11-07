# Where to Use Small Euler Steps

This document tells where small Euler steps help most in the moons-to-circles flow model. It also tells how this relates to the time schedules of SD3 and Movie Gen.

## 1. The Question

With 100 uniform Euler steps, the best model has a radius W1 of 0.0055. With 1000 uniform steps, it has 0.0017. Thus the solver error is large with 100 steps. Which part of the path causes this error?

## 2. A Fixed Number of Steps Mixes Two Effects

A grid with a fixed number of steps cannot make all steps smaller. If it makes the steps at one part of the path smaller, it must make the steps at other parts larger.

Example: the grid $t_i = 1 - (1 - i/100)^2$ has small steps near $t = 1$. But its first step is 0.020, which is 2 times the uniform step (0.010).

Thus a worse result with this grid does not show that small steps at the end hurt. It can come from the larger steps at the other parts.

To separate the two effects, the test also adds extra small steps at one part, without a change to the other steps.

## 3. Setup

- Model: the best model from `human-docs/ot-flow-matching-circles.md` (width 256, 50000 training steps, cosine learning-rate decay, random pairs).
- Path: $t = 0$ is the moons (source). $t = 1$ is the circles (target).
- Metrics (lower is better), with 20000 points:
  - **Radius W1:** how sharp the two rings are.
  - **Sliced W1:** the full 2D distribution.
  - Best possible values: radius W1 0.0011, sliced W1 0.0018.

## 4. Results

| Grid | Steps | First step | Last step | Radius W1 | Sliced W1 |
|---|---|---|---|---|---|
| Uniform | 100 | 0.010 | 0.010 | 0.0055 | 0.0050 |
| Small steps at end, larger at start ($t_i = 1 - (1 - i/100)^2$) | 100 | 0.020 | 0.0001 | 0.0069 | 0.0059 |
| Small steps at start, larger at end ($t_i = (i/100)^2$) | 100 | 0.0001 | 0.020 | 0.0070 | 0.0060 |
| Uniform, last 10 steps split into 10 each | 190 | 0.010 | 0.001 | 0.0050 | 0.0048 |
| Uniform, first 10 steps split into 10 each | 190 | 0.001 | 0.010 | 0.0048 | 0.0054 |
| Uniform, middle 80 steps split into 10 each | 820 | 0.010 | 0.010 | 0.0032 | 0.0036 |
| Uniform | 1000 | 0.001 | 0.001 | 0.0017 | 0.0036 |

## 5. Findings

1. **Extra small steps at the end or at the start do not hurt.** They help a little (radius W1 0.0055 → 0.0050 and 0.0048).
2. **With a fixed number of steps, a move of small steps to one end hurts.** Both "small at end" and "small at start" are worse than uniform (0.0069 and 0.0070). The steps in the middle and at the other end become larger.
3. **The middle of the path has the largest solver error.** Smaller steps in the middle 80% give most of the gain: radius W1 0.0055 → 0.0032. The sliced W1 (0.0036) is the same as with 1000 steps.
4. **The reason is the curved paths.** With random pairs, the points first move in to the center and then move out to the rings. With the base model (width 64), the mean radius goes from 1.05 to its minimum (about 0.65) between $t = 0.5$ and $t = 0.6$, and then to 0.91 (`human-docs/flow-matching-without-t.md`, test script `agent-workspace/check_ot_fm_no_t.py`). The paths turn most in the middle. The local error of one Euler step of size $h$ is $\frac{h^2}{2} x''(\xi)$ for some $\xi$ in the step. Thus the error is largest where the velocity changes fastest, that is, where the paths turn.

## 6. The Rule

With a fixed number of steps, put the small steps where the solver error is largest.

- In this test, the largest error is in the middle of the path.
- SD3 and Movie Gen put the small steps at the noise end of their image and video paths (see `human-docs/sigma-shift.md`). Movie Gen says: "the first inference steps are pivotal in setting up the scene and motion of the video". Its 50-step schedule (25 linear steps, then 25 quadratic steps) "emulat[es] N=250 linear steps".

This test cannot confirm or reject the SD3 and Movie Gen choice. Their source is noise, but the source in this test is moons. Also, SD3 uses the shift in training, not only in sampling.

## 7. Limits

- One model and one seed.
- One 2D dataset. The best place for small steps depends on where the paths curve. This changes with the data and the pairs.

## 8. Files

- `agent-workspace/eval_time_grids.py`: the table in Section 4. Usage: `python eval_time_grids.py OUT_DIR big_long_cosine 256`.
- `agent-workspace/diagnose_ot_fm_circles.py`: trains the model and saves `big_long_cosine_model.pt` in `OUT_DIR`.

## References

- Polyak et al., *Movie Gen: A Cast of Media Foundation Models*, 2024. https://arxiv.org/abs/2410.13720
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3), 2024. https://arxiv.org/abs/2403.03206

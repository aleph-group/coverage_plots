# coverage_plots

A small Python package for computing **empirical coverage curves** of
posterior samples produced by stochastic reconstruction algorithms for
inverse problems (e.g. PnP-ULA, DPS, diffusion posterior samplers, MCMC).

## What is a coverage plot?

Given a stochastic posterior sampler that, for an observation $y$, returns
samples $\hat{x}_1, \dots, \hat{x}_S$, approximating $p(x \mid y)$, we want to
check whether the *spread* of those samples is calibrated against the true
$x$.

The construction proceeds in three steps.

**1. Posterior mean.** From the posterior samples we form the Monte-Carlo
estimate of the posterior mean,

$$
\bar{x} = \frac{1}{S}\sum_{s=1}^{S} \hat{x}_s \approx \mathbb{E}[x \mid y].
$$

**2. Approximate credible balls.** For a confidence level $\alpha \in [0, 1]$,
let $r_\alpha$ be the empirical $\alpha$-quantile of the sample-to-mean
distances,

$$
r_\alpha = Q_\alpha\left(\lbrace \lVert \hat{x}_s - \bar{x} \rVert \rbrace_{s=1}^{S}\right),
$$

and define the ball

$$
B_\alpha(\bar{x}) = \lbrace u : \lVert u - \bar{x} \rVert \le r_\alpha \rbrace.
$$

By construction $B_\alpha(\bar{x})$ contains a fraction $\alpha$ of the
posterior samples, so it approximates a credible region of posterior mass
$\alpha$:

$$
\Pr\left[ x \in B_\alpha(\bar{x}) \mid y \right] \approx \alpha.
$$

**3. Compare against the reference.** A well-calibrated posterior should
actually contain the true $x$ inside $B_\alpha(\bar{x})$ with probability
$\alpha$. We test this by repeating the construction over many $(x, y)$
pairs and recording the empirical *coverage*,

$$
\widehat{\mathrm{cov}}(\alpha) = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\left[ x_i \in B_\alpha(\bar{x}_i) \right].
$$

Plotting $\widehat{\mathrm{cov}}(\alpha)$ against $\alpha$ gives the
*coverage curve*. A perfectly calibrated sampler follows the diagonal
$\widehat{\mathrm{cov}}(\alpha) = \alpha$:

- **Curve below the diagonal** → posterior is *over-confident* (the credible
  balls are too small; the true $x$ falls outside them more often than the
  sampler's credibility budget would allow).
- **Curve above the diagonal** → posterior is *under-confident* (the
  credible balls are too large; the true $x$ is contained more often than
  expected).

Coverage can be computed in pixel space or, optionally, in the latent space
of an embedding $\phi$ (e.g. a VAE encoder), in which case the radii and
the inclusion test are formed from $\lVert \phi(\cdot) - \phi(\bar{x}) \rVert$.

## What's in the package

- [`coverage_plots.Coverage`](coverage_plots/coverage.py) — the public class.
  Accumulates the statistic over many reference batches via
  `update(x_samples, x)` and returns the curve via `compute()`.
  Pixel-space coverage is always computed; an extra embedding-space curve is
  computed in parallel when an `embedding` callable is supplied.
- [`coverage_plots.load_mnist_vae_embedding`](coverage_plots/vae.py) —
  factory that loads a small pre-trained MNIST VAE encoder and returns it as
  an embedding callable suitable for `Coverage(embedding=...)`.
- [example/example_coverage_uniform_blur.py](example/example_coverage_uniform_blur.py)
  — runnable end-to-end example: MNIST uniform-deblurring solved with
  PnP-ULA (`deepinv`), then a coverage curve plotted to `example/outputs/`.
  Pass `--use-vae` to overlay the VAE-embedding coverage curve.

## Install & run

```bash
pip install -e .[example]
python example/example_coverage_uniform_blur.py            # pixel-space curve
python example/example_coverage_uniform_blur.py --use-vae  # + VAE embedding
```

## Minimal usage

```python
import torch
from coverage_plots import Coverage

cov = Coverage(num_alphas=200)            # pixel-space only
# x:         (N, C, H, W)  ground-truth references
# x_samples: (N, S, C, H, W)  S posterior samples per reference
cov.update(x_samples, x)
alphas, coverage, coverage_emb = cov.compute(csv_path="coverage.csv")
# coverage_emb is None unless an embedding callable was passed at construction
```

Tensor-shape contract: references are `(N, C, H, W)` and posterior samples
are `(N, S, C, H, W)`. The metric is meant to be **accumulated** across many
references via repeated `update()` calls before `compute()`.
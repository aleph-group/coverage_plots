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
the inclusion test are formed from $\lVert \phi(\cdot) - \phi(c) \rVert$.

### Choosing the centre

The ball is centred on $c$, which defaults to the posterior mean $\bar{x}$.
That is the right choice when `x_samples` really are draws from a posterior
and the question is whether that posterior is calibrated.

It is the wrong choice when the samples are a **bootstrap around a point
estimate** $\hat{x}$ — each one a reconstruction of re-measured data. Then

$$\bar{x} \;=\; \hat{x} + \text{(the reconstructor's bias on its own output)},$$

an offset that does **not** shrink as $S$ grows. Centring on $\bar{x}$ asks
whether the truth lies near a point that is never reported.

Note this is not a uniformly stricter or looser test — which centre gives
higher coverage depends on where the truth sits relative to both, and on
whether the offset inflates the radius (it enters
$\lVert \hat{x}_s - \hat{x} \rVert$) or the distance (it enters
$\lVert x - \bar{x} \rVert$). The two answer different questions; pick the
one that matches the estimate you report.

Pass the estimate explicitly to ask the question that matches what you
publish:

```python
cov.update(x_samples, x, center=x_hat)     # one centre per reference
```

`center` takes a tensor or array shaped like `x`, i.e. `(N, C, H, W)`, or an
already-flattened `(N, D)`. The flat form cannot be used together with an
`embedding`, since the embedding is a map on images and $\phi(\bar{x})$ is
not the mean of $\phi(\hat{x}_s)$; pass the image form in that case. Omitting
`center` reproduces the behaviour of every earlier version exactly.

## TARP coverage

The package also implements the **TARP** diagnostic of
[Lemos, Coogan et al. 2023](https://arxiv.org/abs/2302.03026), a global
coverage test that does not assume the posterior is unimodal or centred on
its mean. For each reference $x_n$ a uniform reference point
$\theta^\star_n$ is drawn over the bounding box of the references, and a
per-reference credibility value

$$
f_n = \frac{1}{S}\sum_{s=1}^{S} \mathbf{1}\bigl[ d(\theta^\star_n, \hat{x}_{n,s}) < d(\theta^\star_n, x_n) \bigr]
$$

is computed. The expected coverage probability is the empirical CDF of
$\lbrace f_n \rbrace$ evaluated on a fixed grid in $[0, 1]$:

$$
\mathrm{ecp}(\alpha) = \frac{1}{N}\sum_{n=1}^{N} \mathbf{1}\bigl[ f_n \le \alpha \bigr].
$$

A well-calibrated sampler again follows the diagonal
$\mathrm{ecp}(\alpha) = \alpha$. Same reading rules as for the coverage
curve: below-diagonal → over-confident, above-diagonal → under-confident.
Like `Coverage`, TARP supports an optional `embedding` callable and a
pluggable `distance` (default L2).

## What's in the package

- [`coverage_plots.Coverage`](coverage_plots/coverage.py) — the public class.
  Accumulates the statistic over many reference batches via
  `update(x_samples, x)` and returns the curve via `compute()`.
  Pixel-space coverage is always computed; an extra embedding-space curve is
  computed in parallel when an `embedding` callable is supplied.
- [`coverage_plots.run_tarp`](coverage_plots/tarp.py) — stateless function
  implementing the TARP diagnostic. Takes the same `(x_samples, x)` shape
  contract as `Coverage` and returns `(alpha, ecp)` arrays ready to plot.
- [`coverage_plots.load_mnist_vae_embedding`](coverage_plots/vae.py) —
  factory that loads a small pre-trained MNIST VAE encoder and returns it as
  an embedding callable suitable for `Coverage(embedding=...)`.
- [example/example_coverage_uniform_blur.py](example/example_coverage_uniform_blur.py)
  — runnable end-to-end example: MNIST uniform-deblurring solved with
  PnP-ULA (`deepinv`), then both a coverage curve and a TARP curve plotted
  to `example/outputs/`. Pass `--use-vae` to overlay VAE-embedding curves.

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
cov.update(x_samples, x)                  # ball centred on the sample mean
alphas, coverage, coverage_emb = cov.compute(csv_path="coverage.csv")
# coverage_emb is None unless an embedding callable was passed at construction
```

Centred on a reported estimate instead (see *Choosing the centre*):

```python
cov.update(x_samples, x, center=x_hat)    # x_hat: (N, C, H, W)
```

TARP variant:

```python
from coverage_plots import run_tarp

alpha, ecp = run_tarp(x_samples, x)        # pixel-space TARP
# alpha, ecp = run_tarp(x_samples, x, embedding=phi)   # in embedding space
```

Tensor-shape contract: references are `(N, C, H, W)` and posterior samples
are `(N, S, C, H, W)`. The metric is meant to be **accumulated** across many
references via repeated `update()` calls before `compute()`.
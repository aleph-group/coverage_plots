"""
Coverage plot for an MNIST uniform-deblurring problem solved with PnP-ULA.


Forward model
    * 28x28 grayscale MNIST (no resize, no channel replication).
    * Uniform 5x5 blur kernel.
    * Additive Gaussian noise with std ``sigma = 0.02``.

Reconstruction
    * Score prior built from a Lipschitz pre-trained grayscale DnCNN
      (``pretrained="download_lipschitz"``).
    * Plug-and-Play ULA with ``alpha = 2.0``, ``sigma_den = sigma`` and the
      step-size formula
      ``step_size = 1 / (norm / sigma**2 + alpha * norm / sigma_den**2)``,
      where ``norm`` is the squared spectral norm of the forward operator.
    * Thinning 5, burn-in ratio 0.1, box clip ``(-1, 2)``.

Coverage statistic
    Pixel-space L2 distances between each posterior sample and the posterior
    mean, ranked at every level in :data:`alphas`. This matches the
    MSE-to-mean statistic used by the reference script up to a monotone
    transform, so quantile-based coverage is identical.

Run with::

    pip install -e .[example]
    python example/example_coverage_uniform_blur.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import deepinv as dinv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from coverage_plots import Coverage, load_mnist_vae_embedding, run_tarp

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28  # native MNIST resolution, matching the reference script
NUM_IMAGES = 100 if torch.cuda.is_available() else 10  # reference images N
NUM_SAMPLES = (
    1000 if torch.cuda.is_available() else 100
)  # posterior samples per image S (kept from the ULA chain)
NOISE_LEVEL = 0.02  # measurement noise std (matches reference)
KERNEL_SIZE = 5  # uniform PSF support (matches reference)
NUM_ALPHAS = 200  # number of coverage levels evaluated in [0, 1]
SEED = 0
# TARP parameters
TARP_NUM_BINS = 8  # number of bins for TARP histogram (defaults to max(N//10, 2))
# ULA hyperparameters (matches reference; reduced max_iter for CPU runtime).
ULA_ALPHA = 2.0
ULA_SIGMA_DEN = NOISE_LEVEL
ULA_THINNING = 5
ULA_BURNIN = 0.1
ULA_CLIP = (-1.0, 2.0)
ULA_KEPT_SAMPLES_TARGET = 1000  # post-burn-in, post-thinning samples we aim for

REPO_ROOT = Path(__file__).resolve().parents[1]
VAE_WEIGHTS = REPO_ROOT / "models" / "MNIST_VAE.pth.tar"
ORIGINAL_DATA_DIR = dinv.utils.get_data_home() / "datasets" / "MNIST"
OUT_DIR = Path(__file__).resolve().parent / "outputs"  # "example_outputs"
OUT_DIR.mkdir(exist_ok=True)


def get_dataset():
    """MNIST test split, kept at native 28x28 grayscale resolution."""
    tfm = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(
        root=str(ORIGINAL_DATA_DIR), train=False, download=True, transform=tfm
    )


def build_physics():
    """Uniform 5x5 blur with additive Gaussian noise (matches reference)."""
    kernel = torch.full(
        (1, 1, KERNEL_SIZE, KERNEL_SIZE),
        fill_value=1.0 / (KERNEL_SIZE * KERNEL_SIZE),
        device=DEVICE,
    )
    return dinv.physics.BlurFFT(
        img_size=(1, IMG_SIZE, IMG_SIZE),
        filter=kernel,
        noise_model=dinv.physics.GaussianNoise(sigma=NOISE_LEVEL),
        device=DEVICE,
    )


def build_sampler(physics, x_for_norm, num_samples):
    """
    PnP-ULA sampler reproducing the reference script's hyperparameters.

    The step size is set from the squared spectral norm of ``physics`` using
    the same formula as the reference. ``thinning`` matches the reference
    (5) and ``max_iter`` is sized so that ``num_samples`` thinned samples are
    kept after burn-in.
    """
    denoiser = (
        dinv.models.DnCNN(
            in_channels=1,
            out_channels=1,
            pretrained="download_lipschitz",
        )
        .to(DEVICE)
        .eval()
    )
    prior = dinv.optim.ScorePrior(denoiser=denoiser).to(DEVICE)
    likelihood = dinv.optim.data_fidelity.L2(sigma=NOISE_LEVEL)

    norm = physics.compute_norm(x_for_norm).detach().cpu().item()
    step_size = 1.0 / (norm / NOISE_LEVEL**2 + ULA_ALPHA * norm / ULA_SIGMA_DEN**2)

    # Reference: max_iter = MC * thinning / (1 - burnin), with MC the number
    # of post-burn-in thinned samples we want the chain to produce. Only the
    # last ``num_samples`` of those are kept in ``history_size``.
    max_iter = int(
        max(num_samples, ULA_KEPT_SAMPLES_TARGET) * ULA_THINNING / (1.0 - ULA_BURNIN)
    )
    thinning = ULA_THINNING
    kept = int((1.0 - ULA_BURNIN) * max_iter / thinning)

    print(
        f"  ULA: norm={norm:.3f}, step_size={step_size:.3e}, "
        f"max_iter={max_iter}, thinning={thinning}, kept~={kept}"
    )

    params = {
        "step_size": step_size,
        "alpha": ULA_ALPHA,
        "sigma": ULA_SIGMA_DEN,
    }
    # ``history_size`` must span the full post-burn-in thinned chain so that
    # the ``num_samples`` returned by ``sample_posterior`` cover the whole
    # mixing window (uniform subsampling), not just the last ``num_samples``
    # consecutive states (which would be highly correlated).
    return dinv.sampling.sampling_builder(
        "ULA",
        prior=prior,
        data_fidelity=likelihood,
        max_iter=max_iter,
        params_algo=params,
        thinning=thinning,
        burnin_ratio=ULA_BURNIN,
        clip=ULA_CLIP,
        history_size=kept,
        verbose=False,
    )


def sample_posterior(sampler, y, physics, num_samples):
    """
    Draw ``num_samples`` posterior samples per measurement from a single ULA chain.

    Each batch element runs an independent ULA chain in parallel. After the
    chain finishes, ``num_samples`` thinned states are picked uniformly
    across the full post-burn-in history so they are well-decorrelated.

    :returns: ``(N, S, C, H, W)`` tensor on CPU.
    """
    sampler.sample(y, physics, x_init=physics.A_adjoint(y))
    chain = sampler.get_chain()
    if len(chain) < num_samples:
        raise RuntimeError(
            f"ULA chain returned {len(chain)} states but {num_samples} were requested. "
            "Increase ULA_KEPT_SAMPLES_TARGET."
        )
    idx = np.linspace(0, len(chain) - 1, num_samples, dtype=int)
    samples = [chain[i]["x"].detach().cpu() for i in idx]
    return torch.stack(samples, dim=1)  # (N, S, C, H, W)


def _psnr(x_hat, x_ref, max_val=1.0):
    """Per-image PSNR in dB; both inputs share shape ``(N, C, H, W)``."""
    mse = (x_hat - x_ref).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(max_val**2 / mse.clamp_min(1e-12))


def _to_displayable(img):
    """Map a ``(C, H, W)`` tensor to a ``(H, W[, 3])`` array in ``[0, 1]``."""
    img = img.detach().cpu().float().clamp(0.0, 1.0)
    if img.shape[0] == 1:
        return img.squeeze(0).numpy()
    return img.permute(1, 2, 0).numpy()


def plot_reconstructions(x, y, x_samples, out_path, max_samples=4):
    """Per-reference grid: ground truth, measurement, posterior mean, samples."""
    n = x.shape[0]
    s = min(max_samples, x_samples.shape[1])
    x_mean = x_samples.mean(dim=1)

    n_cols = 3 + s
    fig, axes = plt.subplots(n, n_cols, figsize=(1.6 * n_cols, 1.6 * n), squeeze=False)
    col_titles = ["Ground truth", "Measurement", "Posterior mean"] + [
        f"Sample {i + 1}" for i in range(s)
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9)
    for i in range(n):
        cells = [x[i], y[i], x_mean[i]] + [x_samples[i, k] for k in range(s)]
        for j, img in enumerate(cells):
            axes[i, j].imshow(_to_displayable(img), cmap="gray", vmin=0.0, vmax=1.0)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-vae",
        action="store_true",
        help=(
            "Compute coverage in the latent space of the pre-trained MNIST VAE "
            f"checkpoint at {VAE_WEIGHTS}. Defaults to pixel-space distances."
        ),
    )
    parser.add_argument(
        "--vae-weights",
        type=Path,
        default=VAE_WEIGHTS,
        help="Path to the MNIST VAE checkpoint (used only with --use-vae).",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)

    dataset = get_dataset()
    loader = DataLoader(dataset, batch_size=NUM_IMAGES, shuffle=True)
    x, _ = next(iter(loader))
    x = x.to(DEVICE)

    physics = build_physics()
    y = physics(x)

    sampler = build_sampler(physics, x, NUM_SAMPLES)
    print(
        f"Running PnP-ULA on {NUM_IMAGES} images "
        f"(keeping {NUM_SAMPLES} thinned posterior samples per chain)..."
    )
    x_samples = sample_posterior(sampler, y, physics, NUM_SAMPLES)
    print(f"posterior samples shape: {tuple(x_samples.shape)}")

    # PSNR diagnostics: posterior mean and average over posterior samples.
    x_cpu = x.cpu()
    x_mean = x_samples.mean(dim=1)
    psnr_mean = _psnr(x_mean, x_cpu)
    psnr_samples = torch.stack(
        [_psnr(x_samples[:, s], x_cpu) for s in range(x_samples.shape[1])], dim=1
    )
    psnr_sample_avg = psnr_samples.mean(dim=1)
    print(
        f"PSNR posterior mean: {psnr_mean.mean().item():.2f} ± "
        f"{psnr_mean.std().item():.2f} dB"
    )
    print(
        f"PSNR posterior samples (avg over S then over N): "
        f"{psnr_sample_avg.mean().item():.2f} ± "
        f"{psnr_sample_avg.std().item():.2f} dB"
    )

    plot_reconstructions(
        x_cpu, y.cpu(), x_samples, OUT_DIR / "reconstructions_uniform.pdf"
    )

    if args.use_vae:
        print(f"Using MNIST VAE embedding from {args.vae_weights}")
        embedding = load_mnist_vae_embedding(args.vae_weights, device=DEVICE)
        suffix = "_vae"
    else:
        embedding = None
        suffix = ""

    coverage = Coverage(embedding=embedding, num_alphas=NUM_ALPHAS, device=DEVICE)
    coverage.update(x_samples, x_cpu)
    alphas, cov, cov_emb = coverage.compute(
        csv_path=OUT_DIR / f"coverage_uniform{suffix}.csv"
    )

    n_marks = min(20, len(alphas))
    mark_idx = np.linspace(0, len(alphas) - 1, n_marks, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="ideal")
    ax.plot(
        alphas,
        cov,
        marker="o",
        markevery=mark_idx.tolist(),
        markersize=4,
        label="pixel space",
    )
    if cov_emb is not None:
        ax.plot(
            alphas,
            cov_emb,
            marker="s",
            markevery=mark_idx.tolist(),
            markersize=4,
            label="VAE embedding",
        )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("coverage")
    ax.set_title("Coverage plot — MNIST uniform deblurring (PnP-ULA)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    out_pdf = OUT_DIR / f"coverage_uniform{suffix}.pdf"
    fig.savefig(out_pdf)
    print(f"saved {out_pdf}")

    # -------------------------------------------------------------------------
    # TARP coverage diagnostic (Lemos et al. 2023, arXiv:2302.03026).
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    tarp_alpha, tarp_ecp = run_tarp(
        x_samples,
        x_cpu,
        embedding=None,
        num_bins=TARP_NUM_BINS,
        rng=rng,
    )
    if embedding is not None:
        _, tarp_ecp_emb = run_tarp(
            x_samples,
            x_cpu,
            embedding=embedding,
            num_bins=TARP_NUM_BINS,
            rng=rng,
        )
    else:
        tarp_ecp_emb = None

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="ideal")
    ax.plot(tarp_alpha, tarp_ecp, marker="o", markersize=4, label="pixel space")
    if tarp_ecp_emb is not None:
        ax.plot(
            tarp_alpha, tarp_ecp_emb, marker="s", markersize=4, label="VAE embedding"
        )
    ax.set_xlabel(r"credibility level $\alpha$")
    ax.set_ylabel("expected coverage probability")
    ax.set_title("TARP coverage — MNIST uniform deblurring (PnP-ULA)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    out_tarp = OUT_DIR / f"tarp_uniform{suffix}.pdf"
    fig.savefig(out_tarp)
    plt.close(fig)
    print(f"saved {out_tarp}")


if __name__ == "__main__":
    main()

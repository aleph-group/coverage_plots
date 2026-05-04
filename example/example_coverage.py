"""
Coverage plot for an MNIST Gaussian-deblurring problem solved with PnP-ULA.

We use ``deepinv`` to:
  * load grayscale MNIST via :class:`torchvision.datasets.MNIST`,
  * build a Gaussian-blur forward operator with additive Gaussian noise,
  * generate ``S`` posterior samples per ground-truth image with the
    Plug-and-Play Unadjusted Langevin Algorithm (:class:`deepinv.sampling.ULAIterator`)
    using a pre-trained, Lipschitz-constrained grayscale DnCNN denoiser as the
    score prior (see ``dev/tutorials/demo_sampling.py``).

The samples are then fed to :class:`coverage_plots.Coverage` and the empirical
coverage curve is plotted.

Run with::

    pip install -e .[example]
    python example/example_coverage.py
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

from coverage_plots import Coverage, load_mnist_vae_embedding

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 32  # MNIST is 28x28; 32 plays nicely with the DnCNN downsamplings
NUM_IMAGES = 16  # reference images N
NUM_SAMPLES = 16  # posterior samples per image S (kept from the ULA chain)
NOISE_LEVEL = 0.05  # measurement noise std
BLUR_SIGMA = 1.0  # Gaussian blur std (in pixels)
ULA_MAX_ITER = 1000  # total ULA iterations per chain
ULA_BURNIN = 0.2  # fraction of iterations discarded as burn-in
ULA_REGULARIZATION = 0.9  # alpha in PnP-ULA
ULA_DENOISER_SIGMA = 2 / 255  # noise level used by the score prior
NUM_ALPHAS = 200  # number of coverage levels evaluated in [0, 1]
SEED = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
VAE_WEIGHTS = REPO_ROOT / "models" / "MNIST_VAE.pth.tar"
ORIGINAL_DATA_DIR = dinv.utils.get_data_home() / "datasets" / "MNIST"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def get_dataset():
    """MNIST test split, resized to ``IMG_SIZE``, kept as single-channel grayscale."""
    tfm = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),  # (1, H, W) in [0, 1]
        ]
    )
    return datasets.MNIST(
        root=str(ORIGINAL_DATA_DIR), train=False, download=True, transform=tfm
    )


def build_physics():
    """Gaussian-blur forward operator with additive Gaussian noise."""
    psf = dinv.physics.blur.gaussian_blur(sigma=(BLUR_SIGMA, BLUR_SIGMA))
    return dinv.physics.BlurFFT(
        img_size=(1, IMG_SIZE, IMG_SIZE),
        filter=psf,
        noise_model=dinv.physics.GaussianNoise(sigma=NOISE_LEVEL),
        device=DEVICE,
    )


def build_sampler(num_samples):
    """
    PnP-ULA sampler with a pre-trained Lipschitz grayscale DnCNN as score prior.

    The chain runs for ``ULA_MAX_ITER`` iterations with a ``ULA_BURNIN`` warm-up
    fraction; ``thinning`` is chosen so that exactly ``num_samples`` post-burn-in
    samples are kept in the history buffer.
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

    kept_iters = max(1, int((1.0 - ULA_BURNIN) * ULA_MAX_ITER))
    thinning = max(1, kept_iters // num_samples)

    params = {
        "step_size": 0.01 * NOISE_LEVEL**2,
        "alpha": ULA_REGULARIZATION,
        "sigma": ULA_DENOISER_SIGMA,
    }
    return dinv.sampling.sampling_builder(
        "ULA",
        prior=prior,
        data_fidelity=likelihood,
        max_iter=ULA_MAX_ITER,
        params_algo=params,
        thinning=thinning,
        burnin_ratio=ULA_BURNIN,
        history_size=num_samples,
        verbose=False,
    )


def sample_posterior(sampler, y, physics, num_samples):
    """
    Draw ``num_samples`` posterior samples per measurement from a single ULA chain.

    All ``N`` measurements are processed in parallel by the same Markov chain
    (each batch element is an independent chain). After the chain finishes,
    the last ``num_samples`` thinned states stored in the history are used.

    :returns: ``(N, S, C, H, W)`` tensor on CPU.
    """
    sampler.sample(y, physics)
    chain = sampler.get_chain()  # list of dicts with key "x"
    if len(chain) < num_samples:
        raise RuntimeError(
            f"ULA chain returned {len(chain)} states but {num_samples} were requested. "
            "Increase ULA_MAX_ITER or reduce NUM_SAMPLES."
        )
    samples = [state["x"].detach().cpu() for state in chain[-num_samples:]]
    return torch.stack(samples, dim=1)  # (N, S, C, H, W)


def _psnr(x_hat, x_ref, max_val=1.0):
    """
    Per-image PSNR in dB between ``x_hat`` and ``x_ref``.

    Both tensors are expected to share shape ``(N, C, H, W)``. Returns a
    1-D tensor of length ``N``.
    """
    mse = (x_hat - x_ref).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(max_val**2 / mse.clamp_min(1e-12))


def _to_displayable(img):
    """Map a ``(C, H, W)`` tensor to a ``(H, W[, 3])`` array in ``[0, 1]``."""
    img = img.detach().cpu().float()
    img = img.clamp(0.0, 1.0)
    if img.shape[0] == 1:
        return img.squeeze(0).numpy()
    return img.permute(1, 2, 0).numpy()


def plot_reconstructions(x, y, x_samples, out_path, max_samples=4):
    """
    Save a per-reference grid of ground truth, measurement, posterior mean and
    a few posterior samples.

    :param torch.Tensor x: ground truth, shape ``(N, C, H, W)``.
    :param torch.Tensor y: measurements, shape ``(N, C, H, W)``.
    :param torch.Tensor x_samples: posterior samples, shape ``(N, S, C, H, W)``.
    :param Path out_path: PDF file to write.
    :param int max_samples: number of posterior samples to display per row.
    """
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

    sampler = build_sampler(NUM_SAMPLES)
    print(
        f"Running PnP-ULA for {ULA_MAX_ITER} iterations on {NUM_IMAGES} images "
        f"(keeping {NUM_SAMPLES} thinned posterior samples per chain)..."
    )
    x_samples = sample_posterior(sampler, y, physics, NUM_SAMPLES)
    print(f"posterior samples shape: {tuple(x_samples.shape)}")

    # PSNR diagnostics: posterior mean vs. ground truth, and average over samples.
    x_cpu = x.cpu()
    x_mean = x_samples.mean(dim=1)  # (N, C, H, W)
    psnr_mean = _psnr(x_mean, x_cpu)  # (N,)
    psnr_samples = torch.stack(
        [_psnr(x_samples[:, s], x_cpu) for s in range(x_samples.shape[1])], dim=1
    )  # (N, S)
    psnr_sample_avg = psnr_samples.mean(dim=1)  # (N,)
    print(
        f"PSNR posterior mean: {psnr_mean.mean().item():.2f} ± "
        f"{psnr_mean.std().item():.2f} dB"
    )
    print(
        f"PSNR posterior samples (avg over S then over N): "
        f"{psnr_sample_avg.mean().item():.2f} ± "
        f"{psnr_sample_avg.std().item():.2f} dB"
    )

    # Visual sanity check: ground truth | measurement | posterior mean | samples.
    plot_reconstructions(x_cpu, y.cpu(), x_samples, OUT_DIR / "reconstructions.pdf")

    if args.use_vae:
        print(f"Using MNIST VAE embedding from {args.vae_weights}")
        embedding = load_mnist_vae_embedding(args.vae_weights, device=DEVICE)
        suffix = "_vae"
        title = "Coverage plot — MNIST Gaussian deblurring (PnP-ULA, VAE embedding)"
    else:
        embedding = None
        suffix = ""
        title = "Coverage plot — MNIST Gaussian deblurring (PnP-ULA, pixel space)"

    coverage = Coverage(embedding=embedding, num_alphas=NUM_ALPHAS, device=DEVICE)
    coverage.update(x_samples, x.cpu())
    alphas, cov, _ = coverage.compute(csv_path=OUT_DIR / f"coverage{suffix}.csv")

    # Show markers on a sparse subset of alphas so they remain legible.
    n_marks = min(20, len(alphas))
    mark_idx = np.linspace(0, len(alphas) - 1, n_marks, dtype=int)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="ideal")
    ax.plot(
        alphas,
        cov,
        marker="o",
        markevery=mark_idx.tolist(),
        markersize=4,
        label="empirical",
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("coverage")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    out_pdf = OUT_DIR / f"coverage{suffix}.pdf"
    fig.savefig(out_pdf)
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()

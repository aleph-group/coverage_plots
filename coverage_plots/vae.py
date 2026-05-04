import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from time import time


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.beta_gen = 1.0

    def posterior_sample(self, mu_z, logsd_z):
        sd_z = torch.exp(logsd_z)
        eps = torch.randn_like(sd_z)
        return mu_z + eps * sd_z

    def forward(self, x):
        mu_z, logsd_z = self.encoder(x)
        z = self.posterior_sample(mu_z, logsd_z)
        mu_x, gamma_x = self.decoder(z)
        return mu_x, gamma_x, mu_z, logsd_z

    def compute_loss(self, x, mu_x, gamma_x, mu_z, logsd_z):
        BATCH_SIZE = x.size(0)
        HALF_LOG_TWO_PI = 0.91894
        sd_z = torch.exp(logsd_z)

        kl_loss = (
            0.5 * torch.sum(mu_z.pow(2) + sd_z.pow(2) - 1 - 2 * logsd_z) / BATCH_SIZE
        )
        gen_loss = (
            torch.sum(
                0.5 * ((x - mu_x) / gamma_x).pow(2) + gamma_x.log() + HALF_LOG_TWO_PI
            )
            / BATCH_SIZE
        )

        return kl_loss + self.beta_gen * gen_loss

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def fit(
        self,
        dataloaders,
        n_epochs=10,
        lr=1e-3,
        dvae_sigma=0,
        device="cuda",
        nprint=1,
        path=".",
    ):

        tic = time()

        self.to(device)
        train_loader, test_loader = dataloaders
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            train_loss = 0.0
            self.train()
            train_iter = iter(train_loader)
            test_iter = iter(test_loader)
            # Training steps
            for i in range(len(train_loader)):
                x = next(train_iter)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = x[0]
                # Load data on device
                x = x.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if dvae_sigma:  # DenoisingVAE
                    x_noisy = x + dvae_sigma / 255 * torch.randn(
                        *x.shape, device=device
                    )
                    mu_x, gamma_x, mu_z, logsd_z = self.forward(x_noisy)
                else:
                    mu_x, gamma_x, mu_z, logsd_z = self.forward(x)

                loss = self.compute_loss(x, mu_x, gamma_x, mu_z, logsd_z)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.shape[0]

            train_loss /= len(train_loader.dataset)

            # Print statistics every 'nprint' epochs
            if epoch % nprint == nprint - 1:

                # Model evaluation (test)
                with torch.no_grad():
                    self.eval()

                    # Test loss
                    test_loss = 0
                    for i in range(len(test_loader)):
                        data = next(test_iter)
                        if isinstance(data, tuple) or isinstance(data, list):
                            data = data[0]
                        data = data.to(device)
                        mu_x, gamma_x, mu_z, logsd_z = self.forward(data)
                        test_loss += (
                            data.shape[0]
                            * self.compute_loss(
                                data, mu_x, gamma_x, mu_z, logsd_z
                            ).item()
                        )

                        # Reconstructions
                        if i == 0:
                            n = min(data.size(0), 8)
                            comparison = torch.cat([data[:n], mu_x[:n]])
                            save_image(
                                comparison.cpu(),
                                os.path.join(
                                    path, "reconstruction_" + str(epoch) + ".png"
                                ),
                                nrow=n,
                            )

                    test_loss /= len(test_loader.dataset)

                    print(
                        "[Epoch %d / %d]  Train loss = %.5f  |  Test loss = %.5f  |  gamma_x = %.4f  |  %.2f sec"
                        % (
                            epoch + 1,
                            n_epochs,
                            train_loss,
                            test_loss,
                            gamma_x.item(),
                            time() - tic,
                        )
                    )

                    # Random samples
                    sample = torch.randn(64, self.latent_dim).to(device)
                    sample = self.decoder(sample)[0].cpu()
                    save_image(
                        sample, os.path.join(path, "sample_" + str(epoch) + ".png")
                    )

        toc = time()
        print("Training finished - Elapsed time: %.2f sec" % (toc - tic))

    def evaluate(self, test_loader, device="cuda"):
        self.eval()
        test_iter = iter(test_loader)
        test_loss = 0
        MU = []
        for i in range(len(test_loader)):
            print("\r%d/%d" % (i, len(test_loader)), end="")
            data = next(test_iter)
            if isinstance(data, tuple) or isinstance(data, list):
                data = data[0]
            data = data.to(device)
            mu_x, gamma_x, mu_z, logsd_z = self.forward(data)
            MU.append(mu_z)
            test_loss += (
                data.shape[0]
                * self.compute_loss(data, mu_x, gamma_x, mu_z, logsd_z).item()
            )

        print("Computing covariance matrix")
        MU = torch.cat(MU, dim=0).T
        tr = torch.trace(torch.cov(MU))
        test_loss /= len(test_loader.dataset)
        print("Test loss = %.5f" % test_loss, "Trace = %.5f" % tr)
        return test_loss, tr


class VAEfc(VAE):
    def __init__(
        self,
        x_shape,
        latent_dim,
        enc_hidden_dims,
        d_activ=F.relu,
        return_latent_sd=True,
    ):
        super(VAEfc, self).__init__()
        self.x_shape = x_shape
        self.input_dim = np.prod(x_shape)
        self.latent_dim = latent_dim
        self.num_hidden_layers = len(enc_hidden_dims)
        self.d_activ = d_activ

        self.encoder_layers = nn.ModuleList(
            [nn.Linear(self.input_dim, enc_hidden_dims[0])]
        )
        self.encoder_layers.extend(
            [
                nn.Linear(enc_hidden_dims[i], enc_hidden_dims[i + 1])
                for i in range(self.num_hidden_layers - 1)
            ]
        )
        self.encoder_layers.append(
            nn.Linear(enc_hidden_dims[-1], self.latent_dim)
        )  # mu_z
        self.encoder_layers.append(
            nn.Linear(enc_hidden_dims[-1], self.latent_dim)
        )  # logsd_z

        self.decoder_layers = nn.ModuleList(
            [nn.Linear(self.latent_dim, enc_hidden_dims[-1])]
        )
        self.decoder_layers.extend(
            [
                nn.Linear(enc_hidden_dims[i], enc_hidden_dims[i - 1])
                for i in range(self.num_hidden_layers - 1, 0, -1)
            ]
        )
        self.decoder_layers.append(
            nn.Linear(enc_hidden_dims[0], self.input_dim)
        )  # mu_x
        self.gamma_x = nn.Parameter(torch.ones(1))  # gamma_x

        self.return_latent_sd = return_latent_sd

    def encoder(self, x):
        h = F.relu(self.encoder_layers[0](x.flatten(start_dim=1)))
        for i in range(1, self.num_hidden_layers):
            h = F.relu(self.encoder_layers[i](h))
        mu_z = self.encoder_layers[-2](h)
        logsd_z = self.encoder_layers[-1](h)
        return (mu_z, logsd_z) if self.return_latent_sd else mu_z

    def decoder(self, z):
        h = self.d_activ(self.decoder_layers[0](z))
        for i in range(1, self.num_hidden_layers):
            h = self.d_activ(self.decoder_layers[i](h))
        mu_x = torch.sigmoid(self.decoder_layers[-1](h))
        return mu_x.view(-1, *self.x_shape), self.gamma_x


def load_mnist_vae_embedding(
    weights_path,
    device=None,
    latent_dim=12,
    enc_hidden_dims=(512, 512),
    image_size=28,
):
    r"""
    Build an embedding callable backed by a pre-trained MNIST VAE.

    The returned function maps a batch of images of shape ``(B, C, H, W)`` to
    the VAE's latent mean of shape ``(B, latent_dim)``. Inputs are first
    converted to single-channel grayscale (averaging the channel dimension if
    needed), min-max normalised to ``[0, 1]`` per batch, resized to
    ``image_size x image_size`` and flattened before being fed to the encoder.

    The VAE architecture must match the checkpoint; defaults match the weights
    shipped in ``models/MNIST_VAE.pth.tar`` (28x28 MNIST,
    latent dim 12, two hidden layers of width 512).

    :param str | os.PathLike weights_path: path to the ``.pth.tar`` checkpoint.
    :param torch.device device: device on which to place the VAE.
    :param int latent_dim: latent dimensionality of the VAE.
    :param tuple[int, ...] enc_hidden_dims: encoder hidden dimensions.
    :param int image_size: spatial size expected by the VAE.
    :returns: a callable ``embedding(x) -> torch.Tensor`` of shape
        ``(B, latent_dim)``, suitable for ``Coverage(embedding=...)``.
    """
    import torch.nn.functional as F_

    vae = VAEfc(
        x_shape=(image_size, image_size),
        latent_dim=latent_dim,
        enc_hidden_dims=list(enc_hidden_dims),
        d_activ=F_.relu,
        return_latent_sd=False,
    )
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    vae.load_state_dict(state)
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def embedding(x):
        x = x.to(device)
        if x.ndim == 4 and x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        if x.shape[-1] != image_size or x.shape[-2] != image_size:
            x = F_.interpolate(
                x, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
        # Per-batch min-max normalisation to [0, 1] (matches training preprocessing).
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        return vae.encoder(x)

    return embedding

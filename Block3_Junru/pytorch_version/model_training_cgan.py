#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：convnet_prize 
@File    ：model_training_cgan.py
@Author  ：Junru Jin
@Date    ：5/23/24 6:07 PM 
'''
import torch
from torch import nn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
# import lightning.pytorch as lp
from torchvision import transforms, datasets
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torchvision.utils import make_grid

from torch import nn, optim
import torch.nn.functional as F


class VisualizationCallback(Callback):
    def __init__(self, latent_dim, num_samples=10, num_classes=10):
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.num_classes = num_classes

    def on_epoch_end(self, trainer, pl_module):
        z = torch.randn(self.num_samples, self.latent_dim, device=pl_module.device)
        labels = torch.randint(0, self.num_classes, (self.num_samples,), device=pl_module.device)

        # Random Samples + Reconstruction
        self.visualize_random_samples(pl_module, z, labels)
        # Interpolation (choose two random points)
        self.visualize_interpolation(pl_module, z[0:1], z[1:2], labels[0:1])
        # Grid Search in 2D latent space visualization
        self.visualize_grid_search(pl_module)
        # t-SNE Visualization of the Latent Space
        self.visualize_tsne(pl_module, trainer.train_dataloader)

    def visualize_random_samples(self, pl_module, z, labels):
        with torch.no_grad():
            samples = pl_module(z, labels).cpu()
        grid = make_grid(samples, nrow=5)
        plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title("Random Samples")
        plt.axis('off')
        plt.show()

    def visualize_interpolation(self, pl_module, start, end, labels):
        steps = 10
        z = torch.linspace(0, 1, steps=steps)[:, None].to(pl_module.device) * (end - start) + start
        with torch.no_grad():
            samples = pl_module(z, labels.expand(steps))
        grid = make_grid(samples, nrow=steps)
        plt.figure(figsize=(15, 3))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title("Interpolation")
        plt.axis('off')
        plt.show()

    def visualize_grid_search(self, pl_module):
        # Assuming a simplified 2D latent space for visualization
        dim_range = 10
        z = torch.stack([torch.tensor([i, j]).float() for i in np.linspace(-2, 2, dim_range) for j in
                         np.linspace(-2, 2, dim_range)]).to(pl_module.device)
        labels = torch.zeros(z.size(0), dtype=torch.long).to(pl_module.device)  # Using one class for simplicity
        with torch.no_grad():
            samples = pl_module(z, labels).cpu()
        grid = make_grid(samples, nrow=dim_range)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title("Grid Search")
        plt.axis('off')
        plt.show()

    def visualize_tsne(self, pl_module, dataloader):
        embeddings, labels = [], []
        for batch, _labels in dataloader:
            batch = batch.to(pl_module.device)
            with torch.no_grad():
                z, _ = pl_module.encoder(batch)  # Assuming CGAN has an encoder method for obtaining z
            embeddings.append(z)
            labels.append(_labels)

        embeddings = torch.cat(embeddings).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar()
        plt.title("t-SNE Visualization of the Latent Space")
        plt.show()




class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        # utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        # utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class CGAN(LightningModule):
    def __init__(self, latent_dim=100, num_classes=10, lr=0.0002, b1=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.G = generator(input_dim=latent_dim, output_dim=1, input_size=self.hparams.input_size,
                           class_num=num_classes)
        self.D = discriminator(input_dim=1, output_dim=1, input_size=self.hparams.input_size,
                               class_num=self.hparams.class_num)


    def forward(self, z, labels):
        return self.generator(z, labels)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        # Sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=self.device)
        # Generate images
        generated_imgs = self(z, labels)

        # Ground truths
        valid = torch.ones(imgs.size(0), 1, device=self.device)
        fake = torch.zeros(imgs.size(0), 1, device=self.device)

        # Train Generator
        if optimizer_idx == 0:
            g_loss = self.adversarial_loss(self.discriminator(generated_imgs, labels), valid)
            self.log('g_loss', g_loss)
            return g_loss

        # Train Discriminator
        if optimizer_idx == 1:
            real_loss = self.adversarial_loss(self.discriminator(imgs, labels), valid)
            fake_loss = self.adversarial_loss(self.discriminator(generated_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            self.log('d_loss', d_loss)
            return d_loss

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = 0.999
        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_train_epoch_end(self) -> None:
        z = torch.randn(8, self.hparams.latent_dim, device=self.device)
        labels = torch.randint(0, 10, (8,), device=self.device)
        sample_images = self.generator(z, labels)
        grid = make_grid(sample_images, nrow=4, normalize=True)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


def objective(trial):
    # Define hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    b1 = trial.suggest_uniform('b1', 0.5, 0.9)
    latent_dim = trial.suggest_int('latent_dim', 50, 200)

    # Model setup
    model = CGAN(channels=3, width=28, height=28, latent_dim=latent_dim, lr=lr, b1=b1)
    image_callback = VisualizationCallback(latent_dim=latent_dim)

    # Trainer setup
    trainer = Trainer(
        max_epochs=10,
        callbacks=[image_callback],
        logger=False,
        checkpoint_callback=False  # Turn off checkpointing for optimization
    )

    # Data setup
    mnist_train = datasets.MNIST(root='/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/CVAE/',
                                 train=True, download=True)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)

    # Train the model
    trainer.fit(model, train_loader)

    # You can use any metric for the objective; here we use a dummy for illustration
    return trainer.callback_metrics.get('loss', 0.0)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
print('Best trial:', study.best_trial.params)

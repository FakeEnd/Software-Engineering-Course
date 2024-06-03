#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：convnet_prize 
@File    ：model_training_V1.py
@Author  ：Junru Jin
@Date    ：5/23/24 3:42 PM 
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

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ConvCVAE(LightningModule):
    def __init__(self, latent_dim=20, num_conv_layers=3, conv_units=16, kernel_size=3, activation='relu',
                 dense_units=128, lr=1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.lr = lr

        self.encoder = nn.Sequential()
        current_channels = 1 + num_classes  # initial channels + embedding channels
        current_size = 28  # initial image size (MNIST 28x28)

        for i in range(num_conv_layers):
            self.encoder.add_module(f"conv-{i}",
                                    nn.Conv2d(current_channels, conv_units, kernel_size, stride=2, padding=1))
            # self.encoder.add_module(f"activation-{i}", nn.ReLU() if activation == 'relu' else nn.ELU())
            if activation == 'relu':
                self.encoder.add_module(f"activation-{i}", nn.ReLU())
            elif activation == 'elu':
                self.encoder.add_module(f"activation-{i}", nn.ELU())
            elif activation == 'linear':
                self.encoder.add_module(f"activation-{i}", nn.Identity())
            elif activation == 'softmax':
                self.encoder.add_module(f"activation-{i}", nn.Softmax(dim=1))
            current_channels = conv_units
            current_size = (current_size + 2 - kernel_size) // 2 + 1  # update size for each layer
            if current_size < 5:
                break

            # Flattened size calculation
        self.flattened_size = current_channels * current_size * current_size
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)

        # Starting dimensions for the decoder
        self.initial_height = 7  # This should be calculated based on the encoder output
        self.initial_width = 7  # This should be calculated based on the encoder output
        self.initial_conv_channels = 32  # Adjust as necessary

        # First layer to map latent space to convolutional space
        self.decoder_input = nn.Linear(latent_dim + num_classes,
                                       self.initial_conv_channels * self.initial_height * self.initial_width)

        # Decoder using ConvTranspose2d layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.initial_conv_channels, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid()
        )

    def encode(self, x, labels):
        labels = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)
        labels = labels.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        z = torch.cat([z, self.label_emb(labels)], dim=1)
        z = self.decoder_input(z)
        z = z.view(-1, self.initial_conv_channels, self.initial_height,
                   self.initial_width)  # Reshape to (batch_size, channels, height, width)
        return self.decoder(z)

    def forward(self, x, labels):
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, labels).squeeze(1)
        return x_recon, mu, log_var

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x_recon, mu, log_var = self(x, labels)
        # print(x_recon.shape, x.view(-1, 784).shape)

        # Ensure output is between 0 and 1
        # assert x_recon.min() >= 0 and x_recon.max() <= 1, "Outputs are not scaled to [0, 1]"

        # Check shapes before loss computation
        # assert x_recon.shape == x.view(-1,
        #                                784).shape, f"Mismatched shapes between output and target: {x_recon.shape} vs {x.view(-1, 784).shape}"

        recon_loss = nn.functional.binary_cross_entropy_with_logits(x_recon.view(-1, 784), x.view(-1, 784), reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kld_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# from lightning.pytorch.callbacks import Callback

class VisualizeLatentSpace(Callback):
    def on_train_end(self, trainer, pl_module):
        print("Visualizing latent space...")
        z = torch.randn(1, pl_module.hparams.latent_dim).to(pl_module.device)
        labels = torch.randint(0, 10, (1,)).to(pl_module.device)
        img = nn.functional.sigmoid(pl_module.decode(z, labels)).view(28, 28).detach().cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f"Sample with Label: {labels.item()}")
        plt.show()


def objective(trial):
    lr = trial.suggest_loguniform("lr", 0.000001, 0.01)
    latent_dim = trial.suggest_int("latent_dim", 2, 128)
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
    conv_units = trial.suggest_int("conv_units", 4, 256)
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    activation = trial.suggest_categorical("activation", ['relu', 'elu', 'linear'])
    dense_units = trial.suggest_int("dense_units", 4, 256)

    model = ConvCVAE(latent_dim=64, num_conv_layers=3, conv_units=64,
                     kernel_size=3, activation=activation, dense_units=128, lr=lr, num_classes=10)
    trainer = Trainer(
        devices=1,
        max_epochs=10,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="train_loss"), VisualizeLatentSpace()],
        logger=False
    )

    # mnist_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    mnist_train = datasets.MNIST(root='/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/CVAE/', train=True, download=True)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, num_workers=4, shuffle=True)

    trainer.fit(model, train_loader)
    return trainer.callback_metrics["train_loss"].item()


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print("Best trial:", study.best_trial.params)

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：convnet_prize 
@File    ：model_training_V2.py
@Author  ：Junru Jin
@Date    ：5/23/24 7:48 PM 
'''
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import optuna

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import os
# Seed for reproducibility
torch.manual_seed(0)

from torchmetrics.image.fid import FrechetInceptionDistance
def calculate_fid(generator, device, data_loader, latent_dim, num_samples=1000):
    fid = FrechetInceptionDistance(feature=64, )  # Initialize FID computation
    generator.eval()

    # Collect real and generated images
    real_count = 0
    fake_count = 0

    with torch.no_grad():
        for real_imgs, _ in data_loader:
            # Convert real images to 3 channels, resize, scale to [0, 255], and convert to uint8
            real_imgs_3ch = real_imgs.repeat(1, 3, 1, 1)
            real_imgs_resized = torch.nn.functional.interpolate(real_imgs_3ch, size=(299, 299), mode='bilinear', align_corners=False)
            real_imgs_uint8 = (real_imgs_resized * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)

            fid.update(real_imgs_uint8.to("cpu"), real=True)
            real_count += real_imgs.size(0)

            # Generate fake images
            z = torch.randn(real_imgs.size(0), latent_dim, device=device)
            labels = torch.randint(0, 10, (real_imgs.size(0),), device=device)
            gen_imgs = generator(z, labels)
            gen_imgs_3ch = gen_imgs.repeat(1, 3, 1, 1)
            gen_imgs_resized = torch.nn.functional.interpolate(gen_imgs_3ch, size=(299, 299), mode='bilinear', align_corners=False)
            gen_imgs_uint8 = (gen_imgs_resized * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)

            fid.update(gen_imgs_uint8.to(("cpu")), real=False)
            fake_count += gen_imgs.size(0)

            if real_count >= num_samples and fake_count >= num_samples:
                break

    fid_score = fid.compute()
    return fid_score


def visualize_generated_images(generator, device, latent_dim, lr, result, num_images=100, ):
    generator.eval()  # Set the generator to evaluation mode
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        labels = [i % 10 for i in range(num_images)]
        labels = torch.tensor(labels, device=device)
        generated_images = generator(z, labels).cpu().detach().numpy()

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    for ax, img in zip(axes.flatten(), generated_images):
        ax.axis('off')
        ax.set_adjustable('box')
        ax.imshow(img.squeeze(), cmap='gray')

    plt.suptitle(f'Generated Images, latent_dim={latent_dim}, lr={lr:.5f}, FID={result:.4f}')
    if not os.path.exists(f'/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/CVAE/imgs/dim_{latent_dim}_lr_{lr:.5f}'):
        os.makedirs(f'/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/CVAE/imgs/dim_{latent_dim}_lr_{lr:.5f}')
    plt.savefig(f'/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/CVAE/imgs/dim_{latent_dim}_lr_{lr:.5f}/Generate.png')
    plt.show()

def visualize_latent_space(generator, device, latent_dim, lr, result, num_points=1000):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_points, latent_dim, device=device)
        labels = torch.randint(0, 10, (num_points,), device=device)
        generated_images = generator(z, labels).view(num_points, -1).cpu().detach().numpy()

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(generated_images)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels.cpu().numpy(), cmap='tab10', edgecolor='k')
    plt.title(f'PCA, latent_dim={latent_dim}, lr={lr:.5f}, FID={result:.4f}')
    plt.colorbar()
    plt.savefig(
        f'/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/CVAE/imgs/dim_{latent_dim}_lr_{lr:.5f}/PCA.png')
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(generated_images)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels.cpu().numpy(), cmap='tab10', edgecolor='k')
    # plt.title('t-SNE of Generated Images')
    plt.title(f't-SNE, latent_dim={latent_dim}, lr={lr:.5f}, FID={result:.4f}')
    plt.colorbar()
    plt.savefig(
        f'/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/CVAE/imgs/dim_{latent_dim}_lr_{lr:.5f}/TSNE.png')
    plt.show()

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.init_size = 7  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(num_classes + 784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

def train(generator, discriminator, device, data_loader, optimizer_G, optimizer_D, criterion, latent_dim, num_epochs=10):
    generator.train()
    discriminator.train()
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]
            valid = torch.ones(batch_size, 1, device=device, requires_grad=False)
            fake = torch.zeros(batch_size, 1, device=device, requires_grad=False)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, 10, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs, labels), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

def train_and_evaluate(generator, discriminator, device, data_loader, optimizer_G, optimizer_D, criterion, latent_dim, num_epochs=10):
    generator.train()
    discriminator.train()
    total_g_loss = 0
    total_d_loss = 0
    total_d_correct = 0  # Total discriminator correct predictions
    total_d_count = 0    # Total discriminator predictions

    for epoch in range(num_epochs):
        for imgs, labels in data_loader:
            batch_size = imgs.size(0)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Training Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, 10, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()
            total_g_loss += g_loss.item()

            # Training Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs, labels), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            total_d_loss += d_loss.item()

            # Evaluate discriminator accuracy
            with torch.no_grad():
                preds_real = discriminator(real_imgs, labels) >= 0.5
                preds_fake = discriminator(gen_imgs.detach(), gen_labels) < 0.5
                total_d_correct += (preds_real.sum() + preds_fake.sum()).item()
                total_d_count += 2 * batch_size

    fid_score = calculate_fid(generator, device, data_loader, latent_dim, num_samples=2000)
    # print(f"FID Score: {fid_score}")

    # Combine generator and discriminator loss, adjust based on discriminator accuracy
    return fid_score




def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    latent_dim = trial.suggest_int("latent_dim", 2, 256)

    # Model, optimizer, and dataloader setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = torchvision.datasets.MNIST('/archive/bioinformatics/Zhou_lab/shared/jjin/project/convnet_prize/CVAE/', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Training
    # train(generator, discriminator, device, data_loader, optimizer_G, optimizer_D, criterion, latent_dim, num_epochs=10)
    # In the objective function, replace the evaluation call
    result = train_and_evaluate(generator, discriminator, device, data_loader, optimizer_G, optimizer_D, criterion,
                                latent_dim, num_epochs=20)
    visualize_generated_images(generator, device, latent_dim, lr, result)
    visualize_latent_space(generator, device, latent_dim, lr, result)
    return result

study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="minimize", study_name="cgan_mnist")
study.optimize(objective, n_trials=20)
print(study.best_trial)

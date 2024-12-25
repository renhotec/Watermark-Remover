import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg19
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
from scripts.dataset import WatermarkDataset
from scripts.model import EnhancedGenerator, Discriminator
import cv2
import numpy as np
from scripts.test import test_model
import keyboard  # Add this import at the top of the file
import matplotlib.pyplot as plt
import sys


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.model = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.model(input)
        target_features = self.model(target)
        return F.l1_loss(input_features, target_features)

class ColorArtifactLoss(nn.Module):
    def __init__(self, threshold=0.1):
        super(ColorArtifactLoss, self).__init__()
        self.threshold = threshold

    def forward(self, generated, target):
        # 计算颜色差异
        color_diff = torch.abs(generated - target)
        # 只对颜色差异超过阈值的部分进行惩罚
        artifact_mask = (color_diff > self.threshold).float()
        return F.l1_loss(generated * artifact_mask, target * artifact_mask)

class BlackRegionLoss(nn.Module):
    def __init__(self, threshold=0.1):
        super(BlackRegionLoss, self).__init__()
        self.threshold = threshold

    def forward(self, generated, target):
        mask = (target < self.threshold).float()
        return F.l1_loss(generated * mask, target * mask)

class EdgePreservingLoss(nn.Module):
    def __init__(self):
        super(EdgePreservingLoss, self).__init__()

    def forward(self, generated, target):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(generated.device)
        sobel_y = sobel_x.transpose(2, 3)
        sobel_x = sobel_x.repeat(3, 1, 1, 1)
        sobel_y = sobel_y.repeat(3, 1, 1, 1)

        generated_edge_x = F.conv2d(generated, sobel_x, padding=1, groups=3)
        generated_edge_y = F.conv2d(generated, sobel_y, padding=1, groups=3)
        target_edge_x = F.conv2d(target, sobel_x, padding=1, groups=3)
        target_edge_y = F.conv2d(target, sobel_y, padding=1, groups=3)

        edge_loss = F.l1_loss(generated_edge_x, target_edge_x) + F.l1_loss(generated_edge_y, target_edge_y)
        return edge_loss


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = kernel.repeat(3, 1, 1, 1)
        self.kernel = self.kernel.cuda() if torch.cuda.is_available() else self.kernel

    def forward(self, generated, target):
        laplacian_generated = F.conv2d(generated, self.kernel, padding=1, groups=3)
        laplacian_target = F.conv2d(target, self.kernel, padding=1, groups=3)
        return F.l1_loss(laplacian_generated, laplacian_target)


class ColorConsistencyLoss(nn.Module):
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()

    def forward(self, generated, target):
        return F.l1_loss(generated.mean(dim=(2, 3)), target.mean(dim=(2, 3)))

learning_rate = 1e-5

def light_area_loss(generated, target, threshold=0.8):
    mask = (target.mean(dim=1, keepdim=True) > threshold).float()
    return F.l1_loss(generated * mask, target * mask)

def train_model(epochs=100, dir="", pretrained_pth=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if dir == "":
        dir = "images"
    dataset = WatermarkDataset(root_dir="data/train", clean=dir, watermarked=dir + "-watermarked")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    generator = EnhancedGenerator().to(device)
    if pretrained_pth != "":
        generator.load_state_dict(torch.load(pretrained_pth), strict=False)
        print(f"Loaded pretrained model from {pretrained_pth}")
    discriminator = Discriminator().to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss().to(device)
    criterion_l1 = nn.L1Loss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    edge_loss_fn = EdgePreservingLoss().to(device)
    laplacian_loss_fn = LaplacianLoss().to(device)
    color_loss_fn = ColorConsistencyLoss().to(device)

    scaler = GradScaler(device=device.type)
    total_start_time = time.time()

    paused = False
    stop_training = False

    def toggle_pause():
        nonlocal paused
        paused = not paused
        if paused:
            print("Training paused. Press 'ctrl+alt+shift+o' to continue.")
        else:
            print("Training resumed.")

    def stop_training_fn():
        nonlocal stop_training
        stop_training = True
        print("Training stopped. Exiting the loop.")

    keyboard.add_hotkey('ctrl+alt+shift+p', toggle_pause)
    keyboard.add_hotkey('ctrl+alt+shift+o', toggle_pause)
    keyboard.add_hotkey('ctrl+alt+shift+x', stop_training_fn)
        
    losses = []

    best_performance = 100
    for epoch in range(epochs):
        if stop_training:
            break
        epoch_start_time = time.time()

        while paused:
            time.sleep(1)

        for batch_idx, (watermarked_images, clean_images) in enumerate(dataloader):
            if stop_training:
                break
            while paused:
                time.sleep(1)

            watermarked_images, clean_images = watermarked_images.to(device), clean_images.to(device)

            # 训练判别器
            d_optimizer.zero_grad()
            with autocast(device_type=device.type):
                real_outputs = discriminator(watermarked_images, clean_images)
                fake_clean_images = generator(watermarked_images)
                fake_outputs = discriminator(watermarked_images, fake_clean_images.detach())

                d_loss_real = criterion_gan(real_outputs, torch.ones_like(real_outputs, device=device))
                d_loss_fake = criterion_gan(fake_outputs, torch.zeros_like(fake_outputs, device=device))
                d_loss = (d_loss_real + d_loss_fake) / 2

            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)
            scaler.update()

            # 训练生成器
            g_optimizer.zero_grad()
            with autocast(device_type=device.type):
                fake_clean_images = generator(watermarked_images)
                fake_outputs = discriminator(watermarked_images, fake_clean_images)

                g_gan_loss = criterion_gan(fake_outputs, torch.ones_like(fake_outputs, device=device))
                g_l1_loss = criterion_l1(fake_clean_images, clean_images)
                g_perceptual_loss = perceptual_loss(fake_clean_images, clean_images)
                g_edge_loss = edge_loss_fn(fake_clean_images, clean_images)
                g_laplacian_loss = laplacian_loss_fn(fake_clean_images, clean_images)
                g_color_loss = color_loss_fn(fake_clean_images, clean_images)

                color_artifact_loss_fn = ColorArtifactLoss().to(device)

                # 综合损失函数
                g_loss = (
                    g_gan_loss +
                    2 * g_l1_loss +
                    1 * g_perceptual_loss +
                    3 * g_edge_loss +
                    1 * g_laplacian_loss +
                    1 * g_color_loss +
                    1 * color_artifact_loss_fn(fake_clean_images, clean_images) 
                )
                g_loss += 2 * light_area_loss(fake_clean_images, clean_images)

            scaler.scale(g_loss).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(g_optimizer)
            scaler.update()

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Time: {epoch_duration:.2f}s")

        losses.append(g_loss.item())

        if g_loss.item() < best_performance:
            best_performance = g_loss.item()
            torch.save(generator.state_dict(), f"models/generator_epoch_best_{epoch + 1}.pth")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch + 1}.pth")
            test_model(model_path=f"models/generator_epoch_{epoch + 1}.pth", input_image_path="data/train/test/", output_folder="data/train/outputs/")
            with open("losses.txt", "w") as f:
                for loss in losses:
                    f.write(f"{loss}\n")

    total_duration = time.time() - total_start_time
    print(f"Training completed in {total_duration:.2f}s.")

    with open("losses.txt", "r") as f:
        losses = [float(line.strip()) for line in f]

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Generator Loss")
    plt.title("Generator Loss Over Epochs")
    plt.grid()
    plt.show()
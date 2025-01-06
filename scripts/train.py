import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from torch.amp import autocast, GradScaler
from scripts.dataset import WatermarkDataset
from scripts.model import EnhancedGenerator, Discriminator
import cv2
import numpy as np
from scripts.test import test_model
import keyboard
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

class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, generated, target):
        gen_fft = torch.fft.rfft2(generated, norm="ortho")
        tgt_fft = torch.fft.rfft2(target, norm="ortho")
        # 只关注高频区域
        gen_high = torch.abs(gen_fft[:, :, -10:, -10:])
        tgt_high = torch.abs(tgt_fft[:, :, -10:, -10:])
        return F.l1_loss(gen_high, tgt_high)

learning_rate = 1e-5

def light_area_loss(generated, target, threshold=0.8):
    mask = (target.mean(dim=1, keepdim=True) > threshold).float()
    return F.l1_loss(generated * mask, target * mask)

def color_masked_loss(generated, target, mask_color):
    # 生成一个针对目标颜色区域的掩码
    mask = (target.mean(dim=1, keepdim=True) - mask_color).abs() < 0.2
    return F.l1_loss(generated * mask.float(), target * mask.float())

def color_contrast_loss(generated, target, target_color):
    # 创建针对目标颜色的掩码
    mask = torch.abs(target.mean(dim=1, keepdim=True) - target_color) < 0.2
    return F.l1_loss(generated * mask.float(), target * mask.float())

# 增强红色信道的学习
def red_channel_loss(generated, target):
    # 提取红色通道
    gen_red = generated[:, 0, :, :]  # 生成图像的红色通道
    tgt_red = target[:, 0, :, :]    # 目标图像的红色通道
    # 计算红色通道的 L1 损失
    return F.l1_loss(gen_red, tgt_red)

# 增加尺度感知损失函数
def scale_aware_loss(generated, target):
    # 计算图像梯度来检测边缘
    generated_grad = torch.abs(generated[:, :, 1:, :] - generated[:, :, :-1, :])[:, :, :, :-1] + \
                    torch.abs(generated[:, :, :, 1:] - generated[:, :, :, :-1])[:, :, :-1, :]
    target_grad = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])[:, :, :, :-1] + \
                  torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])[:, :, :-1, :]
    return F.l1_loss(generated_grad, target_grad)

def train_model(epochs=100, dir="", pretrained_pth=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集
    dataset = WatermarkDataset(root_dir="data/train", clean=dir, watermarked=dir + "-watermarked")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 

    # 模型加载
    generator = EnhancedGenerator().to(device)
    if pretrained_pth:
        generator.load_state_dict(torch.load(pretrained_pth), strict=False)
        print(f"Loaded pretrained model from {pretrained_pth}")
    discriminator = Discriminator().to(device)

    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # 损失函数
    criterion_gan = nn.BCEWithLogitsLoss().to(device)
    criterion_l1 = nn.L1Loss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    color_loss_fn = ColorConsistencyLoss().to(device)
    edge_loss_fn = EdgePreservingLoss().to(device)

    scaler = GradScaler()
    total_start_time = time.time()

    losses = []
    best_performance = float("inf")
    epoch = 0

    while epoch < epochs:
        epoch_start_time = time.time()

        for batch_idx, (watermarked_images, clean_images) in enumerate(dataloader):
            watermarked_images, clean_images = watermarked_images.to(device), clean_images.to(device)

            # Train Discriminator
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

            # Train Generator
            g_optimizer.zero_grad()
            with autocast(device_type=device.type):
                fake_clean_images = generator(watermarked_images)
                fake_outputs = discriminator(watermarked_images, fake_clean_images)

                g_gan_loss = criterion_gan(fake_outputs, torch.ones_like(fake_outputs, device=device))
                g_l1_loss = criterion_l1(fake_clean_images, clean_images)
                g_perceptual_loss = perceptual_loss(fake_clean_images, clean_images)
                g_color_loss = color_loss_fn(fake_clean_images, clean_images)
                g_edge_loss = edge_loss_fn(fake_clean_images, clean_images)

                # 针对水印区域的颜色对比损失
                def watermark_contrast_loss(generated, target, watermark_mask):
                    return F.l1_loss(generated * watermark_mask, target * watermark_mask)

                # 针对蓝色水印区域生成掩码
                def generate_watermark_mask(target_image, color_range=(0.4, 0.6)):
                    mask = ((target_image.mean(dim=1, keepdim=True) > color_range[0]) & 
                            (target_image.mean(dim=1, keepdim=True) < color_range[1])).float()
                    return mask

                watermark_mask = generate_watermark_mask(clean_images)
                g_watermark_loss = watermark_contrast_loss(fake_clean_images, clean_images, watermark_mask)

                # 背景一致性损失
                def background_loss(generated, target, threshold=0.8):
                    mask = (target.mean(dim=1, keepdim=True) > threshold).float()
                    return F.l1_loss(generated * mask, target * mask)

                g_background_loss = background_loss(fake_clean_images, clean_images)
                 # 动态调整对抗损失的权重
                gan_loss_weight = 2 if epoch < 50 else 1

                # 最终生成器损失
                g_loss = (
                    gan_loss_weight * g_gan_loss +    # 提高对抗损失权重
                    5.0 * g_l1_loss +                
                    4.0 * g_perceptual_loss +        
                    6.0 * g_color_loss +             # 保持颜色一致性损失的权重
                    3.0 * g_edge_loss  +
                    1.0 * g_watermark_loss +             # 水印区域对比损失
                    6.0 * g_background_loss              # 背景一致性损失          
                )

            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Time: {epoch_duration:.2f}s")

        losses.append(g_loss.item())

        if g_loss.item() < best_performance:
            best_performance = g_loss.item()
            torch.save(generator.state_dict(), f"models/generator_epoch_best_{epoch + 1}.pth")

        if epoch_duration > 0:
            output_step = max(1, int(360 / epoch_duration))

        if (epoch + 1) % output_step == 0:
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch + 1}.pth")
            test_model(model_path=f"models/generator_epoch_{epoch + 1}.pth", input_image_path="data/train/test/", output_folder="data/train/outputs/")
            with open("losses.txt", "w") as f:
                for loss in losses:
                    f.write(f"{loss}\n")

        # Ask if more epochs are needed
        if epoch == epochs - 1:
            plt.plot(losses)
            plt.ylabel("Generator Loss")
            plt.title("Generator Loss Over Epochs")
            plt.grid()
            plt.show()
            additional_epochs = input("Do you want to add more epochs? If yes, please enter the number of additional epochs (or press Enter to finish): ")
            if additional_epochs.isdigit():
                epochs += int(additional_epochs)
                print(f"Training will continue for {additional_epochs} more epochs.")

        epoch += 1

    total_duration = time.time() - total_start_time
    print(f"Training completed in {total_duration:.2f}s.")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Generator Loss")
    plt.title("Generator Loss Over Epochs")
    plt.grid()
    plt.show()
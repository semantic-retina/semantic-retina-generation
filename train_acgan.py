import json
import time
from pathlib import Path
from typing import List

import torch
import torch.cuda
import torchvision.transforms as transforms
from torch import nn
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from src.data.common import Labels, get_labels
from src.data.datasets.combined import CombinedDataset
from src.logger.acgan import ACGANLogger, ACGANMetrics
from src.losses.hinge_loss import HingeLoss
from src.losses.wasserstein_loss import WassersteinLoss
from src.models.acgan import Discriminator
from src.models.acgan.generator import Generator
from src.models.acgan.weights import weights_init_normal
from src.options.acgan.train import get_args
from src.transforms import custom_affine, custom_noise, custom_rotate
from src.transforms.discriminator_transform import DiscriminatorTransform
from src.utils.device import get_device
from src.utils.sample import sample_gan
from src.utils.seed import set_seed
from src.utils.time import format_seconds


def save_models(generator, discriminator, output_chkpt_path, suffix):
    torch.save(generator.state_dict(), output_chkpt_path / f"generator_{suffix}.pth")
    torch.save(
        discriminator.state_dict(), output_chkpt_path / f"discriminator_{suffix}.pth"
    )


def train(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_path: Path,
    lr_g: float,
    lr_d: float,
    n_epochs: int,
    n_critic: int,
    n_gen: int,
    clip_gradient: bool,
    label_smoothing: bool,
    latent_dim: int,
    n_classes: int,
    sample_interval: int,
    chkpt_interval: int,
    logger: ACGANLogger,
    log_step: int,
    lesions: List[str],
):
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    image_path = output_path / "images"
    image_path.mkdir(exist_ok=True, parents=True)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)
    )

    # Discriminator transforms augment every image the discriminator sees for ADA.
    # We aim for a target ADA r value of 0.6.
    initial_ada_p = 0.0
    d_transform = DiscriminatorTransform(
        0.6,
        [
            custom_rotate.Rotate(initial_ada_p),
            custom_noise.GaussianNoise(initial_ada_p, 0.0, 1.0),
        ],
        max_p=0.85,
    )

    hinge_loss = HingeLoss()
    wasserstein_loss = WassersteinLoss()
    aux_loss = NLLLoss()

    valid_val = 1
    if label_smoothing:
        valid_val *= 0.9

    which_labels = sorted([Labels[l] for l in lesions], key=lambda x: x.value)
    for epoch in range(n_epochs):
        d_acc = 0.0
        d_loss = 0.0
        g_loss = 0.0
        D_x = 0.0
        D_G_z = 0.0

        for i, batch in enumerate(dataloader):
            imgs = batch["label"]
            imgs = get_labels(which_labels, imgs)
            labels = batch["grade"]

            # Track ada_r for this minibatch. Ensure that this is always a float,
            # and not accidentally a Tensor by calling `.item()`. Otherwise, issues will
            # arise from inadvertently storing intermediate results between epochs.
            ada_r = 0.0

            step = epoch * len(dataloader) + i + 1

            batch_size = imgs.shape[0]

            labels = labels.long().to(device)
            real_imgs = imgs.float().to(device)

            # Train discriminator.
            for _ in range(n_critic):
                optimizer_D.zero_grad()

                # Loss for real images.
                d_real_imgs = d_transform(real_imgs)
                real_pred, real_aux = discriminator(d_real_imgs)

                ada_r = torch.mean(torch.sign(real_pred)).item()

                # Sample noise and labels as generator input.
                z = torch.randn((batch_size, latent_dim), device=device)
                with torch.no_grad():
                    gen_labels = torch.randint(
                        0, n_classes, (batch_size,), device=device
                    )
                    gen_imgs = generator(z, gen_labels)
                    d_gen_imgs = d_transform(gen_imgs)

                # Loss for fake images.
                fake_pred, fake_aux = discriminator(d_gen_imgs.detach())

                # Total discriminator loss.
                d_loss = (
                    hinge_loss.dis_loss(real_pred, fake_pred)
                    + aux_loss(fake_aux, gen_labels)
                    + aux_loss(real_aux, labels)
                )
                d_loss.backward()

                if clip_gradient:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.01)

                optimizer_D.step()

                # Calculate discriminator accuracy.
                pred_aux = torch.argmax(torch.cat((real_aux, fake_aux), dim=0), dim=1)
                true_aux = torch.cat((labels, gen_labels), dim=0)
                d_acc = torch.mean(torch.eq(pred_aux, true_aux).float()).item()

                D_x = real_pred.mean().item()
                D_G_z = fake_pred.mean().item()

            # Train generator.
            for _ in range(n_gen):
                optimizer_G.zero_grad()

                # Sample noise and labels as generator input.
                z = torch.randn((batch_size, latent_dim), device=device)
                gen_labels = torch.randint(0, n_classes, (batch_size,), device=device)
                gen_imgs = generator(z, gen_labels)
                d_gen_imgs = d_transform(gen_imgs)

                # Loss measures generator's ability to fool the discriminator.
                validity, pred_label = discriminator(d_gen_imgs)
                g_loss = wasserstein_loss.gen_loss(validity)
                g_loss.backward()
                optimizer_G.step()

            d_transform.update(ada_r)

            if step % log_step == 0:
                metrics = ACGANMetrics(
                    step,
                    epoch,
                    g_loss.item(),
                    d_loss.item(),
                    d_acc,
                    D_x,
                    D_G_z,
                    d_transform.p,
                    ada_r,
                    real_imgs,
                    gen_imgs,
                )
                logger.log(metrics)

        if epoch % sample_interval == 0:
            sample_gan(image_path, generator, device, latent_dim, n_classes, epoch)

        if epoch % chkpt_interval == 0:
            save_models(generator, discriminator, checkpoint_path, epoch)

    save_models(generator, discriminator, checkpoint_path, "final")


def main():
    opt = get_args()
    print(opt)
    set_seed(213)
    n_classes = 5

    device = get_device()

    output_path = Path(opt.output_dir) / "acgan" / opt.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Save options.
    with open(output_path / "opt.json", "w") as f:
        json.dump(vars(opt), f, indent=4)

    n_channels = len(opt.lesions) + 1

    generator = Generator(n_channels, opt.img_size, n_classes, opt.latent_dim)
    discriminator = Discriminator(n_channels, opt.img_size, n_classes)

    generator.to(device)
    discriminator.to(device)

    # Initialize weights.
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    transform = transforms.Compose(
        [
            transforms.Resize(opt.img_size, InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ],
    )
    dataset = CombinedDataset(
        return_inst=False, return_image=False, label_transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
    )

    logger = ACGANLogger(opt.name, opt.n_epochs, opt.tensorboard)

    start_time = time.time()

    train(
        generator,
        discriminator,
        dataloader,
        device,
        output_path,
        opt.lr_g,
        opt.lr_d,
        opt.n_epochs,
        opt.n_critic,
        opt.n_gen,
        opt.clip_gradient,
        opt.label_smoothing,
        opt.latent_dim,
        n_classes,
        opt.sample_interval,
        opt.chkpt_interval,
        logger,
        opt.log_step,
        opt.lesions,
    )

    logger.close()

    end_time = time.time()
    # Throw away fractional seconds since we don't need that level of precision.
    execution_time = int(end_time - start_time)
    print(f"Finished in {format_seconds(execution_time)}")


if __name__ == "__main__":
    main()

import copy
import json
import time
from pathlib import Path
from typing import List

import torch
import torch.cuda
import torchvision.transforms as transforms
from torch import Tensor, nn
from torch.nn import DataParallel
from torch.nn.functional import avg_pool2d, interpolate
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from src.data.common import Labels, get_labels
from src.data.datasets.combined import CombinedDataset
from src.data.datasets.copy_paste import CopyPasteDataset
from src.logger.progan import ProGANLogger, ProGANMetrics
from src.losses.hinge_loss import HingeLoss
from src.losses.wasserstein_loss import WassersteinLoss
from src.models.progan.custom_layers import update_average
from src.models.progan.networks import Discriminator, Generator
from src.options.progan.train import get_args
from src.transforms import probabilistic
from src.transforms.discriminator import DiscriminatorTransform
from src.utils.device import get_device
from src.utils.sample import colour_labels
from src.utils.seed import set_seed
from src.utils.time import format_seconds


def save_models(generator, discriminator, output_chkpt_path, suffix):
    torch.save(generator.state_dict(), output_chkpt_path / f"generator_{suffix}.pth")
    torch.save(
        discriminator.state_dict(), output_chkpt_path / f"discriminator_{suffix}.pth"
    )


def progressive_downsample_batch(real_batch, final_depth, current_depth, alpha):

    down_sample_factor = int(2 ** (final_depth - current_depth))
    prior_downsample_factor = int(2 ** (final_depth - current_depth + 1))

    ds_real_samples = avg_pool2d(
        real_batch, kernel_size=down_sample_factor, stride=down_sample_factor
    )

    if current_depth > 2:
        prior_ds_real_samples = interpolate(
            avg_pool2d(
                real_batch,
                kernel_size=prior_downsample_factor,
                stride=prior_downsample_factor,
            ),
            scale_factor=2,
        )
    else:
        prior_ds_real_samples = ds_real_samples

    real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)
    return real_samples


def sample_progan(
    output_path: Path,
    generator: nn.Module,
    fixed_noise: Tensor,
    labels: Tensor,
    n_row: int,
    epoch: int,
    current_depth: int,
    alpha: int,
):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    with torch.no_grad():
        gen_imgs = generator(fixed_noise, labels, current_depth, alpha)

    coloured_val = colour_labels(gen_imgs)

    output_file = output_path / f"{epoch:05}.png"
    print(f"Saved image to {output_file}")
    save_image(coloured_val.data, output_file, nrow=n_row, normalize=False, pad_value=1)


def train(
    generator: Generator,
    discriminator: Discriminator,
    dataset: Dataset,
    batch_sizes: List[int],
    device: torch.device,
    output_path: Path,
    lr_g: float,
    lr_d: float,
    depth_epochs: List[int],
    n_critic: int,
    n_gen: int,
    clip_gradient: bool,
    label_smoothing: bool,
    latent_dim: int,
    n_classes: int,
    sample_interval: int,
    chkpt_interval: int,
    logger: ProGANLogger,
    log_step: int,
    lesions: List[str],
    use_ada: bool,
    start_depth: int,
    final_depth: int,
):
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    image_path = output_path / "images"
    image_path.mkdir(exist_ok=True, parents=True)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.99))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr_d, betas=(0.0, 0.99)
    )

    # Discriminator transforms augment every image the discriminator sees for ADA.
    # We aim for a target ADA r value of 0.6.
    n_channels = len(lesions) + 1
    initial_ada_p = 0.0
    d_transform = DiscriminatorTransform(
        0.6,
        [
            probabilistic.Rotate(initial_ada_p),
            probabilistic.Affine(initial_ada_p, n_channels),
            probabilistic.GaussianNoise(initial_ada_p, 0.0, 1.0),
        ],
        max_p=0.85,
    )
    if not use_ada:
        d_transform = DiscriminatorTransform(0, [], max_p=0.0)

    hinge_loss = HingeLoss()
    wasserstein_loss = WassersteinLoss()

    valid_val = 1
    if label_smoothing:
        valid_val *= 0.9

    step = 0

    which_labels = sorted([Labels[l] for l in lesions], key=lambda x: x.value)

    use_ema = True
    if use_ema:
        gen_shadow = copy.deepcopy(generator)
        update_average(gen_shadow, generator, beta=0)

    fade_in_percentage = 50.0
    global_epoch = 0

    if n_classes is None:
        n_rows = 5
        fixed_labels = None
        fixed_noise = torch.randn((n_rows ** 2, latent_dim), device=device)
    else:
        n_rows = n_classes
        fixed_labels = torch.tensor(
            [num for _ in range(n_rows) for num in range(n_rows)], device=device
        )
        fixed_noise = torch.randn((n_rows ** 2, latent_dim - n_classes), device=device)

    for current_depth in range(start_depth, final_depth + 1):
        current_res = int(2 ** current_depth)
        depth_list_index = current_depth - start_depth
        ticker = 1

        n_epochs = depth_epochs[depth_list_index]
        for epoch in range(n_epochs):
            d_acc = 0.0
            d_loss = 0.0
            g_loss = 0.0
            D_x = 0.0
            D_G_z = 0.0

            batch_size = batch_sizes[depth_list_index]
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )

            total_batches = len(dataloader)

            fader_point = int((fade_in_percentage / 100) * n_epochs * total_batches)

            for i, batch in enumerate(dataloader):
                real_imgs = batch["label"]
                real_imgs = get_labels(which_labels, real_imgs)
                real_labels = batch["grade"]

                alpha = ticker / fader_point if ticker <= fader_point else 1

                # Track ada_r for this minibatch. Ensure that this is always a float,
                # and not accidentally a Tensor by calling `.item()`. Otherwise, issues will
                # arise from inadvertently storing intermediate results between epochs.
                ada_r = 0.0

                batch_size = real_imgs.shape[0]

                real_labels = real_labels.long().to(device)
                real_imgs = real_imgs.float().to(device)

                # Loss for real images.
                real_imgs_ds = progressive_downsample_batch(
                    real_imgs, final_depth, current_depth, alpha
                )

                # Train discriminator.
                for _ in range(n_critic):
                    optimizer_D.zero_grad()

                    d_real_imgs = d_transform(real_imgs_ds)
                    real_pred = discriminator(
                        d_real_imgs, current_depth, alpha, labels=real_labels
                    )

                    ada_r = torch.mean(torch.sign(real_pred)).item()

                    with torch.no_grad():
                        if n_classes is not None:
                            # Sample noise and labels as generator input.
                            z = torch.randn(
                                (batch_size, latent_dim - n_classes), device=device
                            )
                            gen_labels = torch.randint(
                                0, n_classes, (batch_size,), device=device
                            )
                        else:
                            z = torch.randn((batch_size, latent_dim), device=device)
                            gen_labels = None

                        gen_imgs = generator(z, gen_labels, current_depth, alpha)
                        d_gen_imgs = d_transform(gen_imgs)

                    # Loss for fake images.
                    fake_pred = discriminator(
                        d_gen_imgs.detach(), current_depth, alpha, labels=gen_labels
                    )

                    # Total discriminator loss.
                    d_loss = hinge_loss.dis_loss(real_pred, fake_pred)
                    d_loss.backward()

                    if clip_gradient:
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.01)

                    optimizer_D.step()

                    D_x = real_pred.mean().item()
                    D_G_z = fake_pred.mean().item()

                if use_ema:
                    update_average(gen_shadow, generator, 0.999)

                # Train generator.
                for _ in range(n_gen):
                    optimizer_G.zero_grad()

                    if n_classes is not None:
                        # Sample noise and labels as generator input.
                        z = torch.randn(
                            (batch_size, latent_dim - n_classes), device=device
                        )
                        gen_labels = torch.randint(
                            0, n_classes, (batch_size,), device=device
                        )
                    else:
                        z = torch.randn((batch_size, latent_dim), device=device)
                        gen_labels = None

                    gen_imgs = generator(z, gen_labels, current_depth, alpha)
                    d_gen_imgs = d_transform(gen_imgs)

                    # Loss measures generator's ability to fool the discriminator.
                    validity = discriminator(
                        d_gen_imgs, current_depth, alpha, labels=real_labels
                    )
                    g_loss = wasserstein_loss.gen_loss(validity)
                    g_loss.backward()
                    optimizer_G.step()

                d_transform.update(ada_r)

                ticker += 1
                step += 1

                if step % log_step == 0:
                    metrics = ProGANMetrics(
                        step,
                        global_epoch,
                        current_depth,
                        g_loss.item(),
                        d_loss.item(),
                        d_acc,
                        D_x,
                        D_G_z,
                        d_transform.p,
                        ada_r,
                        real_imgs_ds,
                        gen_imgs,
                    )
                    logger.log(metrics)

            if global_epoch % sample_interval == 0:
                sample_progan(
                    image_path,
                    generator,
                    fixed_noise,
                    fixed_labels,
                    n_rows,
                    global_epoch,
                    current_depth,
                    alpha,
                )

            if global_epoch % chkpt_interval == 0:
                save_models(generator, discriminator, checkpoint_path, global_epoch)

            global_epoch += 1

        sample_progan(
            image_path,
            generator,
            fixed_noise,
            fixed_labels,
            n_rows,
            global_epoch,
            current_depth,
            alpha,
        )

    save_models(generator, discriminator, checkpoint_path, "final")


def main():
    opt = get_args()
    print(opt)
    set_seed(213)

    if opt.conditional:
        n_classes = 5
    else:
        n_classes = None

    device = get_device()
    print(f"Device count: {torch.cuda.device_count()}")

    output_path = Path(opt.output_dir) / "progan" / opt.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Save options.
    with open(output_path / "opt.json", "w") as f:
        json.dump(vars(opt), f, indent=4)

    n_channels = len(opt.lesions) + 1

    # The initial resolution will be 2**2 = 4.
    # The final resolution will be 2**9 = 512.
    start_depth = 2
    final_depth = 9
    generator = Generator(
        depth=final_depth,
        n_channels=n_channels,
        latent_size=opt.latent_dim,
        n_classes=n_classes,
    )
    discriminator = Discriminator(
        depth=final_depth,
        num_channels=n_channels,
        latent_size=opt.latent_dim,
        n_classes=n_classes,
    )

    generator = DataParallel(generator)
    discriminator = DataParallel(discriminator)

    generator.to(device)
    discriminator.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(opt.img_size, InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ],
    )
    dataset = CombinedDataset(
        return_inst=False,
        return_image=False,
        return_transformed=False,
        label_transform=transform,
    )
    if opt.use_copypaste:
        synthetic_dataset = CopyPasteDataset(label_transform=transform)
        dataset = ConcatDataset((dataset, synthetic_dataset))

    depth_epochs = [20, 40, 60, 80, 100, 120, 140, 160]
    batch_sizes = [512, 256, 128, 64, 32, 16, 8, 4]

    n_stages = final_depth - start_depth + 1
    assert len(depth_epochs) == n_stages
    assert len(batch_sizes) == n_stages

    total_epochs = sum(depth_epochs)
    logger = ProGANLogger(opt.name, total_epochs, opt.tensorboard)

    start_time = time.time()

    train(
        generator,
        discriminator,
        dataset,
        batch_sizes,
        device,
        output_path,
        opt.lr_g,
        opt.lr_d,
        depth_epochs,
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
        opt.use_ada,
        start_depth,
        final_depth,
    )

    logger.close()

    end_time = time.time()
    # Throw away fractional seconds since we don't need that level of precision.
    execution_time = int(end_time - start_time)
    print(f"Finished in {format_seconds(execution_time)}")


if __name__ == "__main__":
    main()

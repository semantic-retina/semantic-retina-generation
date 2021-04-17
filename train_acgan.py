from pathlib import Path

import torch
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.combined import CombinedDataset
from src.data.common import get_label_semantics
from src.losses.gan_loss import GANLoss
from src.models.acgan import Discriminator
from src.models.acgan.generator import Generator
from src.models.acgan.weights import weights_init_normal
from src.options.acgan.train import get_args
from src.transforms import custom_affine, custom_noise, custom_rotate
from src.transforms.discriminator_transform import DiscriminatorTransform
from src.utils.device import get_device
from src.utils.sample import sample_gan
from src.utils.seed import set_seed


def save_models(generator, discriminator, output_chkpt_path, suffix):
    torch.save(generator.state_dict(), output_chkpt_path / f"generator_{suffix}.pth")
    torch.save(
        discriminator.state_dict(), output_chkpt_path / f"discriminator_{suffix}.pth"
    )


def get_max_p(epoch: int) -> float:
    thresh = 1000
    if epoch <= thresh:
        return 1.0
    return max(0.1, 1.0 / (epoch - thresh))


def train(
    name: str,
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_path: Path,
    lr: float,
    n_epochs: int,
    n_critic: int,
    n_gen: int,
    clip_gradient: bool,
    label_smoothing: bool,
    latent_dim: int,
    n_classes: int,
    n_channels: int,
    sample_interval: int,
    chkpt_interval: int,
):
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    image_path = output_path / "images"
    image_path.mkdir(exist_ok=True, parents=True)

    loss = GANLoss(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
    )

    # Discriminator transforms augments every image the discriminator sees.
    d_transform_p = 0.0
    d_transform = DiscriminatorTransform(
        [
            custom_rotate.Rotate(d_transform_p),
            custom_affine.Affine(d_transform_p),
            custom_noise.GaussianNoise(d_transform_p, 0.5, 1.0),
        ]
    )

    writer = SummaryWriter(comment=f"_{name}")
    log_step = 10

    for epoch in range(n_epochs):
        d_loss = 0
        d_acc = 0
        d_valid_acc = 0
        g_loss = 0

        for i, batch in enumerate(dataloader):
            imgs = batch["label"]
            imgs = get_label_semantics(imgs)[:, [0, 1, 4], :, :]
            labels = batch["grade"]

            # Track r_t for this minibatch.
            r_t = 0

            step = epoch * len(dataloader) + i + 1

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
            fake = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

            if label_smoothing:
                valid *= 0.9

            labels = labels.long().to(device)
            real_imgs = imgs.float().to(device)

            # Train discriminator.
            for _ in range(n_critic):
                optimizer_D.zero_grad()

                real_imgs = d_transform(real_imgs)

                # Loss for real images.
                real_pred, real_aux = discriminator(real_imgs)
                d_real_loss = loss.loss_function(real_pred, valid, real_aux, labels)

                r_t = torch.mean(real_pred)

                # Sample noise and labels as generator input.
                z = torch.randn((batch_size, latent_dim), device=device)
                with torch.no_grad():
                    gen_labels = torch.randint(
                        0, n_classes, (batch_size,), device=device
                    )
                    gen_imgs = generator(z, gen_labels)
                    gen_imgs = d_transform(gen_imgs)

                # Loss for fake images.
                fake_pred, fake_aux = discriminator(gen_imgs.detach())
                d_fake_loss = loss.loss_function(fake_pred, fake, fake_aux, gen_labels)

                # Total discriminator loss.
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()

                if clip_gradient:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.01)

                optimizer_D.step()

                # Calculate discriminator accuracy.
                pred_aux = torch.argmax(torch.cat((real_aux, fake_aux), dim=0), dim=1)
                true_aux = torch.cat((labels, gen_labels), dim=0)
                d_acc = torch.mean(torch.eq(pred_aux, true_aux).float())

                # Calculate discriminator valid accuracy.
                pred_validity = torch.cat((real_pred, fake_pred), dim=0)
                pred_validity = torch.gt(pred_validity, 0.5).long()
                true_validity = torch.cat((valid, fake), dim=0)
                d_valid_acc = torch.mean(torch.eq(pred_validity, true_validity).float())

            # Train generator.
            for _ in range(n_gen):
                # Sample noise and labels as generator input.
                z = torch.randn((batch_size, latent_dim), device=device)
                gen_labels = torch.randint(0, n_classes, (batch_size,), device=device)
                gen_imgs = generator(z, gen_labels)
                gen_imgs = d_transform(gen_imgs)

                optimizer_G.zero_grad()

                # Loss measures generator's ability to fool the discriminator.
                validity, pred_label = discriminator(gen_imgs)
                g_loss = loss.loss_function(validity, valid, pred_label, gen_labels)

                g_loss.backward()
                optimizer_G.step()

            if step % log_step == 0:
                writer.add_scalars(
                    "scalars",
                    {
                        "Loss/generator": g_loss.item(),
                        "Loss/discriminator": d_loss.item(),
                        "accuracy": d_acc,
                    },
                    step,
                )
                print(
                    f"[Epoch {epoch}/{n_epochs}] "
                    f"[D loss: {d_loss.item():.4f}, acc: {100 * d_acc:.2f}%, valid_acc: {100 * d_valid_acc:.2f}%] "
                    f"[G loss: {g_loss.item():.4f}] "
                    f"[p: {d_transform_p:.2f}] "
                    f"[r_t: {r_t:.2f}]",
                    flush=True,
                )

            if r_t > 0.6:
                d_transform_p += 0.05
            if r_t < 0.6:
                d_transform_p -= 0.05

            # Clamp.
            max_p = get_max_p(epoch)
            if d_transform_p < 0:
                d_transform_p = 0
            if d_transform_p > max_p:
                d_transform_p = max_p
            d_transform.update_p(d_transform_p)

        if epoch % sample_interval == 0:
            sample_gan(
                image_path, generator, device, n_channels, latent_dim, n_classes, epoch
            )

        if epoch % chkpt_interval == 0:
            save_models(generator, discriminator, checkpoint_path, epoch)

    save_models(generator, discriminator, checkpoint_path, "final")

    writer.close()


def main():
    opt = get_args()
    print(opt)
    set_seed(213)

    device = get_device()

    output_path = Path(opt.output_dir) / "acgan" / opt.name
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "opt.txt", "w") as f:
        print(opt, file=f)

    generator = Generator(opt.n_channels, opt.img_size, opt.n_classes, opt.latent_dim)
    discriminator = Discriminator(opt.n_channels, opt.img_size, opt.n_classes)

    generator.to(device)
    discriminator.to(device)

    # Initialize weights.
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    transform = transforms.Compose(
        [
            transforms.Resize(opt.img_size, Image.NEAREST),
            transforms.RandomHorizontalFlip(),
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
    )

    train(
        opt.name,
        generator,
        discriminator,
        dataloader,
        device,
        output_path,
        opt.lr,
        opt.n_epochs,
        opt.n_critic,
        opt.n_gen,
        opt.clip_gradient,
        opt.label_smoothing,
        opt.latent_dim,
        opt.n_classes,
        opt.n_channels,
        opt.sample_interval,
        opt.chkpt_interval,
    )


if __name__ == "__main__":
    main()

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
import config
import utils
import train_utils
from torchvision.transforms import transforms
from dataset.pannuke import PanNuke
from discriminator_model import Discriminator
from generator_model import Generator

transform_train = transforms.Compose([
    transforms.FiveCrop(256),
    transforms.Lambda(lambda crops: torch.stack([transforms.RandomHorizontalFlip()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([transforms.RandomVerticalFlip()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([utils.RandomRotate90()(crop) for crop in crops])),
])
transform_test = transforms.Compose([transforms.RandomCrop(1024)])
WANDB_PROJECT_NAME = "unitopatho-generative"


def main():
    # Models
    num_classes = len(PanNuke.labels())
    disc = Discriminator(in_channels=3 + num_classes).to(config.DEVICE)
    gen = Generator(in_channels=num_classes, features=64).to(config.DEVICE)

    # Optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))

    # Losses
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    print(disc)
    print(gen)

    # Weight initialization
    disc.apply(utils.init_weights)
    gen.apply(utils.init_weights)

    # Load checkpoints from wandb.
    if config.LOAD_MODEL:
        wandb_run_path = "daviderubi/pix2pixgan/1l0hnnnn"  # The wandb run is daviderubi/pix2pixgan/upbeat-river-42
        train_utils.wandb_load_model(wandb_run_path, "disc.pth", disc, opt_disc, config.LEARNING_RATE, config.DEVICE, True)
        train_utils.wandb_load_model(wandb_run_path, "gen.pth", gen, opt_gen, config.LEARNING_RATE, config.DEVICE, True)

    # Load dataset.
    train_dataset, test_dataset = train_utils.load_dataset_UTP(transform_train, transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE)

    # GradScaler
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if config.LOG_WANDB:
        # Get some images from testloader. Every epoch we will log the generated images for this batch on wandb.
        test_batch_im, test_batch_masks = train_utils.wandb_get_images_to_log(test_loader)
        img_masks_test = [PanNuke.get_img_mask(mask) for mask in test_batch_masks]
        wandb.log({"Reals": wandb.Image(torchvision.utils.make_grid(test_batch_im), caption="Reals"),
                   "Masks": wandb.Image(torchvision.utils.make_grid(img_masks_test), caption="Masks")})

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        g_adv_loss, g_l1_loss, d_loss = train_utils.train_epoch(disc, gen, train_loader, opt_disc, opt_gen, l1_loss,
                                                                bce, g_scaler, d_scaler, config.DEVICE)

        # Save checkpoint
        if config.SAVE_MODEL and (epoch + 1) % 10 == 0:
            print(f"Saving checkpoint at epoch {epoch + 1}...")
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            if config.LOG_WANDB:
                wandb.save(config.CHECKPOINT_GEN)
                wandb.save(config.CHECKPOINT_DISC)

        # Log generated images after the epoch.
        if config.LOG_WANDB:
            train_utils.wandb_log_epoch(gen, test_batch_masks, g_adv_loss, g_l1_loss, d_loss)

    # Save generator and discriminator models.
    utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
    utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

    if config.LOG_WANDB:
        train_utils.wandb_log_generated_images(gen, test_loader, batch_to_log=math.ceil(100 / config.BATCH_SIZE))
        wandb.finish()


if __name__ == "__main__":
    if config.LOG_WANDB:
        train_utils.wandb_init(config.WANDB_KEY_LOGIN, WANDB_PROJECT_NAME)

    print(f"Working on {config.DEVICE} device.")
    if "cuda" in str(config.DEVICE):
        for idx in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(idx))
    main()

# Tutorial DDP: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import config
import utils
import train_utils
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


def main(gpu):
    print(f"GPU #{gpu} started")
    # DDP
    world_size = config.NGPU * config.NUM_NODES
    nr = 0  # it is the rank of the current node. Now we use only one node
    rank = nr * config.NGPU + gpu
    utils.setup_ddp(rank, world_size)
    torch.cuda.set_device(gpu)
    is_master = rank == 0
    do_wandb_log = config.LOG_WANDB and is_master  # only master logs on wandb.
    if do_wandb_log:
        train_utils.wandb_init(config.WANDB_KEY_LOGIN, WANDB_PROJECT_NAME)

    # Load models
    num_classes = len(PanNuke.labels())
    disc = Discriminator(in_channels=3 + num_classes).cuda(gpu)
    gen = Generator(in_channels=num_classes, features=64).cuda(gpu)
    # Use SynchBatchNorm for Multi-GPU trainings
    disc = nn.SyncBatchNorm.convert_sync_batchnorm(disc)
    gen = nn.SyncBatchNorm.convert_sync_batchnorm(gen)
    if do_wandb_log:
        print(disc)
        print(gen)

    # DDP
    disc = nn.parallel.DistributedDataParallel(disc, device_ids=[gpu])
    gen = nn.parallel.DistributedDataParallel(gen, device_ids=[gpu])

    # Optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))

    # Losses
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Load checkpoints from wandb
    if config.LOAD_MODEL:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        wandb_run_path = "daviderubi/pix2pixgan/1l0hnnnn"  # The wandb run is daviderubi/pix2pixgan/upbeat-river-42
        train_utils.wandb_load_model(wandb_run_path, "disc.pth", disc, opt_disc, config.LEARNING_RATE, map_location)
        train_utils.wandb_load_model(wandb_run_path, "gen.pth", gen, opt_gen, config.LEARNING_RATE, map_location)

    # load dataset
    train_dataset, test_dataset = train_utils.load_dataset_UTP(transform_train, transform_test)
    # DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                               num_workers=config.NUM_WORKERS, sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                              num_workers=config.NUM_WORKERS, sampler=test_sampler)

    # grad_scaler
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if do_wandb_log:
        # Get some images from testloader. Every epoch we will log the generated images for this batch on wandb.
        test_batch_im, test_batch_masks = train_utils.wandb_get_images_to_log(test_loader)
        img_masks_test = [PanNuke.get_img_mask(mask) for mask in test_batch_masks]
        wandb.log({"Reals": wandb.Image(torchvision.utils.make_grid(test_batch_im), caption="Reals"),
                   "Masks": wandb.Image(torchvision.utils.make_grid(img_masks_test), caption="Masks")})

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        g_adv_loss, g_l1_loss, d_loss = train_utils.train_epoch(disc, gen, train_loader, opt_disc, opt_gen, l1_loss,
                                                                bce, g_scaler, d_scaler, gpu)

        # Save checkpoint.
        if config.SAVE_MODEL and (epoch + 1) % 10 == 0 and is_master:
            print(f"Saving checkpoint at epoch {epoch + 1}...")
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN, epoch=epoch + 1)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC, epoch=epoch + 1)
            if do_wandb_log:
                wandb.save(config.CHECKPOINT_GEN)
                wandb.save(config.CHECKPOINT_DISC)

        # Log generated images after the training epoch.
        if do_wandb_log:
            train_utils.wandb_log_epoch(gen, test_batch_masks, g_adv_loss, g_l1_loss, d_loss)

    # Save generator and discriminator models.
    if is_master:
        utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN, epoch=config.NUM_EPOCHS)
        utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC, epoch=config.NUM_EPOCHS)

    # Log on wandb some generated images.
    if do_wandb_log:
        train_utils.wandb_log_generated_images(gen, test_loader, batch_to_log=math.ceil(100 / config.BATCH_SIZE))
        wandb.finish()

    torch.distributed.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print(f"Working on {config.DEVICE} device.")
    if "cuda" in str(config.DEVICE):
        for idx in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(idx))

    # DistributedDataParallel
    mp.spawn(main, nprocs=config.NGPU, args=())

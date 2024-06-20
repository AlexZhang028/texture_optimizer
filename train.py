import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.optim as optim
from dataloader import get_dataloader
from model import TextOptiModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(args):
    epochs = args.epochs
    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    mesh_path = os.path.join(data_dir, "mesh", args.mesh_name)
    train_loader, val_loader, image_shape = get_dataloader(data_dir, batch_size=1, split_ratio=0.99, device=args.device)
    model = TextOptiModel(mesh_path, args.texture_size, image_shape, args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model.train()
    tb_writer = SummaryWriter(save_dir)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch: {epoch + 1}/{epochs}', total=len(train_loader))
        for i, data in enumerate(pbar):
            iteration = epoch * len(train_loader) + i + 1
            camera = data["camera"]
            gt_image = data["image"]

            iter_start.record()
            rendered_image = model(camera)
            loss = F.l1_loss(rendered_image, gt_image)
            loss.backward()

            iter_end.record()
            torch.cuda.synchronize()
            iter_time = iter_start.elapsed_time(iter_end)
            pbar.set_postfix({'Loss': loss.item()})
            tb_writer.add_scalar("Loss", loss.item(), iteration)
            tb_writer.add_scalar("Iteration Time", iter_time, iteration)
            tb_writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], iteration)
            optimizer.step()
            optimizer.zero_grad()
            if iteration % 500 == 0:
                validate(model, tb_writer, val_loader, iteration, epoch)
                scheduler.step()
                
        

    model.save_texture(os.path.join(save_dir, "texture.png"))

    

def validate(model, writer, val_loader, iteration, epoch):
    model.eval()
    imgs = torch.tensor([], device=model.device)
    gts = torch.tensor([], device=model.device)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            camera = data["camera"]
            gt_image = data["image"]
            rendered_image = model(camera)
            imgs = torch.cat((imgs, rendered_image.unsqueeze(0)), dim=0)
            gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
            if i == 10:
                break
        loss = F.l1_loss(imgs, gts)
        writer.add_scalar("Val Loss", loss.item(), iteration)
        writer.add_images("Rendered Images", imgs, iteration, dataformats="NHWC")
        writer.add_images("Ground Truth Images", gts, iteration, dataformats="NHWC")
        writer.add_image("Texture", model.texture_img.data.clamp_(0, 1), iteration, dataformats="HWC")
        # print(f"Epoch: {epoch+1}, validation Loss: {loss.item()}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for texture optimizer")
    parser.add_argument("--data_dir", type=str, help="Path to the training data directory")
    parser.add_argument("--save_dir", type=str, help="Path to the save directory")
    parser.add_argument("--mesh_name", type=str, help="Name of the mesh file")
    parser.add_argument("--camera_param_type", type=str, help="Type of camera parameterization", default="fov")
    parser.add_argument("--device", type=str, help="Device to run the training on", default="cuda")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.01)
    parser.add_argument("--texture_size", type=int, help="Size of the texture image", default=512)
    args = parser.parse_args()
    train(args)



    

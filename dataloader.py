import os

import cv2
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras
from torch.utils.data import DataLoader, Dataset
from utils import load_kitti_poses


def load_gt_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).float() / 255.0

    return image_tensor

class CustomDataset(Dataset):
    # Todo: optionally implement tran val split
    def __init__(self, data_dir, camera_param_type, device="cuda"): # camera_param_type: "fov" or "intrinsic"
        self.device = device
        images_dir = os.path.join(data_dir, "imgs")
        self.image_paths = os.listdir(images_dir)
        self.image_paths = [os.path.join(images_dir, i) for i in self.image_paths if i.endswith(".png")]
        self.image_paths.sort()
        self.camera_param_type = camera_param_type
        poses = load_kitti_poses(os.path.join(data_dir, "poses.txt"))
        self.R = torch.tensor(poses[:, :3, :3])
        self.T = torch.tensor(poses[:, :3, 3])
        assert len(self.image_paths) == len(self.R) == len(self.T), "Different number of images and poses"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        gt_image = load_gt_image(image_path).to(self.device)
        R = self.R[index].unsqueeze(0)
        T = self.T[index].unsqueeze(0)
        if self.camera_param_type == "fov":
            camera = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        elif self.camera_param_type == "intrinsic":
            # todo: implement intrinsic camera including loading intrinsic matrix
            camera = PerspectiveCameras(device=self.device, R=R, T=T)
        return {
            "image": gt_image,
            "camera": camera,
            "image_path": image_path,
            }

def collate_fn(batch):
    return batch[0]

def get_dataloader(data_dir, camera_param_type, batch_size=1, device="cuda"):
    image_path = os.path.join(data_dir, "imgs")
    assert os.path.exists(image_path), f"Image path {image_path} does not exist"
    image = cv2.imread(os.path.join(image_path, os.listdir(image_path)[0]))
    image_shape = (image.shape[0], image.shape[1])
    dataset = CustomDataset(data_dir, camera_param_type, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader, image_shape

if __name__ == "__main__":
    train_loader = get_dataloader("data/teapot", "fov", 1)
    for data in train_loader:
        print(data)
        break
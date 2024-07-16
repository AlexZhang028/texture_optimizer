import os
import json
import numpy as np
import cv2
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras
from pytorch3d.utils import cameras_from_opencv_projection
from torch.utils.data import DataLoader, Dataset
from utils import load_kitti_poses
from sklearn.model_selection import train_test_split


def load_gt_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).float() / 255.0

    return image_tensor

class CustomDataset(Dataset):
    def __init__(self, data_dir, split='train', split_ratio = 0.8, device="cuda"): # camera_param_type: "fov" or "intrinsic"
        self.device = device
        images_dir = os.path.join(data_dir, "imgs")
        camera_dir = os.listdir(images_dir)
        self.poses = load_kitti_poses(os.path.join(data_dir, "poses.txt"))

        if len(camera_dir) > 1:
            print("Multiple views detected, loading camera info from camera_infos.json")
            self.camera_info = json.load(open(os.path.join(data_dir, "camera_infos.json")))
            self.poses = np.linalg.inv(self.poses)
            self.camera_param_type = "intrinsic"
            num_images = {}
            self.image_paths = []
            for camera_name in camera_dir:
                num_images[camera_name] = len(os.listdir(os.path.join(images_dir, camera_name)))
                image_paths = [os.path.join(images_dir, camera_name, i) for i in os.listdir(os.path.join(images_dir, camera_name)) if i.endswith(".png") or i.endswith(".jpg") or i.endswith(".jpeg")]
                if os.path.exists(os.path.join(data_dir, "indices.npy")):
                    image_paths.sort()
                    moving_indices = np.load(os.path.join(data_dir, "indices.npy"))
                    image_paths = [image_paths[i] for i in moving_indices]
                self.image_paths += image_paths

        else:
            self.camera_param_type = "fov"
            images_dir = os.path.join(images_dir, camera_dir[0])
            self.image_paths = os.listdir(images_dir)
            self.image_paths = [os.path.join(images_dir, i) for i in self.image_paths if i.endswith(".png")]
            self.image_paths.sort()
            self.R = torch.tensor(self.poses[:, :3, :3])
            self.T = torch.tensor(self.poses[:, :3, 3])

        if split == 'val':
            self.image_paths = np.random.choice(self.image_paths, size=10, replace=False)
            




    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.camera_param_type == "fov":
            image_path = self.image_paths[index]
            gt_image = load_gt_image(image_path).to(self.device)
            image_idx = int(image_path.split("/")[-1].split(".")[0])
            R = self.R[image_idx].unsqueeze(0)
            T = self.T[image_idx].unsqueeze(0)
            camera = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        elif self.camera_param_type == "intrinsic":
            image_path = self.image_paths[index]
            gt_image = load_gt_image(image_path).to(self.device)
            mask_path = image_path.replace("imgs", "masks").split(".")[0] + ".npz"
            mask = np.load(mask_path)["arr_0"]
            camera_name = image_path.split("/")[-2]
            intrinsic = torch.tensor(self.camera_info[camera_name]["intrinsic"]).unsqueeze(0).to(self.device)
            transform = torch.tensor(self.camera_info[camera_name]["transform"]).unsqueeze(0).to(self.device).float()
            image_size = torch.tensor(gt_image.shape[-3:-1]).unsqueeze(0).to(self.device)
            image_idx = int(image_path.split("/")[-1].split(".")[0])
            final_transform = transform @ torch.tensor(self.poses[image_idx]).to(self.device).float()
            R = final_transform[:,:3, :3]
            T = final_transform[:,:3, 3]
            camera = cameras_from_opencv_projection(R=R.float(), 
                            tvec=T.float(), 
                            camera_matrix=intrinsic.float(), 
                            image_size=image_size.float())

        return {
            "image": gt_image,
            "mask": mask,
            "camera": camera,
            "image_path": image_path,
            }

def collate_fn(batch):
    return batch[0]

def get_dataloader(data_dir, batch_size=1, device="cuda", split_ratio=0.8):
    image_path = os.path.join(data_dir, "imgs")
    assert os.path.exists(image_path), f"Image path {image_path} does not exist"
    camera_name = os.listdir(image_path)[0]
    image_path = os.path.join(image_path, camera_name)
    image = cv2.imread(os.path.join(image_path, os.listdir(image_path)[0]))
    image_shape = (image.shape[0], image.shape[1])
    dataset_train = CustomDataset(data_dir, split='train', split_ratio=split_ratio, device=device)
    dataset_val = CustomDataset(data_dir, split='val', split_ratio=split_ratio, device=device)
    print("Train dataset size: ", len(dataset_train), "Val dataset size: ", len(dataset_val))
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader_train, dataloader_val, image_shape

if __name__ == "__main__":
    train_loader, val_loader, image_shape = get_dataloader("/home/zhang/hdd/data_texture_optimization/campus_garching", split_ratio=0.8)
    print(len(train_loader))
    print(len(val_loader))
    for data in train_loader:
        print(data)
        break
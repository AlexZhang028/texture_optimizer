import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.renderer import (MeshRasterizer, MeshRenderer,
                                RasterizationSettings, SoftPhongShader,
                                TexturesUV)
from pytorch3d.structures import Meshes


class TextOptiModel(nn.Module):
    def __init__(
            self,
            mesh_path: str,
            texture_size: int,
            render_image_size: tuple, #(H, W)
            device: str = "cuda",
        ):
        super().__init__()
        # Load the obj and ignore the texture and materials.
        self.device = device
        self.render_image_size = render_image_size
        self.texture_size = texture_size
        self.verts, self.faces, aux = load_obj(mesh_path, device=device)
        self.verts_uvs = aux.verts_uvs
        self.faces_uvs = self.faces.textures_idx

        self.texture_img = nn.Parameter(torch.zeros((self.texture_size, self.texture_size, 3)).to(device))

        self.raster_settings = RasterizationSettings(
            image_size=self.render_image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1,
            bin_size=100
        )
        texture = TexturesUV(maps=self.texture_img[None, ...], faces_uvs=self.faces_uvs[None, ...], verts_uvs=self.verts_uvs[None, ...])
        self.meshes = Meshes(verts=[self.verts], faces=[self.faces.verts_idx], textures=texture)

    def forward(self, cameras):
        # Update the mesh with the new vertices and faces.
        texture = TexturesUV(maps=self.texture_img[None, ...], faces_uvs=self.faces_uvs[None, ...], verts_uvs=self.verts_uvs[None, ...])
        self.meshes.textures = texture

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
            )
        )

        images = renderer(self.meshes)

        return images.squeeze()[:, :, :3]
    
    def save_texture(self, file_path: str):
        self.texture_img.data.clamp_(0, 1)
        texture_img = self.texture_img.detach().cpu().numpy()
        np.save(file_path.split(".")[0] + ".npy", texture_img)
        texture_img = (texture_img * 255).astype(np.uint8)
        cv2.imwrite(file_path, cv2.cvtColor(texture_img, cv2.COLOR_RGB2BGR))

    def save_mesh(self, file_path: str):
        pass
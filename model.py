import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.renderer import (MeshRasterizer, MeshRenderer,
                                RasterizationSettings, SoftPhongShader,
                                TexturesUV, AmbientLights)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import HardDepthShader, ShaderBase, BlendParams
from pytorch3d.renderer.blending import hard_rgb_blend, softmax_rgb_blend

class VertexColorShader(ShaderBase):
    def __init__(self, blend_soft=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.blend_soft = blend_soft

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        if self.blend_soft:
            return softmax_rgb_blend(texels, fragments, blend_params)
        else:
            return hard_rgb_blend(texels, fragments, blend_params)
        

class NeuralTexture(nn.Module):
    def __init__(self, tex_size, C):
        super(NeuralTexture, self).__init__()
        self.tex_size = tex_size
        self.C = C
        self.data = torch.nn.Parameter(torch.zeros(C, self.tex_size, self.tex_size, requires_grad=True), requires_grad=True)

    def normalize(self):
        with torch.no_grad():
            normalized_data = torch.clamp(self.data, -123.6800, 151.0610)
            self.data.copy_(normalized_data)

    def forward(self, x):
        self.normalize()
        batch_size = x.shape[0]
        y = F.grid_sample(self.data.repeat(batch_size, 1, 1, 1),
                          x,
                          mode="bilinear",
                          padding_mode='border',
                          align_corners=True)  # this treats (0,0) as origin and not as the center of the lower left texel
        return y

    def get_image(self):
        return self.data.permute(1, 2, 0)
        

class HierarchicalTexture(nn.Module):
    def __init__(self, texture_size: int, layers: int, device: str = "cuda"):
        super().__init__()
        self.hierachical_texture = [nn.Parameter(torch.zeros((3, texture_size // 2**i, texture_size // 2**i)).to(device)) for i in range(layers)]
        self.texture_size = texture_size
        self.device = device


    def layer_normalize(self, layer):
        normailzed_texture = torch.clamp(layer.data, 0, 1)
        layer.data.copy_(normailzed_texture)

    def layer_forward(self, layer, x):
        self.layer_normalize(layer)
        y = F.grid_sample(layer.data.repeat(x.shape[0], 1, 1, 1),
                          x, mode='bilinear', align_corners=True)
        return y
                          
    def forward(self, x):
        y = [self.layer_forward(layer, x) for layer in self.hierachical_texture]
        y = torch.stack(y)
        y = torch.sum(y, dim=0)
        return y
    
    def get_image(self):
        size_range = torch.arange(0, self.texture_size, dtype=torch.float) / (self.texture_size - 1.0) * 2.0 - 1.0
        v, u = torch.meshgrid(size_range, size_range)
        uv_id = torch.stack((u, v), 2)
        uv_id = uv_id[None, ...].to(self.device)
        uv_id = uv_id.type_as(self.hierachical_texture[0].data)

        texture = self.forward(uv_id)[0, 0:3, :, :]
        return texture.permute(1, 2, 0)



class TextOptiModel(nn.Module):
    def __init__(
            self,
            mesh_path: str,
            texture_size: int,
            render_image_size: tuple, #(H, W)
            device: str = "cuda",
            hierarchical_texture: bool = False,
            hierarchical_layers: int = 4
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

    def normalize_texture(self):
        with torch.no_grad():
            normailzed_texture = torch.clamp(self.texture_img.data, 0, 1)
            self.texture_img.data.copy_(normailzed_texture)

    def forward(self, cameras):
        # Update the mesh with the new vertices and faces.
        self.normalize_texture()
        texture = TexturesUV(maps=self.texture_img[None, ...], faces_uvs=self.faces_uvs[None, ...], verts_uvs=self.verts_uvs[None, ...])
        self.meshes.textures = texture

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings
            ),
            # shader=SoftPhongShader(
            #     device=self.device, 
            #     cameras=cameras,
            #     lights=AmbientLights(device=self.device)
            # )
            shader=VertexColorShader(blend_soft=False, device=self.device, cameras=cameras)
        )

        images = renderer(self.meshes)

        return images.squeeze()[:, :, :3]
    
    def save_texture(self, file_path: str):
        self.normalize_texture()
        texture_img = self.texture_img.detach().cpu().numpy()
        np.save(file_path.split(".")[0] + ".npy", texture_img)
        texture_img = (texture_img * 255).astype(np.uint8)
        cv2.imwrite(file_path, cv2.cvtColor(texture_img, cv2.COLOR_RGB2BGR))


    def save_mesh(self, file_path: str):
        pass
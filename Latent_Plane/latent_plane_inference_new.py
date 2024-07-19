# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from diffusers.models import  UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import logging
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class Unet2DConditionOutputFor3D(UNet2DConditionOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None
    image_occ: torch.FloatTensor = None
    ray_occ: torch.FloatTensor = None
    image_depth: torch.FloatTensor = None
    pixel_mask: torch.FloatTensor = None
    ray_rgb: torch.FloatTensor = None
    image_rgb: torch.FloatTensor = None
    voxel_sigma: torch.FloatTensor = None

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class LatentPlane_Coarse(nn.Module):
    def __init__(self, latent_embed_dim, time_embed_dim):
        super(LatentPlane_Coarse, self).__init__()
        self.grid_size = 64
        self.latent_embed_dim = latent_embed_dim
        self.latent_feature_dim = 32
        self.latent_feature_ln = nn.Linear(self.latent_embed_dim, self.latent_feature_dim)
        
        self.time_embed_dim = time_embed_dim
        self.time_feature_dim = 32
        self.time_feature_ln = nn.Linear(self.time_embed_dim, self.time_feature_dim)

        self.embed_num = 8
        elevation = 30
        azimuth_interval = 360.0 / self.embed_num
        camera_distance = 1.5

        multires= 10
        embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires - 1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
            }
        self.embedder_3D = Embedder(**embed_kwargs)

        elevation_rad = np.radians(elevation)
        azimuth_angles = np.arange(0, 360, azimuth_interval)
        azimuth_rads = np.radians(azimuth_angles)

        self.R = []
        self.T = []
        self.c2w = []
        camera_posi_list = []
        for azimuth_rad in azimuth_rads:
            x = camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = camera_distance * np.sin(elevation_rad)
            eye = np.array([x, y, z])
            at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            zaxis = at - eye
            zaxis = zaxis / np.linalg.norm(zaxis)
            xaxis = np.cross(-up, zaxis)
            xaxis = xaxis / np.linalg.norm(xaxis)
            yaxis = np.cross(zaxis, xaxis)

            _viewMatrix = np.array([
                [xaxis[0], yaxis[0], zaxis[0], eye[0]],
                [xaxis[1], yaxis[1], zaxis[1], eye[1]],
                [xaxis[2], yaxis[2], zaxis[2], eye[2]],
                [0       , 0       , 0       , 1     ]
            ])

            R = np.array([
                [xaxis[0], xaxis[1], xaxis[2]],
                [yaxis[0], yaxis[1], yaxis[2]],
                [zaxis[0], zaxis[1], zaxis[2]],
            ])
            T = np.array([
                -np.dot(xaxis, eye),
                -np.dot(yaxis, eye),
                -np.dot(zaxis, eye),
            ])
            
            self.R.append(torch.Tensor(R))
            self.T.append(torch.Tensor(T))
            self.c2w.append(torch.Tensor(_viewMatrix))
            camera_posi_list.append(torch.Tensor([x, y, z]))
        
        all_cam_posi = torch.stack(camera_posi_list, dim=0)
        self.camera_posi_feat = self.embedder_3D.embed(all_cam_posi)

        self.K_256 = torch.Tensor(\
            [[280.,      0.,    128.],
            [   0.,    280.,    128.],
            [   0.,      0.,      1.]]
            )
        
        self.n_samples = 32

        self.feature_dim = self.latent_feature_dim + 2 * self.embedder_3D.out_dim
        self.sigma_embeddings = nn.Parameter(torch.randn(self.latent_feature_dim))
        self.sigma_query_ln = nn.Linear(self.latent_feature_dim + self.time_feature_dim, self.feature_dim)

        self.transformer_layer_norm = nn.LayerNorm(self.feature_dim)
        self.transformer = nn.Sequential(*([Attention(query_dim=self.feature_dim, heads=8, dim_head=8, residual_connection=True)] * 2))
        
        self.sigma_net = nn.Linear(self.feature_dim, 1)

    @staticmethod
    def _get_signature_keys(obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}
        return expected_modules, optional_parameters
    
    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            fn_recursive_set_mem_eff(module)
    
    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/). When this
        option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
        up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        >>> # Workaround for not accepting attention shape using VAE for Flash Attention
        >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def gen_rays(self, pose, W, H):
        """
        Generate rays at world space given pose.
        """
        tx = torch.linspace(0, W - 1, int(W)).to(pose.device)
        ty = torch.linspace(0, H - 1, int(H)).to(pose.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()

        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         (pixels_y - self.K[1][2]) / self.K[1][1],
                         torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = p
        rays_v = torch.sum(rays_v[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o, rays_v
    
    def get_near_far_from_cube(self, rays_o, rays_d):
        # r.dir is unit direction vector of ray
        if (rays_d == 0).any():
            pass
        dirfrac = 1 / rays_d
        # lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
        bound_min = torch.tensor([-0.5, -0.5, -0.5]).to(rays_o.device)
        bound_max = torch.tensor([0.5, 0.5, 0.5]).to(rays_o.device)
        # r.org is origin of ray
        t_1 = (bound_min - rays_o) * dirfrac
        t_2 = (bound_max - rays_o) * dirfrac
        tmin = torch.max(torch.minimum(t_1, t_2), dim=1).values
        tmax = torch.min(torch.maximum(t_1, t_2), dim=1).values

        mask = torch.ones(rays_o.shape[0]).to(rays_o.device)
        mask[tmax < 0] = 0
        mask[tmin > tmax] = 0
        tmp = tmin.clone().detach()
        tmin = torch.where(mask > 0.5, tmin, tmax)
        tmax = torch.where(mask > 0.5, tmax, tmp)
        assert (tmin <= tmax).all()
        return tmin.unsqueeze(-1), tmax.unsqueeze(-1), mask > 0.5
    
    def volume_rendering(self, sigma, dists, color=None, z_vals=None):
        raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw)*dists)
        alpha = raw2alpha(sigma, dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        outputs = {}
        if color is not None:
            outputs["rgb_map"] = torch.sum(weights[...,None] * color, -2)  # [N_rays, 3]
        if z_vals is not None:
            outputs["depth_map"] = torch.sum(weights * z_vals, -1)
        outputs["occ_map"] = torch.sum(weights, -1)
        # outputs["weights"] = weights

        return outputs

    def interpolate_sigma(self, sigma_voxels, pts):
        batch_size = sigma_voxels.shape[0]
        rays_num, pts_per_ray = pts.shape[:2]
        normal_pts = (pts[None] * 2).flip(-1)
        normal_pts = torch.stack([normal_pts] * batch_size, dim=0)
        sigma = F.interpolate(sigma_voxels, normal_pts, mode="trilinear", align_corners=True)
        sigma = sigma.reshape(batch_size, 1, rays_num, pts_per_ray).permute(0, 2, 3, 1)
        return sigma
    
    def random_pixel_mask(self, image_shape, num_pixels):
        # 获取图像的高度和宽度
        height, width, _ = image_shape
        
        # 创建一个全零的掩码
        mask = torch.zeros((height, width), dtype=torch.bool)
        
        # 生成随机行和列的索引
        random_rows = torch.randint(0, height, (num_pixels,))
        random_cols = torch.randint(0, width, (num_pixels,))
        
        # 将相应位置的像素设置为True
        mask[random_rows, random_cols] = True
        
        return mask
    
    def project_embeddings_to_multi_view(self, voxels, pts):
        # pts [batch_size_3D, n_rays, n_samples, 3]
        batch_size_3D, voxel_grid_size, voxel_grid_size, voxel_grid_size, embed_dim = voxels.shape
        # voxels = voxels.reshape(batch_size, self.grid_size, self.grid_size, self.grid_size, -1)

        normal_pts = (pts * 2).flip(-1)
        normal_pts = normal_pts[:, None]

        sample = F.grid_sample(voxels.permute(0, 4, 1, 2, 3), normal_pts, mode="bilinear", align_corners=True, padding_mode="border")
        sample = sample.permute(0, 2, 3, 4, 1).squeeze(1)
        return sample

    def forward(self, sample, temb, embed_mask=None, render_camera=None, render_rays_num=0, coarse_pixel_mask=None):
        output = {}
        output_sample = sample.clone().detach()
        batch_size, latent_embed_dim, H, W = sample.shape
        assert latent_embed_dim == self.latent_embed_dim, "Wrong latent embedding dim."
        batch_size_3D = batch_size // self.embed_num
        sample = sample.reshape(batch_size_3D, self.embed_num, latent_embed_dim, H, W)
        sample = sample.permute(0, 1, 3, 4, 2)
        sample = self.latent_feature_ln(sample)
        sample = sample.permute(0, 1, 4, 2, 3)

        batch_size, time_embed_dim = temb.shape
        assert time_embed_dim == self.time_embed_dim, "Wrong time embedding dim."
        temb = temb.reshape(batch_size_3D, self.embed_num, time_embed_dim)
        temb = self.time_feature_ln(temb)

        batch_size_2D = embed_mask.int().sum().item()
        # embed_mask_indices = torch.nonzero(embed_mask).squeeze()

        if embed_mask is not None and embed_mask.float().sum() > 0:            
            reso_down = 256 / H
            self.K = self.K_256 / reso_down
            self.K[:, 2] -= 0.5
            self.K[2, 2] = 1.0
            
            pt_list = []
            dist_list = []
            z_val_list = []
            for cam_id in range(self.embed_num):
                if not embed_mask[cam_id]:
                    continue
                # generate rays
                camera = self.c2w[cam_id].to(sample.device)
                rays_o, rays_d = self.gen_rays(camera, W, H)
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                near, far, _ = self.get_near_far_from_cube(rays_o, rays_d)

                # sample points in each ray
                z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(rays_o.device)
                z_vals = near + (far - near) * z_vals[None, :]
                dists = z_vals[..., 1:] - z_vals[..., :-1]
                dists = torch.cat([dists, dists[...,-1:]], -1)
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
                pt_list.append(pts)
                dist_list.append(dists)
                z_val_list.append(z_vals)
            pts = torch.cat(pt_list, dim=0)
            pts = torch.stack([pts] * batch_size_3D, dim=0)
            dists = torch.cat(dist_list, dim=0)
            dists = torch.stack([dists] * batch_size_3D, dim=0)
            z_vals = torch.cat(z_val_list, dim=0)
            z_vals = torch.stack([z_vals] * batch_size_3D, dim=0)

            # project points to uv-coor to get features
            pts_feature_list = []
            for embed_id in range(self.embed_num):
                uv = (pts @ self.R[embed_id].T.to(sample.device) + self.T[embed_id].to(sample.device)) 
                uv = uv @ self.K.T.to(sample.device)
                uv = uv[..., :2] / uv[..., 2:]
                uv = uv / (256.0 // reso_down - 1.0) * 2 - 1
                pts_feature = F.grid_sample(sample[:, embed_id], uv.to(sample.dtype), mode="bilinear", align_corners=True, padding_mode="border")
                pts_feature = pts_feature.permute(0, 2, 3, 1)
                pts_feature_list.append(pts_feature)
            pts_features = torch.stack(pts_feature_list, dim=-2)
            
            # go through the network to get values
            rays_num = H * W
            # resi_ray_features = pts_features.reshape(batch_size_3D, batch_size_2D, rays_num, self.n_samples, self.embed_num, latent_embed_dim)
            pts_features = pts_features.reshape(batch_size_3D * batch_size_2D * rays_num * self.n_samples, self.embed_num, self.latent_feature_dim)
            camera_embed = self.camera_posi_feat[None].repeat(batch_size_3D * batch_size_2D * rays_num * self.n_samples, 1, 1).to(pts_features.device, dtype=pts_features.dtype)
            xyz_embed = self.embedder_3D.embed(pts)[:, :, :, None, :].repeat(1, 1, 1, self.embed_num, 1).reshape(batch_size_3D * batch_size_2D * rays_num * self.n_samples, self.embed_num, self.embedder_3D.out_dim)

            pts_features = torch.cat([pts_features, camera_embed, xyz_embed.to(dtype=pts_features.dtype)], dim=-1)

            sigma_timestep_embeddings = torch.cat([self.sigma_embeddings[None].repeat(temb.shape[0], 1), temb[:, 0]], dim=-1)
            sigma_timestep_embeddings = self.sigma_query_ln(sigma_timestep_embeddings)

            pts_features = torch.cat([sigma_timestep_embeddings[:, None, :].repeat(1, batch_size_2D * rays_num * self.n_samples, 1).reshape(batch_size_3D * batch_size_2D * rays_num * self.n_samples, 1, self.feature_dim), pts_features], dim=1)

            # pts_features = self.transformer_layer_norm(pts_features)
            # pts_features = pts_features.reshape(batch_size_3D, batch_size_2D, H * W * self.n_samples, self.embed_num + 1, self.feature_dim)
            pts_features = self.transformer_layer_norm(pts_features)
            block_features_list = []
            for block in range((pts_features.shape[0] - 1) // 32768 + 1):
                if not block == ((pts_features.shape[0] - 1) // 32768):
                    # print(1)
                    block_features = self.transformer(pts_features[block * 32768: (block + 1) * 32768])
                else:
                    # exit(0)
                    block_features = self.transformer(pts_features[block * 32768:])
                block_features_list.append(block_features)
            add_feature = torch.cat(block_features_list, dim=0)
                
            sigma_feature = add_feature[..., 0, :]
            sigma = self.sigma_net(sigma_feature)
            render_out = self.volume_rendering(sigma.squeeze(1).reshape(-1, self.n_samples), dists.reshape(-1, self.n_samples))
            occ_map  = render_out["occ_map"]
            # feature_map = feature_map.reshape(batch_size_3D * batch_size_2D, H, W, latent_embed_dim).permute(0, 3, 1, 2)
            occ_map = occ_map.reshape(batch_size_3D * batch_size_2D, H, W, 1).permute(0, 3, 1, 2)
            # depth_map = depth_map.reshape(batch_size_3D * batch_size_2D, H, W, 1).permute(0, 3, 1, 2)
            # output_sample[torch.cat([embed_mask] * batch_size_3D, dim=0)] = (output_sample[torch.cat([embed_mask] * batch_size_3D, dim=0)] * (1 - occ_map) + feature_map).to(output_sample.dtype)
            # output["sample"] = output_sample
            output["occ_map"] = occ_map
            # output["depth_map"] = depth_map
            # output["weight_map"] = render_out["weights"]
            output["sigma"] = sigma
        else:
            output["sample"] = None
            output["occ_map"] = None
            # output["sigma_embeddings_map"] = None
            output["sigma"] = None
        
        if render_camera is not None and (render_rays_num > 0 or coarse_pixel_mask is not None):
            assert batch_size_3D == 1
            reso_down = 256 / H
            self.K = self.K_256 / 1.0
            self.K[:, 2] -= 0.5
            self.K[2, 2] = 1.0
            rays_o, rays_d = self.gen_rays(render_camera[0], 256, 256)
            if coarse_pixel_mask is None:
                pixel_mask = self.random_pixel_mask(rays_o.shape, render_rays_num)
            else:
                pixel_mask = coarse_pixel_mask

            rays_o = rays_o[pixel_mask]
            rays_d = rays_d[pixel_mask]
            near, far, _ = self.get_near_far_from_cube(rays_o, rays_d)

            # sample points in each ray
            all_z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(rays_o.device)
            all_z_vals = near + (far - near) * all_z_vals[None, :]
            all_dists = all_z_vals[..., 1:] - all_z_vals[..., :-1]
            all_dists = torch.cat([all_dists, all_dists[...,-1:]], -1)
            all_pts = rays_o[:, None, :] + rays_d[:, None, :] * all_z_vals[..., :, None]
            
            all_ray_num = all_pts.shape[0]
            chunk_size = 2048
            # project points to uv-coor to get features
            chunk_occ_rays_list = []
            chunk_weight_rays_list = []
            chunk_sigma_embeddings_rays_list = []
            for chunk_id in range((all_ray_num - 1) // chunk_size + 1):
                if chunk_id < (all_ray_num - 1) // chunk_size:
                    pts = all_pts[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
                    dists = all_dists[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
                else:
                    pts = all_pts[chunk_id*chunk_size:]
                    dists = all_dists[chunk_id*chunk_size:]
                ray_num = pts.shape[0]
                pts_feature_list = []
                self.K = self.K_256 / reso_down
                self.K[:, 2] -= 0.5
                self.K[2, 2] = 1.0
                for embed_id in range(self.embed_num):
                    uv = (pts @ self.R[embed_id].T.to(sample.device) + self.T[embed_id].to(sample.device)) 
                    uv = uv @ self.K.T.to(sample.device)
                    uv = uv[..., :2] / uv[..., 2:]
                    uv = uv / (256.0 // reso_down - 1.0) * 2 - 1
                    uv = torch.stack([uv] * batch_size_3D, dim=0)
                    pts_feature = F.grid_sample(sample[:, embed_id], uv.to(sample.dtype), mode="bilinear", align_corners=True, padding_mode="border")
                    pts_feature = pts_feature.permute(0, 2, 3, 1)
                    pts_feature_list.append(pts_feature)

                pts_features = torch.stack(pts_feature_list, dim=-2)
                pts_features = pts_features.reshape(ray_num * self.n_samples, self.embed_num, self.latent_feature_dim)
                camera_embed = self.camera_posi_feat[None].repeat(ray_num * self.n_samples, 1, 1).to(sample.device)

                xyz_embed = self.embedder_3D.embed(pts)[:, :, None, :].repeat(1, 1, self.embed_num, 1).reshape(ray_num * self.n_samples, self.embed_num, self.embedder_3D.out_dim)
                pts_features = torch.cat([pts_features, camera_embed, xyz_embed], dim=-1)

                sigma_timestep_embeddings = torch.cat([self.sigma_embeddings, temb[0, 0]], dim=-1)
                sigma_timestep_embeddings = self.sigma_query_ln(sigma_timestep_embeddings)
                pts_features = torch.cat([sigma_timestep_embeddings[None, None, :].repeat(ray_num * self.n_samples, 1, 1), pts_features], dim=1)

                pts_features = self.transformer_layer_norm(pts_features)
                block_features_list = []
                for block in range((pts_features.shape[0] - 1) // 32768 + 1):
                    if not block == ((pts_features.shape[0] - 1) // 32768):
                        block_features = self.transformer(pts_features[block * 32768: (block + 1) * 32768])
                    else:
                        block_features = self.transformer(pts_features[block * 32768:])
                    block_features_list.append(block_features)
                pts_features = torch.cat(block_features_list, dim=0)

                    
                sigma_feature = pts_features[..., 0, :]
                sigma = self.sigma_net(sigma_feature)
                sigma = sigma[..., :1]

                # volume rendering with latent values as RGB
                render_out = self.volume_rendering(sigma.squeeze(1).reshape(-1, self.n_samples), dists.reshape(-1, self.n_samples))
                chunk_occ_rays_list.append(render_out["occ_map"])
                chunk_weight_rays_list.append(render_out["weights"])
                chunk_sigma_embeddings_rays_list.append(sigma_feature)
            output["occ_rays"] = torch.cat(chunk_occ_rays_list, dim=0)
            output["pixel_mask"] = pixel_mask
            output["weight_rays"] = torch.cat(chunk_weight_rays_list, dim=0)
            output["sigma_embeddings_rays"] = torch.cat(chunk_sigma_embeddings_rays_list, dim=0)
        else:
            output["occ_rays"] = None
            output["pixel_mask"] = None
            output["weight_rays"] = None
            output["sigma_embeddings_rays"] = None
        
        voxel_features, voxel_pts = self.unproject_embeddings_from_multi_view(sample, reso_down)
        # ori_pts_features = voxel_features.reshape(-, self.image_num, embed_dim)
        voxel_sample = voxel_features.reshape(batch_size_3D * self.grid_size * self.grid_size * self.grid_size, self.embed_num, self.latent_feature_dim)
        # sample = self.fc1(self.sample_layer_norm(sample))
        camera_embed = self.camera_posi_feat[None].repeat(voxel_sample.shape[0], 1, 1).to(sample.device)
        xyz_embed = self.embedder_3D.embed(voxel_pts)[None, :, :, :, None, :].repeat(batch_size_3D, 1, 1, 1, self.embed_num, 1).reshape(batch_size_3D * self.grid_size * self.grid_size * self.grid_size, self.embed_num, self.embedder_3D.out_dim)

        voxel_sample = torch.cat([voxel_sample, camera_embed, xyz_embed], dim=-1)

        sigma_timestep_embeddings = torch.cat([self.sigma_embeddings[None].repeat(temb.shape[0], 1), temb[:, 0]], dim=-1)
        sigma_timestep_embeddings = self.sigma_query_ln(sigma_timestep_embeddings)

        voxel_sample = torch.cat([sigma_timestep_embeddings[:, None, :].repeat(1, self.grid_size * self.grid_size * self.grid_size, 1).reshape(batch_size_3D * self.grid_size * self.grid_size * self.grid_size, 1, sigma_timestep_embeddings.shape[-1]), voxel_sample], dim=1)
        voxel_sample = self.transformer_layer_norm(voxel_sample)
        
        voxel_block_features_list = []
        voxel_block_size = 16384
        for voxel_block in range((voxel_sample.shape[0] - 1) // voxel_block_size + 1):
            if not voxel_block == ((voxel_sample.shape[0] - 1) // voxel_block_size):
                voxel_block_features = self.transformer(voxel_sample[voxel_block * voxel_block_size: (voxel_block + 1) * voxel_block_size])
            else:
                voxel_block_features = self.transformer(voxel_sample[voxel_block * voxel_block_size:])
            voxel_block_features_list.append(voxel_block_features)
        voxel_sample = torch.cat(voxel_block_features_list, dim=0)
            
        voxel_sigma_feature = voxel_sample[..., 0, :]
        voxel_sigma = self.sigma_net(voxel_sigma_feature)
        voxel_sigma = voxel_sigma.reshape(batch_size_3D, self.grid_size, self.grid_size, self.grid_size, 1)
        output["voxel_sigma"] = voxel_sigma

        return output

    def unproject_embeddings_from_multi_view(self, sample, reso_down):
        batch_size_3D = sample.shape[0]
        x = torch.linspace(-0.5, 0.5, self.grid_size).to(sample.device)
        y = torch.linspace(-0.5, 0.5, self.grid_size).to(sample.device)
        z = torch.linspace(-0.5, 0.5, self.grid_size).to(sample.device)
        xv, yv, zv = torch.meshgrid(x, y, z)
        voxels = torch.stack([xv, yv, zv], dim=-1)
        voxels[0][0][0] = torch.Tensor([0, 0, 0])
        voxels_features = []
        for i in range(self.embed_num):
            uv = (voxels @ self.R[i].T.to(sample.device) + self.T[i].to(sample.device)) 
            uv = uv @ self.K.T.to(sample.device)
            uv = uv[..., :2] / uv[..., 2:]
            uv = uv.reshape(self.grid_size, self.grid_size*self.grid_size, 2) / (256.0 // reso_down - 1) * 2 - 1
            uv = torch.stack([uv] * batch_size_3D, dim=0)
            voxels_feature = F.grid_sample(sample[:, i], uv, padding_mode= "border", mode="bilinear", align_corners=True)
            voxels_feature = voxels_feature.permute(0, 2, 3, 1).reshape(batch_size_3D, self.grid_size, self.grid_size, self.grid_size, -1)
            voxels_features.append(voxels_feature)
        
        voxels_features = torch.stack(voxels_features, dim=-2)
        return voxels_features, voxels

class LatentPlane_Fine(nn.Module):
    def __init__(self, latent_embed_dim, time_embed_dim, xt_latent_embed_dim=None):
        super(LatentPlane_Fine, self).__init__()
        self.latent_embed_dim = latent_embed_dim
        # self.xt_latent_embed_dim = xt_latent_embed_dim or latent_embed_dim
        # self.embed_dim = self.latent_embed_dim + self.xt_latent_embed_dim
        self.time_embed_dim = time_embed_dim
        self.embed_num = 8
        elevation = 30
        azimuth_interval = 360.0 / self.embed_num
        camera_distance = 1.5

        multires= 10
        embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires - 1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
            }
        self.embedder_3D = Embedder(**embed_kwargs)

        elevation_rad = np.radians(elevation)
        azimuth_angles = np.arange(0, 360, azimuth_interval)
        azimuth_rads = np.radians(azimuth_angles)

        self.R = []
        self.T = []
        self.c2w = []
        camera_posi_list = []
        for azimuth_rad in azimuth_rads:
            x = camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = camera_distance * np.sin(elevation_rad)
            eye = np.array([x, y, z])
            at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            zaxis = at - eye
            zaxis = zaxis / np.linalg.norm(zaxis)
            xaxis = np.cross(-up, zaxis)
            xaxis = xaxis / np.linalg.norm(xaxis)
            yaxis = np.cross(zaxis, xaxis)

            _viewMatrix = np.array([
                [xaxis[0], yaxis[0], zaxis[0], eye[0]],
                [xaxis[1], yaxis[1], zaxis[1], eye[1]],
                [xaxis[2], yaxis[2], zaxis[2], eye[2]],
                [0       , 0       , 0       , 1     ]
            ])

            R = np.array([
                [xaxis[0], xaxis[1], xaxis[2]],
                [yaxis[0], yaxis[1], yaxis[2]],
                [zaxis[0], zaxis[1], zaxis[2]],
            ])
            T = np.array([
                -np.dot(xaxis, eye),
                -np.dot(yaxis, eye),
                -np.dot(zaxis, eye),
            ])
            
            self.R.append(torch.Tensor(R))
            self.T.append(torch.Tensor(T))
            self.c2w.append(torch.Tensor(_viewMatrix))
            camera_posi_list.append(torch.Tensor([x, y, z]))
        
        all_cam_posi = torch.stack(camera_posi_list, dim=0)
        self.camera_posi_feat = self.embedder_3D.embed(all_cam_posi)

        self.K_256 = torch.Tensor(\
            [[280.,      0.,    128.],
            [   0.,    280.,    128.],
            [   0.,      0.,      1.]]
            )
        
        self.n_samples = 32
        self.n_importance = 0

        # self.eps_feature_ln = nn.Sequential(nn.Linear(self.latent_embed_dim, self.latent_embed_dim), nn.GELU())
        # self.xt_feature_ln = nn.Sequential(nn.Linear(self.xt_latent_embed_dim, self.xt_latent_embed_dim), nn.GELU())

        self.feature_dim = self.latent_embed_dim + 2 * self.embedder_3D.out_dim
        # self.sigma_embeddings = nn.Parameter(torch.randn(self.latent_embed_dim))
        # self.sigma_query_ln = nn.Linear(self.latent_embed_dim + time_embed_dim, self.feature_dim)

        self.transformer_layer_norm = nn.LayerNorm(self.feature_dim)
        self.transformer = nn.Sequential(*([Attention(query_dim=self.feature_dim, heads=8, dim_head=32, residual_connection=True)] * 2))
        
        # self.sigma_net = nn.Linear(self.feature_dim + 640 + 126, 4)
    
    @staticmethod
    def _get_signature_keys(obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}
        return expected_modules, optional_parameters

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            fn_recursive_set_mem_eff(module)
    
    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/). When this
        option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
        up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        >>> # Workaround for not accepting attention shape using VAE for Flash Attention
        >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def gen_rays(self, pose, W, H):
        """
        Generate rays at world space given pose.
        """
        tx = torch.linspace(0, W - 1, int(W)).to(pose.device)
        ty = torch.linspace(0, H - 1, int(H)).to(pose.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()

        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         (pixels_y - self.K[1][2]) / self.K[1][1],
                         torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = p
        rays_v = torch.sum(rays_v[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o, rays_v
    
    def get_near_far_from_cube(self, rays_o, rays_d):
        # r.dir is unit direction vector of ray
        if (rays_d == 0).any():
            pass
        dirfrac = 1 / rays_d
        # lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
        bound_min = torch.tensor([-0.5, -0.5, -0.5]).to(rays_o.device)
        bound_max = torch.tensor([0.5, 0.5, 0.5]).to(rays_o.device)
        # r.org is origin of ray
        t_1 = (bound_min - rays_o) * dirfrac
        t_2 = (bound_max - rays_o) * dirfrac
        tmin = torch.max(torch.minimum(t_1, t_2), dim=1).values
        tmax = torch.min(torch.maximum(t_1, t_2), dim=1).values

        mask = torch.ones(rays_o.shape[0]).to(rays_o.device)
        mask[tmax < 0] = 0
        mask[tmin > tmax] = 0
        tmp = tmin.clone().detach()
        tmin = torch.where(mask > 0.5, tmin, tmax)
        tmax = torch.where(mask > 0.5, tmax, tmp)
        assert (tmin <= tmax).all()
        return tmin.unsqueeze(-1), tmax.unsqueeze(-1), mask > 0.5
    
    def volume_rendering(self, sigma, dists, color=None, z_vals=None):
        raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw)*dists)
        alpha = raw2alpha(sigma, dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        outputs = {}
        if color is not None:
            outputs["rgb_map"] = torch.sum(weights[...,None] * color, -2)  # [N_rays, 3]
        if z_vals is not None:
            outputs["depth_map"] = torch.sum(weights * z_vals, -1)
        outputs["occ_map"] = torch.sum(weights, -1)

        return outputs

    def interpolate_sigma(self, sigma_voxels, pts):
        batch_size = sigma_voxels.shape[0]
        rays_num, pts_per_ray = pts.shape[:2]
        normal_pts = (pts[None] * 2).flip(-1)
        normal_pts = torch.stack([normal_pts] * batch_size, dim=0)
        sigma = F.interpolate(sigma_voxels, normal_pts, mode="trilinear", align_corners=True)
        sigma = sigma.reshape(batch_size, 1, rays_num, pts_per_ray).permute(0, 2, 3, 1)
        return sigma
    
    def random_pixel_mask(self, image_shape, num_pixels):
        # 获取图像的高度和宽度
        height, width, _ = image_shape
        
        # 创建一个全零的掩码
        mask = torch.zeros((height, width), dtype=torch.bool)
        
        # 生成随机行和列的索引
        random_rows = torch.randint(0, height, (num_pixels,))
        random_cols = torch.randint(0, width, (num_pixels,))
        
        # 将相应位置的像素设置为True
        mask[random_rows, random_cols] = True
        
        return mask
    
    def project_embeddings_to_multi_view(self, voxels, pts):
        # pts [batch_size_3D, n_rays, n_samples, 3]
        batch_size_3D, voxel_grid_size, voxel_grid_size, voxel_grid_size, embed_dim = voxels.shape
        # voxels = voxels.reshape(batch_size, self.grid_size, self.grid_size, self.grid_size, -1)

        normal_pts = (pts * 2).flip(-1)
        normal_pts = normal_pts[:, None]

        sample = F.grid_sample(voxels.permute(0, 4, 1, 2, 3), normal_pts, mode="bilinear", align_corners=True, padding_mode="border")
        sample = sample.permute(0, 2, 3, 4, 1).squeeze(1)
        return sample

    def forward(self, sample, temb, embed_mask=None, render_camera=None, render_rays_num=0, coarse_weight_map=None, coarse_weight_rays=None, coarse_pixel_mask=None, coarse_sigma=None, coarse_sigma_embeddings_rays=None):
        output = {}
        output_sample = sample.clone().detach()
        batch_size, latent_embed_dim, H, W = sample.shape
        assert latent_embed_dim == self.latent_embed_dim, "Wrong latent embedding dim."
        batch_size_3D = batch_size // self.embed_num
        sample = sample.reshape(batch_size_3D, self.embed_num, latent_embed_dim, H, W)

        batch_size, time_embed_dim = temb.shape
        assert time_embed_dim == self.time_embed_dim, "Wrong time embedding dim."
        temb = temb.reshape(batch_size_3D, self.embed_num, time_embed_dim)
        
        batch_size_2D = embed_mask.int().sum().item()
        embed_mask_indices = torch.nonzero(embed_mask).squeeze()

        if embed_mask is not None and embed_mask.float().sum() > 0:            
            reso_down = 256 / H
            self.K = self.K_256 / reso_down
            self.K[:, 2] -= 0.5
            self.K[2, 2] = 1.0
            
            pt_list = []
            dist_list = []
            z_val_list = []
            for cam_id in range(self.embed_num):
                if not embed_mask[cam_id]:
                    continue
                # generate rays
                camera = self.c2w[cam_id].to(sample.device)
                rays_o, rays_d = self.gen_rays(camera, W, H)
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                near, far, _ = self.get_near_far_from_cube(rays_o, rays_d)

                # sample points in each ray
                z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(rays_o.device)
                z_vals = near + (far - near) * z_vals[None, :]
                if self.n_importance > 0:
                    coarse_weight_map = coarse_weight_map.reshape(batch_size, H * W, self.n_samples)
                    z_vals_new = self.sample_pdf(z_vals, coarse_weight_map[cam_id].reshape(H * W, self.n_samples)[:, :-1], self.n_importance)
                    z_vals, indices = torch.sort(torch.cat([z_vals, z_vals_new], dim=-1), dim=-1)

                dists = z_vals[..., 1:] - z_vals[..., :-1]
                dists = torch.cat([dists, dists[...,-1:]], -1)
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
                pt_list.append(pts)
                dist_list.append(dists)
                z_val_list.append(z_vals)
            pts = torch.cat(pt_list, dim=0)
            pts = torch.stack([pts] * batch_size_3D, dim=0)
            dists = torch.cat(dist_list, dim=0)
            dists = torch.stack([dists] * batch_size_3D, dim=0)
            z_vals = torch.cat(z_val_list, dim=0)
            z_vals = torch.stack([z_vals] * batch_size_3D, dim=0)

            # project points to uv-coor to get features
            pts_feature_list = []
            for embed_id in range(self.embed_num):
                uv = (pts @ self.R[embed_id].T.to(sample.device) + self.T[embed_id].to(sample.device)) 
                uv = uv @ self.K.T.to(sample.device)
                uv = uv[..., :2] / uv[..., 2:]
                uv = uv / (256.0 // reso_down - 1.0) * 2 - 1
                pts_feature = F.grid_sample(sample[:, embed_id], uv.to(sample.dtype), mode="bilinear", align_corners=True, padding_mode="border")
                pts_feature = pts_feature.permute(0, 2, 3, 1)
                pts_feature_list.append(pts_feature)
            pts_features = torch.stack(pts_feature_list, dim=-2)

            # go through the network to get values
            rays_num = H * W
            n_all_sample = self.n_samples + self.n_importance
            resi_ray_features = pts_features.reshape(batch_size_3D, batch_size_2D, rays_num, n_all_sample, self.embed_num, latent_embed_dim)
            pts_features = pts_features.reshape(batch_size_3D * batch_size_2D * rays_num * n_all_sample, self.embed_num, latent_embed_dim)
            camera_embed = self.camera_posi_feat[None].repeat(batch_size_3D * batch_size_2D * rays_num * n_all_sample, 1, 1).to(pts_features.device)
            xyz_embed = self.embedder_3D.embed(pts)[:, :, :, None, :].repeat(1, 1, 1, self.embed_num, 1).reshape(batch_size_3D * batch_size_2D * rays_num * n_all_sample, self.embed_num, self.embedder_3D.out_dim)
            pts_features = torch.cat([pts_features, camera_embed, xyz_embed], dim=-1)

            # sigma_timestep_embeddings = torch.cat([self.sigma_embeddings[None].repeat(temb.shape[0], 1), temb[:, 0]], dim=-1)
            # sigma_timestep_embeddings = self.sigma_query_ln(sigma_timestep_embeddings)

            # pts_features = torch.cat([sigma_timestep_embeddings[:, None, :].repeat(1, batch_size_2D * rays_num * n_all_sample, 1).reshape(batch_size_3D * batch_size_2D * rays_num * n_all_sample, 1, self.feature_dim), pts_features], dim=1)

            pts_features = self.transformer_layer_norm(pts_features)
            # pts_features = pts_features.reshape(batch_size_3D, batch_size_2D, H * W * n_all_sample, self.embed_num + 1, self.feature_dim)
            block_features_list = []
            for block in range((pts_features.shape[0] - 1) // 32768 + 1):
                if not block == ((pts_features.shape[0] - 1) // 32768):
                    block_features = self.transformer(pts_features[block * 32768: (block + 1) * 32768])
                else:
                    block_features = self.transformer(pts_features[block * 32768:])
                block_features_list.append(block_features)
            pts_features = torch.cat(block_features_list, dim=0)[..., :latent_embed_dim]
                
            # nerf_feature = add_feature[..., 0, :]
            sigma = coarse_sigma.reshape(batch_size_3D * self.embed_num, H * W, self.n_samples, -1)[torch.cat([embed_mask] * batch_size_3D, dim=0)]
            # coarse_sigma_embeddings_map = coarse_sigma_embeddings_map.reshape(batch_size_3D * self.embed_num, H * W, self.n_samples, -1)
            # coarse_sigma_embeddings_map = coarse_sigma_embeddings_map[torch.cat([embed_mask] * batch_size_3D, dim=0)].reshape(batch_size_2D * H * W * self.n_samples, -1)

            # nerf_feature = self.sigma_net(nerf_feature)
            # sigma = nerf_feature[..., :1]
            # color = torch.sigmoid(nerf_feature[..., 1:])
            # color = color.reshape(batch_size_3D, batch_size_2D, rays_num, n_all_sample, 3)


            new_pts_features = resi_ray_features + pts_features.reshape(batch_size_3D, batch_size_2D, rays_num, n_all_sample, self.embed_num, latent_embed_dim)
            
            selected_feature_list = []

            for i in range(batch_size_2D):
                selected_feature = new_pts_features[:, i, :, :, embed_mask_indices[i], :]
                # selected_color = color[:, i]
                selected_feature_list.append(selected_feature)
            selected_features = torch.stack(selected_feature_list, dim=1)

            # volume rendering with latent values as RGB
            render_out = self.volume_rendering(sigma.squeeze(1).reshape(-1, n_all_sample), dists.reshape(-1, n_all_sample), selected_features.reshape(-1, n_all_sample, latent_embed_dim))
            feature_map, occ_map = render_out["rgb_map"], render_out["occ_map"]
            feature_map = feature_map.reshape(batch_size_3D * batch_size_2D, H, W, latent_embed_dim).permute(0, 3, 1, 2)
            occ_map = occ_map.reshape(batch_size_3D * batch_size_2D, H, W, 1).permute(0, 3, 1, 2)
            # depth_map = depth_map.reshape(batch_size_3D * batch_size_2D, H, W, 1).permute(0, 3, 1, 2)
            output_sample[torch.cat([embed_mask] * batch_size_3D, dim=0)] = (output_sample[torch.cat([embed_mask] * batch_size_3D, dim=0)] * (1 - occ_map) + feature_map).to(output_sample.dtype)

            output["sample"] = output_sample
            output["occ_map"] = occ_map
            # coarse_sigma_embeddings_map = coarse_sigma_embeddings_map.reshape(batch_size_3D, self.embed_num, H * W * self.n_samples, -1)
            # dists = dists.reshape(batch_size_3D, batch_size_2D, H * W * n_all_sample, 1)
            # feature_map_list = []
            # occ_map_list = []

            # for view_id in range(batch_size_2D):
            #     view_pts_features = pts_features[:, view_id].reshape(batch_size_3D * H * W * n_all_sample, self.embed_num + 1, self.feature_dim)
            #     view_resi_ray_features = resi_ray_features[:, view_id].reshape(batch_size_3D * H * W * n_all_sample, self.embed_num, latent_embed_dim)
            #     view_coarse_sigma_embeddings_map = coarse_sigma_embeddings_map[:, embed_mask_indices[view_id]].reshape(batch_size_3D * H * W * n_all_sample, 766)
            #     view_dists = dists[:, view_id].reshape(batch_size_3D * H * W * n_all_sample, 1)
            #     view_block_size = 16384

            #     for block in range((view_pts_features.shape[0] - 1) // view_block_size + 1):
            #         if not block == ((view_pts_features.shape[0] - 1) // view_block_size):
            #             view_block_features = self.transformer(view_pts_features[block * view_block_size: (block + 1) * view_block_size])
            #             view_block_resi_ray_features = view_resi_ray_features[block * view_block_size: (block + 1) * view_block_size]
            #             view_block_coarse_sigma_embeddings_map = view_coarse_sigma_embeddings_map[block * view_block_size: (block + 1) * view_block_size]
            #             view_block_dists = view_dists[block * view_block_size: (block + 1) * view_block_size]
            #         else:
            #             view_block_features = self.transformer(view_pts_features[block * view_block_size:])
            #             view_block_resi_ray_features = view_resi_ray_features[block * view_block_size:]
            #             view_block_coarse_sigma_embeddings_map = view_coarse_sigma_embeddings_map[block * view_block_size:]
            #             view_block_dists = view_dists[block * view_block_size:]
                        
            #         view_block_nerf_feature = view_block_features[..., 0, :]
                    
            #         view_block_coarse_sigma_embeddings_map = view_block_coarse_sigma_embeddings_map.reshape(batch_size_3D * view_block_size, 766)
            #         view_block_nerf_feature = self.sigma_net(torch.cat([view_block_nerf_feature, view_block_coarse_sigma_embeddings_map], dim=-1))
            #         view_block_sigma = view_block_nerf_feature[..., :1]
            #         view_block_pts_features = view_block_features[..., 1:, :latent_embed_dim]

            #         new_pts_features = view_block_resi_ray_features.reshape(batch_size_3D, view_block_size, self.embed_num, latent_embed_dim) + view_block_pts_features.reshape(batch_size_3D, view_block_size, self.embed_num, latent_embed_dim)

            #         view_block_selected_features = new_pts_features[:, :, embed_mask_indices[view_id], :]
                    
            #         # volume rendering with latent values as RGB
            #         view_block_render_out = self.volume_rendering(view_block_sigma.squeeze(1).reshape(-1, n_all_sample), view_block_dists.reshape(-1, n_all_sample), view_block_selected_features.reshape(-1, n_all_sample, latent_embed_dim))
            #         view_block_feature_map, view_block_occ_map = view_block_render_out["rgb_map"], view_block_render_out["occ_map"]
            #         feature_map_list.append(view_block_feature_map)
            #         occ_map_list.append(view_block_occ_map)

            # feature_map = torch.cat(feature_map_list, dim=0).reshape(batch_size_3D * batch_size_2D, H, W, latent_embed_dim).permute(0, 3, 1, 2)
            # occ_map = torch.cat(occ_map_list, dim=0).reshape(batch_size_3D * batch_size_2D, H, W, 1).permute(0, 3, 1, 2)
            # output_sample[torch.cat([embed_mask] * batch_size_3D, dim=0)] = (output_sample[torch.cat([embed_mask] * batch_size_3D, dim=0)] * (1 - occ_map) + feature_map).to(output_sample.dtype)

            output["sample"] = output_sample
            output["occ_map"] = occ_map
        else:
            output["sample"] = None
            output["occ_map"] = None
        
        # if render_camera is not None and (render_rays_num > 0 or coarse_pixel_mask is not None):
        #     assert batch_size_3D == 1
        #     reso_down = 256 / H
        #     self.K = self.K_256 / 1.0
        #     self.K[:, 2] -= 0.5
        #     self.K[2, 2] = 1.0
        #     rays_o, rays_d = self.gen_rays(render_camera[0], 256, 256)
        #     if coarse_pixel_mask is None:
        #         pixel_mask = self.random_pixel_mask(rays_o.shape, render_rays_num)
        #     else:
        #         pixel_mask = coarse_pixel_mask

        #     rays_o = rays_o[pixel_mask]
        #     rays_d = rays_d[pixel_mask]
        #     near, far, _ = self.get_near_far_from_cube(rays_o, rays_d)

        #     # sample points in each ray
        #     all_z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(rays_o.device)
        #     all_z_vals = near + (far - near) * all_z_vals[None, :]

        #     if self.n_importance > 0 and coarse_pixel_mask is not None:
        #         coarse_weight_rays = coarse_weight_rays.reshape(pixel_mask.sum().int(), self.n_samples)
        #         all_z_vals_new = self.sample_pdf(all_z_vals, coarse_weight_rays[:, :-1], self.n_importance)
        #         all_z_vals, indices = torch.sort(torch.cat([all_z_vals, all_z_vals_new], dim=-1), dim=-1)

        #     all_dists = all_z_vals[..., 1:] - all_z_vals[..., :-1]
        #     all_dists = torch.cat([all_dists, all_dists[...,-1:]], -1)
        #     all_pts = rays_o[:, None, :] + rays_d[:, None, :] * all_z_vals[..., :, None]
            
        #     all_ray_num = all_pts.shape[0]
        #     n_all_sample = self.n_samples + self.n_importance
        #     chunk_size = 2048
        #     chunk_color_rays_list = []
        #     chunk_occ_rays_list = []
        #     # project points to uv-coor to get features
        #     for chunk_id in range((all_ray_num - 1) // chunk_size + 1):
        #         if chunk_id < (all_ray_num - 1) // chunk_size:
        #             pts = all_pts[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
        #             dists = all_dists[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
        #         else:
        #             pts = all_pts[chunk_id*chunk_size:]
        #             dists = all_dists[chunk_id*chunk_size:]
        #         ray_num = pts.shape[0]
        #         pts_feature_list = []
        #         self.K = self.K_256 / reso_down
        #         self.K[:, 2] -= 0.5
        #         self.K[2, 2] = 1.0
        #         for embed_id in range(self.embed_num):
        #             uv = (pts @ self.R[embed_id].T.to(sample.device) + self.T[embed_id].to(sample.device)) 
        #             uv = uv @ self.K.T.to(sample.device)
        #             uv = uv[..., :2] / uv[..., 2:]
        #             uv = uv / (256.0 // reso_down - 1.0) * 2 - 1
        #             uv = torch.stack([uv] * batch_size_3D, dim=0)
        #             pts_feature = F.grid_sample(sample[:, embed_id], uv.to(sample.dtype), mode="bilinear", align_corners=True, padding_mode="border")
        #             pts_feature = pts_feature.permute(0, 2, 3, 1)
        #             pts_feature_list.append(pts_feature)

        #         pts_features = torch.stack(pts_feature_list, dim=-2)

        #         pts_features = pts_features.reshape(ray_num * n_all_sample, self.embed_num, latent_embed_dim)
        #         camera_embed = self.camera_posi_feat[None].repeat(ray_num * n_all_sample, 1, 1).to(sample.device)

        #         xyz_embed = self.embedder_3D.embed(pts)[:, :, None, :].repeat(1, 1, self.embed_num, 1).reshape(ray_num * n_all_sample, self.embed_num, self.embedder_3D.out_dim)
        #         pts_features = torch.cat([pts_features, camera_embed, xyz_embed], dim=-1)

        #         sigma_timestep_embeddings = torch.cat([self.sigma_embeddings, temb[0, 0]], dim=-1)
        #         sigma_timestep_embeddings = self.sigma_query_ln(sigma_timestep_embeddings)
        #         pts_features = torch.cat([sigma_timestep_embeddings[None, None, :].repeat(ray_num * n_all_sample, 1, 1), pts_features], dim=1)

        #         pts_features = self.transformer_layer_norm(pts_features)
        #         block_features_list = []
        #         for block in range((pts_features.shape[0] - 1) // 32768 + 1):
        #             if not block == ((pts_features.shape[0] - 1) // 32768):
        #                 block_features = self.transformer(pts_features[block * 32768: (block + 1) * 32768])
        #             else:
        #                 block_features = self.transformer(pts_features[block * 32768:])
        #             block_features_list.append(block_features)
        #         pts_features = torch.cat(block_features_list, dim=0)
                    
        #         nerf_feature = pts_features[..., 0, :]
        #         nerf_feature = self.sigma_net(torch.cat([nerf_feature, coarse_sigma_embeddings_rays], dim=-1))
        #         sigma = nerf_feature[..., :1]
        #         color = torch.sigmoid(nerf_feature[..., 1:])

        #         # volume rendering with latent values as RGB
        #         render_out = self.volume_rendering(sigma.squeeze(1).reshape(-1, n_all_sample), dists.reshape(-1, n_all_sample), color.reshape(-1, n_all_sample, 3))
        #         chunk_occ_rays_list.append(render_out["occ_map"])
        #         chunk_color_rays_list.append(render_out["rgb_map"])
            
        #     occ_rays = torch.cat(chunk_occ_rays_list, dim=0)
        #     output["occ_rays"] = occ_rays
        #     output["pixel_mask"] = pixel_mask
        #     color_rays = torch.cat(chunk_color_rays_list, dim=0)

        #     color_rays_with_bg = torch.Tensor([1., 1., 1.]).to(color_rays.device)[None].repeat(all_ray_num, 1)
        #     color_rays_with_bg = color_rays_with_bg * (1 - occ_rays[:, None]) + color_rays
        #     output["color_rays"] = color_rays_with_bg
        # else:
        #     output["occ_rays"] = None
        #     output["pixel_mask"] = None
        #     output["color_rays"] = None

        return output

    def unproject_embeddings_from_multi_view(self, sample, reso_down):
        batch_size, embed_dim, H, W = sample.shape
        batch_size_3D = batch_size // self.image_num
        sample = sample.reshape(batch_size_3D, self.image_num, embed_dim, H, W)
        x = torch.linspace(-0.5, 0.5, self.grid_size).to(sample.device)
        y = torch.linspace(-0.5, 0.5, self.grid_size).to(sample.device)
        z = torch.linspace(-0.5, 0.5, self.grid_size).to(sample.device)
        xv, yv, zv = torch.meshgrid(x, y, z)
        voxels = torch.stack([xv, yv, zv], dim=-1)
        voxels[0][0][0] = torch.Tensor([0, 0, 0])
        voxels_features = []
        for i in range(self.image_num):
            uv = (voxels @ self.R[i].T.to(sample.device) + self.T[i].to(sample.device)) 
            uv = uv @ self.K.T.to(sample.device)
            uv = uv[..., :2] / uv[..., 2:]
            uv = uv.reshape(self.grid_size, self.grid_size*self.grid_size, 2) / (256.0 // reso_down - 1) * 2 - 1
            uv = torch.stack([uv] * batch_size_3D, dim=0)
            voxels_feature = F.grid_sample(sample[:, i], uv, padding_mode= "border", mode="bilinear", align_corners=True)
            voxels_feature = voxels_feature.permute(0, 2, 3, 1).reshape(batch_size_3D, self.grid_size, self.grid_size, self.grid_size, -1)
            voxels_features.append(voxels_feature)
        
        voxels_features = torch.stack(voxels_features, dim=-2)
        return voxels_features, voxels
    
    def sample_pdf(self, bins, weights, N_samples, det=False, pytest=False):
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

        # Invert CDF
        u = u.to(cdf.device).contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

class Unet2DConditionModelFor3D(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        latent_plane_coarse = None,
        latent_plane_fine = None,
        latent_plane_coarse_kwargs = None,
        latent_plane_fine_kwargs = None,
        use_latent_plane = False
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        res_sample_clone = sample.clone().detach()

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_block_additional_residuals) > 0
                and sample.shape == down_block_additional_residuals[0].shape
            ):
                sample += down_block_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        latent_plane_coarse_output = {
            "voxel_sigma": None
            }
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    scale=lora_scale,
                )
            if use_latent_plane and i == 2 and latent_plane_coarse is not None:
                latent_plane_coarse_output = latent_plane_coarse(sample, emb, **latent_plane_coarse_kwargs)

                
            if use_latent_plane and i == 2 and latent_plane_fine is not None:
                latent_plane_fine_kwargs["coarse_sigma"] = latent_plane_coarse_output["sigma"]
                latent_plane_fine_output = latent_plane_fine(sample, emb, **latent_plane_fine_kwargs)
                sample = latent_plane_fine_output["sample"]
                # sample = latent_plane_coarse_output["sample"]
                # latent_plane_fine_kwargs["coarse_sigma_embeddings_map"] = latent_plane_coarse_output["sigma_embeddings_map"]
                # latent_plane_fine_kwargs["coarse_sigma_embeddings_rays"] = latent_plane_coarse_output["sigma_embeddings_rays"]
            #     assert "alpha_cumprod_t" in latent_plane_fine_kwargs.keys() and  "beta_cumprod_t" in latent_plane_fine_kwargs.keys()
            #     sample = latent_plane_fine_output["sample"]

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return Unet2DConditionOutputFor3D(
            sample=sample,
            # ray_occ=latent_plane_coarse_output["occ_rays"], 
            # image_occ=latent_plane_coarse_output["occ_map"],
            voxel_sigma=latent_plane_coarse_output["voxel_sigma"]
            )
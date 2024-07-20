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
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
import torch
from diffusers.utils import logging
import sys
sys.path.append("./Latent_Plane")
from Latent_Plane.zero123 import Zero123PipelineFor3D
from torchvision import transforms
from Latent_Plane.latent_plane_inference_new import LatentPlane_Coarse, LatentPlane_Fine, Unet2DConditionModelFor3D
import random
import argparse
import os

class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        # image = Image.fromarray(image)
        image = self.interface([image])[0]
        # image = np.array(image)
        return image

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--elev', type=float, default=0.0, help="estimated elevation angle of reference image.")
    parser.add_argument('--im_path', type=str, required=True, help="path to the reference image.")
    parser.add_argument('--out_image_path', type=str, default='./multiview_images', help="path to save the output multi-view images.")
    parser.add_argument('--out_mesh_path', type=str, default='./mesh', help="path to save the output reconstruction mesh.")
    args = parser.parse_args()
    unet = Unet2DConditionModelFor3D.from_pretrained("../../zero123-xl-diffusers", subfolder="unet", torch_dtype=torch.float16)
    pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
        }

    pipeline = Zero123PipelineFor3D.from_pretrained("../../zero123-xl-diffusers", unet=unet, **pipe_kwargs, torch_dtype=torch.float16)
    latent_plane_coarse = LatentPlane_Coarse(640, 1280)
    latent_plane_coarse.n_samples = 32
    latent_plane_coarse.load_state_dict(torch.load("./Latent_Plane/checkpoints/coarse_21500.pt", map_location="cpu"))
    latent_plane_coarse.enable_xformers_memory_efficient_attention()
    # latent_plane_coarse.half()
    latent_plane_fine = LatentPlane_Fine(640, 1280)
    latent_plane_fine.n_importance = 0
    latent_plane_fine.n_samples = 32
    latent_plane_fine.load_state_dict(torch.load("./Latent_Plane/checkpoints/fine_8100.pt", map_location="cpu"))
    latent_plane_fine.enable_xformers_memory_efficient_attention()
    # latent_plane_fine.half()
    pipeline.to("cuda")
    latent_plane_coarse.cuda()
    latent_plane_fine.cuda()
    image_transforms = transforms.Compose(
            [
                transforms.Pad(padding=32, fill=(255, 255, 255)),
                transforms.Resize((256, 256)),  # 256, 256
                transforms.ToTensor()
            ]
        )
    img_path = args.im_path
    img = plt.imread(img_path)
    color = [1., 1., 1., 1.]
    mask = img[:, :, -1] > 0.
    if img.shape[-1] == 4:
        img[img[:, :, -1] == 0.] = color

    img = Image.fromarray(np.uint8(img[:, :, :3] * 255.0))
    img = image_transforms(img)

    cond_images = torch.stack([img] * 8, dim=0).cuda()

    d_T_list = []
    d_theta = args.elev
    d_z = 0
    d_azimuth_list = []
    d_theta_list = []
    d_z_list = []
    for view in range(8):
        d_azimuth = 360.0 / 8 * view
        d_T = torch.tensor([np.deg2rad(d_theta), math.sin(np.deg2rad(d_azimuth)), math.cos(np.deg2rad(d_azimuth)), d_z])
        d_T_list.append(d_T)
        d_azimuth_list.append(d_azimuth)
        d_theta_list.append(d_theta)
        d_z_list.append(d_z)
    d_theta = torch.Tensor(d_theta_list).cuda()
    d_azimuth = torch.Tensor(d_azimuth_list).cuda()
    d_z = torch.Tensor(d_z_list).cuda()
    with torch.autocast("cuda"):
        with torch.no_grad():
            model_output, mesh = pipeline(image=cond_images.half(), elevation=d_theta.squeeze(-1).half(), azimuth=d_azimuth.squeeze(-1).half(), distance=d_z.squeeze(-1).half(), height=256, width=256, guidance_scale=2.0, num_inference_steps=50, latent_plane_coarse=latent_plane_coarse, latent_plane_fine=latent_plane_fine)
    images = model_output.images
    os.makedirs(args.out_image_path, exist_ok=True)
    filename = os.path.basename(img_path)[:-4]
    for j in range(len(images)):
        os.makedirs(os.path.join(args.out_image_path, filename), exist_ok=True)
        images[j].save(os.path.join(args.out_image_path, filename, "{}.png".format(j)))
        # mask_predictor = BackgroundRemoval()
        # image_rgba = mask_predictor(images[j])
        # image_rgba.save(os.path.join(args.out_image_path, filename, "{}_rgba.png".format(j)))
    
    os.makedirs(args.out_mesh_path, exist_ok=True)
    mesh.export("{}/{}.obj".format(args.out_mesh_path, filename), "obj")

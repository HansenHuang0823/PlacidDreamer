from scene.gaussian_model import GaussianModel
import torch.nn as nn
import os
import math
import numpy as np
import torch
from gaussian_renderer import render
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from argparse import ArgumentParser
import trimesh
from lavis.models import load_model_and_preprocess
from argparse import ArgumentParser
import sys

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def get_rays_torch(focal, c2w, H=64,W=64):
    """Computes rays using a General Pinhole Camera Model
    Assumes self.h, self.w, self.focal, and self.cam_to_world exist
    """
    x, y = torch.meshgrid(
        torch.arange(W),  # X-Axis (columns)
        torch.arange(H),  # Y-Axis (rows)
        indexing='xy')
    camera_directions = torch.stack(
        [(x - W * 0.5 + 0.5) / focal,
            -(y - H * 0.5 + 0.5) / focal,
            -torch.ones_like(x)],
        dim=-1).to(c2w)

    # Rotate ray directions from camera frame to the world frame
    directions = ((camera_directions[ None,..., None, :] * c2w[None,None, None, :3, :3]).sum(axis=-1))  # Translate camera frame's origin to the world frame
    origins = torch.broadcast_to(c2w[ None,None, None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)

    return torch.cat((origins,viewdirs),dim=-1)

class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

class RCamera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, uid, delta_polar, delta_azimuth, delta_radius, opt,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", SSAA=False
                 ):
        super(RCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.delta_polar = delta_polar
        self.delta_azimuth = delta_azimuth
        self.delta_radius = delta_radius
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01
        
        if SSAA:
            ssaa = opt["SSAA"]
        else:
            ssaa = 1

        self.image_width = opt["image_w"] * ssaa
        self.image_height = opt["image_h"] * ssaa

        self.trans = trans
        self.scale = scale

        RT = torch.tensor(getWorld2View2(R, T, trans, scale))
        self.world_view_transform = RT.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # self.rays = get_rays_torch(fov2focal(FoVx, 64), RT).cuda()
        self.rays = get_rays_torch(fov2focal(FoVx, self.image_width//8), RT, H=self.image_height//8, W=self.image_width//8).cuda()

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--tmp_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args(sys.argv[1:])

model = GaussianModel(0)
with open("configs/prompt_single.txt", "r") as f:
    prompt_list = f.readlines()

eval_model, vis_processors, _ = load_model_and_preprocess(name='blip2_t5', model_type='pretrain_flant5xxl', is_eval=True, device="cpu")

for prompt_ind in range(len(prompt_list)):
    model_path = os.path.join(args.input_dir, str(prompt_ind), "point_cloud/iteration_5000/point_cloud.ply")

    if not os.path.exists(model_path):
        prompt = prompt_list[prompt_ind].strip()
        texts = ["empty"] * 12
        with open(os.path.join(args.output_dir, f'{idx}.txt'), 'a+') as f:
            f.writelines(texts)
        continue

    model.load_ply(model_path)
    prompt = prompt_list[prompt_ind].strip()
    xyz = model._xyz
    scale = (xyz.max(dim=0).values - xyz.min(dim=0).values).max()
    targets = xyz.mean(dim=0).detach().cpu()
    targets = torch.zeros_like(targets)

    radius = 2.2 * scale.item() / 2.0
    icosphere = trimesh.creation.icosphere(subdivisions=0)
    icosphere.vertices *= radius

    for idx, v in enumerate(icosphere.vertices):
        fov_list = [np.pi/3.0]
        size = len(fov_list)
        centers = torch.FloatTensor([v] * size) + targets

        # lookat
        forward_vector = safe_normalize(centers - targets)
        up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
        #up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
        right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
        if right_vector.norm() < 1e-3:
            right_vector = torch.FloatTensor([1, 0, 0]).unsqueeze(0).repeat(size, 1)


        up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1)) #forward_vector

        poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
        poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1) #up_vector
        poses[:, :3, 3] = centers
        

        opt = {
            "SSAA":True,
            "image_w": 512,
            "image_h": 512
        }
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipe = PipelineParams()
        for cam_id in range(size):
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            fov = fov_list[cam_id]
            fovy = focal2fov(fov2focal(fov, 512), 512)
            FovY = fovy
            FovX = fov
            cam = RCamera(cam_id, R, T, FovX, FovY, 0, 0, 0, 0, opt, SSAA=True)

            render_pkg = render(cam, model, pipe, background, bg_aug_ratio=0, test=True)

            image = render_pkg["render"]
            image = image.permute(1, 2, 0).clamp(0.0, 1.0)
            image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image)
            os.makedirs(args.tmp_dir, exist_ok=True)
            image.save(os.path.join(args.tmp_dir, f'{idx:03d}.png'))

    # scores = { i: -114514 for i in range(len(icosphere.vertices)) }
    texts = []
    
    for idx in range(len(icosphere.vertices)):
        img_path = f'{idx:03d}.png'
        # convert color to PIL image
        color = Image.open(os.path.join(args.tmp_dir, img_path)).convert("RGB")
        image = vis_processors["eval"](color).unsqueeze(0)
        x = eval_model.generate({"image": image}, use_nucleus_sampling=True, num_captions=1)
        print(x[0])
        texts.append(x[0] + '\n')
    with open(os.path.join(args.output_dir, f'{prompt_ind}.txt'), 'a+') as f:
        f.writelines(texts)

    # # convolute scores on the icosphere for 3 times
    # for _ in range(3):
    #     new_scores = {}
    #     for idx, v in enumerate(icosphere.vertices):
    #         new_scores[idx] = scores[idx]
    #         for n in icosphere.vertex_neighbors[idx]:
    #             new_scores[idx] += scores[n]
    #         new_scores[idx] /= (len(icosphere.vertex_neighbors[idx]) + 1)
    #     scores = new_scores

    # for idx in sorted(scores, key=lambda x: scores[x], reverse=True)[:1]:
    #     now_score = scores[idx] * 20 + 50
    #     print(now_score)

    #     with open(f'output_T3Bench_3.txt', 'a+') as f:
    #         f.write(f'{now_score:.1f}\t\t{prompt}\n')

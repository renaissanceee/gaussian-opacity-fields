# copy from 2DGS
import math
import torch
import numpy as np

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()#[4, 4]
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')# ---> img_space
    #[800, 800], [800, 800]
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)# [HW, 3]
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T# ray-direction[HW, 3] # ---> points_space
    rays_o = c2w[:3,3]#center: [3] e.g.[0.9056, 3.4974, 1.7884]
    # t: [HW, 1], r: [HW, 3], o: [3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o # x=o+tr
    return points # [HW, 3]


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap ->torch.Size([1, 800, 800])
    """
    # print("view: ", view)
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)#[800, 800, 3]
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)#[798, 798, 3]
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points#[800, 800, 3]

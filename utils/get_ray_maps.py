import torch
import numpy as np

def get_ray_maps(intrinsic, c2w, H, W):
    """
    根据相机内参 intrinsic 与相机位姿 c2w，为每个像素生成射线（ray）：
    - rays_o：射线原点（相机光心位置），每个像素相同
    - rays_d：射线方向（单位向量），随像素变化

    这是 NeRF/3D 视觉里非常常见的几何构造。在本仓库的 GE‑Sim(Cosmos2) 中，
    这些 ray map 会与轨迹图(traj map)一起作为 cond_to_concat 的几何条件输入，
    帮助 world model 在生成视频时“理解”相机视角、成像几何与空间一致性。

    参数：
        intrinsic: torch.Tensor, 形状 (vt, 3, 3)
            - vt = v * t 或者任意 batch 维（例如把多个视角×多帧展平到一起）
            - 内参矩阵通常为：
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        c2w: torch.Tensor, 形状 (vt, 4, 4)
            - 相机坐标系到世界坐标系的齐次变换矩阵
            - c2w[:3,:3] 是旋转 R，c2w[:3,3] 是平移 t（相机光心在世界坐标的位置）
        H, W: int
            - 图像高度/宽度（像素）

    返回：
        rays_o: torch.Tensor, 形状 (vt, H, W, 3)
            - 每个像素的射线原点（相机光心），在世界坐标系
        viewdir: torch.Tensor, 形状 (vt, H, W, 3)
            - 每个像素的单位射线方向（世界坐标系）
    """
    # vt：被展平的 batch 维（例如 v*t）
    vt = intrinsic.shape[0]

    # 从内参中取出 fx/fy/cx/cy，并 reshape 到 (vt,1,1) 方便与像素网格广播
    fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2)
    fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2)
    cx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2)
    cy = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2)

    # 构造像素中心坐标网格：
    # - i 对应 x 方向（列），范围 [0.5, W-0.5]
    # - j 对应 y 方向（行），范围 [0.5, H-0.5]
    # 这里默认 torch.meshgrid 的 indexing='ij'（注释中也指出了），所以会做一次转置来对齐到 (H,W)
    i, j = torch.meshgrid(
        torch.linspace(0.5, W - 0.5, W, device=c2w.device),
        torch.linspace(0.5, H - 0.5, H, device=c2w.device),
    )
    i = i.t()  # (H,W)
    j = j.t()  # (H,W)

    # 扩展到 batch 维 (vt,H,W)
    i = i.unsqueeze(0).repeat(vt, 1, 1)
    j = j.unsqueeze(0).repeat(vt, 1, 1)

    # 将像素坐标反投影到相机坐标系下的方向向量（未归一化）：
    # dir_cam = [(x-cx)/fx, (y-cy)/fy, 1]
    # 这里假设相机坐标系 z 轴指向前方（针孔模型常用约定）
    dirs_cam = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1)  # (vt,H,W,3)

    # 将相机坐标系方向旋转到世界坐标系：
    # dir_world = R * dir_cam，其中 R = c2w[:3,:3]
    # 实现用广播的“逐元素乘+求和”来做批量矩阵乘（等价于 dirs_cam @ R^T）
    rays_d = torch.sum(dirs_cam[..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3, :3], dim=-1)  # (vt,H,W,3)

    # 射线原点：相机光心在世界坐标的位置（对每个像素重复）
    rays_o = c2w[:, :3, -1].unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1)  # (vt,H,W,3)

    # 方向归一化，得到单位向量 view direction
    viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    return rays_o, viewdir
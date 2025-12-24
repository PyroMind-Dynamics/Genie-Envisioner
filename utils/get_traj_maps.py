import numpy as np
import cv2
import matplotlib.cm as cm
import torch
from einops import rearrange

# 说明：
# 本文件提供将 3D 位姿/动作轨迹投影到多视角图像上的可视化工具。
# 输入：双臂末端位姿（含平移+四元数）、相机内外参、图像尺寸。
# 过程：将末端坐标转换到相机坐标系，投影到像素平面，绘制圆点和连线得到轨迹热力图。
# 输出：形状 (c, v, t, h, w) 的轨迹图，c=3 通道，v 视角数，t 帧数。


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵（实部在前），来自 pytorch3d 的实现。
    输入形状 (..., 4)，输出同批次的 (..., 3, 3)。
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_transformation_matrix_from_quat(quat):
    # 将位置 (x,y,z) + 四元数组合的形状 (b,7) 转成 4x4 齐次变换矩阵
    rot_quat = quat[:, 3:]
    rot_quat = rot_quat[:, [3,0,1,2]]
    rot = quaternion_to_matrix(rot_quat)
    trans = quat[:, :3]
    output = torch.eye(4).unsqueeze(0).repeat(quat.shape[0], 1, 1)
    output[:,:3,:3] = rot
    output[:,:3, 3] = trans
    return output


def simple_radius_gen_func(xyzs, c_xyzs):
    # 根据末端与相机距离生成圆点半径的经验函数，距离近 -> 半径大
    radius = torch.clamp(1.0 - torch.sqrt(((xyzs-c_xyzs)**2).sum(-1))-0.07/(0.8-0.07), min=0, max=1) * 100
    return radius


def get_traj_maps(pose, w2c, c2w, intrinsic, sample_size, radius_gen_func=None):
    """
    将双臂末端位姿序列投影到多视角图像，生成轨迹可视化图。
    参数：
        pose: 形状 (t, 16) 或 (t, >=16)，左/右臂各 7+1（平移+四元数+夹爪状态）。
        w2c: 形状 (v, t, 4, 4)，世界到相机的变换矩阵。
        c2w: 形状 (v, t, 4, 4)，相机到世界的变换矩阵。
        intrinsic: 形状 (v, 3, 3)，相机内参。
        sample_size: (h, w)，输出分辨率。
        radius_gen_func: 可选函数，根据末端与相机距离生成绘制半径。
    返回：
        torch.Tensor，形状 (c, v, t, h, w)，值域约 [0,1]，可直接作为图像/视频通道。
    """
    h, w = sample_size
    colormap_l = cm.Greens
    colormap_r = cm.Reds
    color_list_l = [ (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    color_list_r = [ (255, 0, 255), (255, 0, 0), (0, 255, 0)]

    if isinstance(pose, np.ndarray):
        pose = torch.tensor(pose, dtype=torch.float32)
    
    ee_key_pts = torch.tensor([
        [0, 0, 0, 1],
        [0.1, 0, 0, 1],
        [0, 0.1, 0, 1],
        [0, 0, 0.1, 1]
    ], dtype=torch.float32, device=pose.device).view(1,1,4,4).permute(0,1,3,2)


    # 左右臂末端从 (t,7) 转 4x4 位姿矩阵，增加 batch 维 -> (1,t,4,4)
    pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7]).unsqueeze(dim=0)
    pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15]).unsqueeze(dim=0)

    # 末端位姿从世界坐标系变换到相机坐标系
    ee2cam_l = torch.matmul(w2c, pose_l_mat)
    ee2cam_r = torch.matmul(w2c, pose_r_mat)

    correct_matrix = torch.tensor([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.23],
                        [0, 0, 0, 1]
                    ], dtype=torch.float32, device=pose.device).view(1,1,4,4)
    ee2cam_l = torch.matmul(ee2cam_l, correct_matrix)
    ee2cam_r = torch.matmul(ee2cam_r, correct_matrix)

    # 末端关键点（原点 + 三轴单位向量）变换到相机坐标系
    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)
    
    # 内参扩一维以便按时间广播
    intrinsic = intrinsic.unsqueeze(1)

    # 投影到像素平面，得到像素坐标 (u,v)
    uvs_l0 = torch.matmul(intrinsic, pts_l[:,:,:3,:])
    uvs_l = (uvs_l0 / pts_l[:,:,2:3,:])[:,:,:2,:].permute(0,1,3,2).to(dtype=torch.int64)

    uvs_r0 = torch.matmul(intrinsic, pts_r[:,:,:3,:])
    uvs_r = (uvs_r0 / pts_r[:,:,2:3,:])[:,:,:2,:].permute(0,1,3,2).to(dtype=torch.int64)

    all_img_list = []

    for icam in range(w2c.shape[0]):
        
        l_xyz = pose[:, 0:3].clone()
        r_xyz = pose[:, 8:11].clone()
        c_xyz = c2w[icam,:,:3,3].clone()

        if radius_gen_func is None:
            l_dist = 50
            r_dist = 50
        else:
            l_dist = radius_gen_func(l_xyz, c_xyz)
            r_dist = radius_gen_func(r_xyz, c_xyz)

        img_list = []
        for i in range(pose.shape[0]):
            
            # 灰底图，逐帧绘制左右末端的圆点与连线
            img = np.zeros((h, w, 3), dtype=np.uint8) + 50

            normalized_value_l = pose[i, 7].item() / 120
            normalized_value_r = pose[i, 15].item() / 120
            color_l = colormap_l(normalized_value_l)[:3]  # Get RGB values
            color_r = colormap_r(normalized_value_r)[:3]  # Get RGB values
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            i_coord_list = []
            for points, color, colors, radius, lr_tag, eef in zip([uvs_l[icam, i], uvs_r[icam, i]], [color_l, color_r], [color_list_l, color_list_r], [l_dist[i], r_dist[i]], ["left", "right"], [normalized_value_l, normalized_value_r]):
                base = np.array(points[0]) # points:[4,3]
                if base[0]<0 or base[0]>=w or base[1]<0 or base[1]>=h:
                    continue
                point = np.array(points[0][:2])
                radius = int(radius)
                cv2.circle(img, tuple(point), radius, color, -1)
                # color_circle = int(128*eef)+128
                # cv2.circle(img, tuple(point), radius, (color_circle, color_circle, color_circle), 10)

            for points, color, colors, lr_tag in zip([uvs_l[icam, i], uvs_r[icam, i]], [color_l, color_r], [color_list_l, color_list_r], ["left", "right"]):
                base = np.array(points[0]) # points:[4,3]
                if base[0]<0 or base[0]>=w or base[1]<0 or base[1]>=h:
                    continue
                for i, point in enumerate(points):
                    point = np.array(point[:2])
                    if i == 0:
                        continue
                    else:
                        cv2.line(img, tuple(base), tuple(point), colors[i-1], 8)

            img_list.append(img/255.)


        img_list = np.stack(img_list, axis=0) ### t,h,w,c
        all_img_list.append(img_list)

    # 转成 (c, v, t, h, w) 方便后续拼接/存储
    all_img_list = np.stack(all_img_list, axis=0) ### ncam, t, h, w, c
    all_img_list = rearrange(torch.tensor(all_img_list), "v t h w c -> c v t h w").float()

    return all_img_list
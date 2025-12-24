import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation


def normalize_angles(radius):
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)
    return radius_normed


def get_actions_eef(gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None):

    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        ### the first frame is repeated to fill memory
        n = all_ends_p.shape[0]-1+delta_act_sidx
        slices = [0,]*(delta_act_sidx-1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_left_rpy = []
    all_right_rpy = []

    for i in slices:
        rot_l = Rotation.from_quat(all_ends_o[i, 0])
        left_rpy = np.concatenate((all_ends_p[i,0], rot_l.as_euler("xyz", degrees=False)), axis=0)
        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        right_rpy = np.concatenate((all_ends_p[i,1], rot_r.as_euler("xyz", degrees=False)), axis=0)
        all_left_rpy.append(left_rpy)
        all_right_rpy.append(right_rpy)

    ### xyz, rpy
    all_left_rpy = np.stack(all_left_rpy)
    all_right_rpy = np.stack(all_right_rpy)

    ### xyz, xyzw, gripper
    all_abs_actions = np.zeros([n, 14])
    ### xyz, rpy, gripper
    all_delta_actions = np.zeros([n-delta_act_sidx, 14])
    for i in range(0, n):
        all_abs_actions[i, 0:6] = all_left_rpy[i, :6]
        all_abs_actions[i, 6] = gripper[slices[i], 0]
        all_abs_actions[i, 7:13] = all_right_rpy[i, :6]
        all_abs_actions[i, 13] = gripper[slices[i], 1]
        if i >= delta_act_sidx:
            all_delta_actions[i-delta_act_sidx, 0:6] = all_left_rpy[i, :6] - all_left_rpy[i-1, :6]
            all_delta_actions[i-delta_act_sidx, 3:6] = normalize_angles(all_delta_actions[i-delta_act_sidx, 3:6])
            all_delta_actions[i-delta_act_sidx, 6] = gripper[slices[i], 0]
            all_delta_actions[i-delta_act_sidx, 7:13] = all_right_rpy[i, :6] - all_right_rpy[i-1, :6]
            all_delta_actions[i-delta_act_sidx, 10:13] = normalize_angles(all_delta_actions[i-delta_act_sidx, 10:13])
            all_delta_actions[i-delta_act_sidx, 13] = gripper[slices[i], 1]

    return all_abs_actions, all_delta_actions


def get_actions_eef_quat(gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None):
    """
    为 GE-Sim 轨迹条件生成的动作格式（不做欧拉角转换）：
    - 每帧输出 16 维：
        左臂: xyz(3) + quat_xyzw(4) + gripper(1)  -> 8
        右臂: xyz(3) + quat_xyzw(4) + gripper(1)  -> 8
      合计 16 维，满足 utils/get_traj_maps.py 对 pose 的输入假设。
    注意：
    - all_ends_o 直接使用 h5 里读取的四元数（scipy Rotation.from_quat 也假设 xyzw）。
    - delta 这里不做严格定义（四元数差分不稳定），返回零数组仅用于接口兼容。
    """
    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        n = all_ends_p.shape[0] - 1 + delta_act_sidx
        slices = [0] * (delta_act_sidx - 1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_abs_actions = np.zeros([n, 16], dtype=np.float32)
    # 仅为接口兼容：不建议对四元数做简单差分训练
    all_delta_actions = np.zeros([max(n - delta_act_sidx, 0), 16], dtype=np.float32)

    for i in range(n):
        s = slices[i]
        # left: xyz + quat(xyzw) + gripper
        all_abs_actions[i, 0:3] = all_ends_p[s, 0]
        all_abs_actions[i, 3:7] = all_ends_o[s, 0]
        all_abs_actions[i, 7] = gripper[s, 0]
        # right
        all_abs_actions[i, 8:11] = all_ends_p[s, 1]
        all_abs_actions[i, 11:15] = all_ends_o[s, 1]
        all_abs_actions[i, 15] = gripper[s, 1]

    return all_abs_actions, all_delta_actions


def get_actions_joint(gripper, all_joints=None, slices=None, delta_act_sidx=None, n_arm_joints=7):

    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        ### the first frame is repeated to fill memory
        n = all_ends_p.shape[0]-1+delta_act_sidx
        slices = [0,]*(delta_act_sidx-1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_abs_actions = np.zeros([n, n_arm_joints*2+2])
    all_delta_actions = np.zeros([n-delta_act_sidx, n_arm_joints*2+2])
    for i in range(0, n):
        i_joint_l = all_joints[slices[i]][:n_arm_joints]
        i_joint_r = all_joints[slices[i]][n_arm_joints:]
        all_abs_actions[i, :n_arm_joints] = i_joint_l
        all_abs_actions[i, n_arm_joints] = gripper[slices[i], 0]
        all_abs_actions[i, n_arm_joints+1:2*n_arm_joints+1] = i_joint_r
        all_abs_actions[i, 2*n_arm_joints+1] = gripper[slices[i], 1]   
        if i >= delta_act_sidx:
            all_delta_actions[i-delta_act_sidx, :n_arm_joints] = i_joint_l - all_joints[slices[i]-1][:n_arm_joints]
            all_delta_actions[i-delta_act_sidx, n_arm_joints] = gripper[slices[i], 0]
            all_delta_actions[i-delta_act_sidx, n_arm_joints+1:2*n_arm_joints+1] = i_joint_r - all_joints[slices[i]-1][n_arm_joints:]
            all_delta_actions[i-delta_act_sidx, 2*n_arm_joints+1] = gripper[slices[i], 1]

    return all_abs_actions, all_delta_actions


def parse_h5(h5_file, slices=None, delta_act_sidx=1, action_space="eef", n_arm_joints=7):
    """
    read and parse .h5 file, and obtain the absolute actions and the action differences
    """
    with h5py.File(h5_file, "r") as fid:
        
        all_abs_gripper = np.array(fid[f"state/effector/position"], dtype=np.float32)

        if action_space == "eef":
            all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
            all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)
            all_abs_actions, all_delta_actions = get_actions_eef(
                gripper=all_abs_gripper,
                slices=slices,
                delta_act_sidx=delta_act_sidx,
                all_ends_p=all_ends_p,
                all_ends_o=all_ends_o,
            )
        elif action_space == "eef_quat":
            all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
            all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)
            all_abs_actions, all_delta_actions = get_actions_eef_quat(
                gripper=all_abs_gripper,
                slices=slices,
                delta_act_sidx=delta_act_sidx,
                all_ends_p=all_ends_p,
                all_ends_o=all_ends_o,
            )
        elif action_space == "joint":
            all_joints = np.array(fid["state/joint/position"])
            all_abs_actions, all_delta_actions = get_actions_joint(
                gripper=all_abs_gripper,
                slices=slices,
                delta_act_sidx=delta_act_sidx,
                all_joints=all_joints,
                n_arm_joints=n_arm_joints
            )
        else:
            raise NotImplementedError

    return all_abs_actions, all_delta_actions

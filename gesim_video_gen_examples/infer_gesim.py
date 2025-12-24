import os, random, math
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import Any, Dict, List
import argparse

from datetime import datetime, timedelta
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging
import cv2
import av



# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video

from utils.get_traj_maps import get_traj_maps, simple_radius_gen_func
from utils.get_ray_maps import get_ray_maps

# 本脚本用于基于已训练的 GeSim 视频生成模型，给定多视角观测、相机参数与动作序列，
# 逐块生成长视频，并将生成视频与轨迹可视化结果拼接后输出。流程概要：
# 1) 读取配置，加载 tokenizer / 文本编码器、VAE、Transformer、Scheduler、Pipeline。
# 2) 载入多视角历史帧、相机内外参、动作，构造轨迹与射线条件。
# 3) 按 chunk 滚动推理，每次生成一段视频并更新记忆帧与条件。
# 4) 最终把生成视频与轨迹可视化按时间维拼接后保存。

def load_config(config_file):
    # 从 yaml 读取配置并转换为 Namespace，方便点式访问
    cd = load(open(config_file, "r"), Loader=Loader)
    args = argparse.Namespace(**cd)
    return args

def prepare_model(args, dtype=torch.bfloat16, device="cuda:0"):

    ### 加载 tokenizer 与文本编码器
    tokenizer_class = import_custom_class(
        args.tokenizer_class, getattr(args, "tokenizer_class_path", "transformers")
    )
    textenc_class = import_custom_class(
        args.textenc_class, getattr(args, "textenc_class_path", "transformers")
    )
    cond_models = load_condition_models(
        tokenizer_class, textenc_class,
        args.pretrained_model_name_or_path if not hasattr(args, "tokenizer_pretrained_model_name_or_path") else args.tokenizer_pretrained_model_name_or_path,
        load_weights=args.load_weights
    )
    tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
    text_encoder = text_encoder.to(device, dtype=dtype).eval()

    ### 加载 VAE（可从路径或预训练模型）
    vae_class = import_custom_class(
        args.vae_class, getattr(args, "vae_class_path", "transformers")
    )
    if getattr(args, 'vae_path', False):
        vae = load_vae_models(vae_class, args.vae_path).to(device, dtype=dtype).eval()
    else:
        vae = load_latent_models(vae_class, args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
    if isinstance(vae.latents_mean, List):
        vae.latents_mean = torch.FloatTensor(vae.latents_mean)
    if isinstance(vae.latents_std, List):
        vae.latents_std = torch.FloatTensor(vae.latents_std)
    if vae is not None:
        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

    ### 加载视频 Transformer 主干
    diffusion_model_class = import_custom_class(
        args.diffusion_model_class, getattr(args, "diffusion_model_class_path", "transformers")
    )
    diffusion_model = load_diffusion_model(
        model_cls=diffusion_model_class,
        model_dir=args.diffusion_model['model_path'],
        load_weights=args.load_weights and getattr(args, "load_diffusion_model_weights", True),
        **args.diffusion_model['config']
    ).to(device, dtype=dtype)
    total_params = count_model_parameters(diffusion_model)
    print(f'Total parameters for transfomer model:{total_params}')


    ### 加载调度器（若存在预训练 scheduler 则优先使用）
    diffusion_scheduler_class = import_custom_class(
        args.diffusion_scheduler_class, getattr(args, "diffusion_scheduler_class_path", "diffusers")
    )

    if hasattr(diffusion_scheduler_class, "from_pretrained") and os.path.exists(os.path.join(args.pretrained_model_name_or_path, "scheduler")):
        scheduler = diffusion_scheduler_class.from_pretrained(os.path.join(args.pretrained_model_name_or_path, 'scheduler'))
    else:
        if hasattr(args, "diffusion_scheduler_args"):
            scheduler = diffusion_scheduler_class(**args.diffusion_scheduler_args)
        else:
            scheduler = diffusion_scheduler_class()



    # scheduler.config.final_sigmas_type = "sigma_min"

    ### 组装推理 Pipeline
    pipeline_class = import_custom_class(
        args.pipeline_class, getattr(args, "pipeline_class_path", "diffusers")
    )

    pipe = pipeline_class(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=diffusion_model
    )

    return tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe


def load_images(args, image_root, valid_cams, size=(256,192)):
    # 读取每个视角的历史帧，返回形状 (v,c,t,h,w)，并记录原始尺寸以便后续内参缩放
    n_mem = args.data["train"]["n_previous"]
    mv_images = []
    ori_sizes = []
    for cam in valid_cams:
        images = []
        for i in range(n_mem):
            img = cv2.imread(os.path.join(image_root, cam, str(i)+".png"))[:,:,::-1]
            ori_sizes.append(img.shape, )
            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            img = torch.from_numpy(np.transpose(img, (2,0,1)))
            images.append(img)
        ### c,t,h,w
        images = torch.stack(images, dim=1)
        mv_images.append(images)
    ### v,c,t,h,w
    mv_images = torch.stack(mv_images, dim=0)
    return mv_images, ori_sizes


def load_gt_sequence_from_frames(image_root, valid_cams, size, n_frames):
    """从 image_root/{cam}/{idx}.png 读取完整序列；不足补末帧，超出截断。"""
    mv_images = []
    for cam in valid_cams:
        frames = []
        for i in range(n_frames):
            img_path = os.path.join(image_root, cam, f"{i}.png")
            if not os.path.exists(img_path):
                if len(frames) == 0:
                    raise FileNotFoundError(f"GT 缺少首帧: {img_path}")
                # 不足则用最后一帧补齐
                frames.append(frames[-1].clone())
                continue
            img = cv2.imread(img_path)[:,:,::-1]
            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            img = torch.from_numpy(np.transpose(img, (2,0,1)))
            frames.append(img)
        frames = torch.stack(frames, dim=1)  # c,t,h,w
        mv_images.append(frames)
    mv_images = torch.stack(mv_images, dim=0)  # v,c,t,h,w
    return mv_images


def load_gt_sequence_from_mp4(data_root, task_id, episode_id, valid_cams, size, n_frames):
    """
    从原始数据集 mp4 读取完整 GT 序列，路径形如
    {data_root}/observations/{task_id}/{episode_id}/videos/{cam}_color.mp4 或 {cam}.mp4
    读取前 n_frames，若不足则末帧补齐。返回 (v,c,t,h,w)，值域 [-1,1]。
    """
    mv_images = []
    for cam in valid_cams:
        candidate_paths = [
            os.path.join(data_root, "observations", str(task_id), str(episode_id), "videos", f"{cam}.mp4"),
        ]
        video_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                video_path = p
                break
        if video_path is None:
            raise FileNotFoundError(f"未找到 GT 视频: {candidate_paths[0]} 或 {candidate_paths[1]}")

        frames = []
        try:
            # 用 PyAV 读取，兼容 AV1 等编码；若失败会抛异常由上层兜底到帧读取
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    if len(frames) >= n_frames:
                        break
                    arr = frame.to_ndarray(format="rgb24")
                    arr = cv2.resize(arr, size)
                    arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
                    frames.append(torch.from_numpy(np.transpose(arr, (2,0,1))))
        except Exception as e:
            raise RuntimeError(f"解码 GT 视频失败: {video_path}, err={e}")

        if len(frames) == 0:
            raise RuntimeError(f"GT 视频无有效帧: {video_path}")
        while len(frames) < n_frames:
            frames.append(frames[-1].clone())
        frames = torch.stack(frames[:n_frames], dim=1)  # c,t,h,w
        mv_images.append(frames)
    mv_images = torch.stack(mv_images, dim=0)  # v,c,t,h,w
    return mv_images


def load_cam_infos(extrinsic_root, intrinsic_root, valid_cams, orisize=None, size=(192,256)):
    # 载入相机外参/内参，并根据目标分辨率对内参进行缩放
    extrinsics = []
    intrinsics = []
    for cam in valid_cams:
        extrinsics.append(np.load(os.path.join(extrinsic_root, f"extrinsic_{cam}.npy")))
        intrinsics.append(np.load(os.path.join(intrinsic_root, f"intrinsic_{cam}.npy")))
    ### v,t,4,4
    extrinsics = np.stack(extrinsics, axis=0)
    ### v,3,3
    intrinsics = np.stack(intrinsics, axis=0)

    intrinsics[:,0,0] = intrinsics[:,0,0] * size[1] / orisize[0][1]
    intrinsics[:,0,2] = intrinsics[:,0,2] * size[1] / orisize[0][1]
    intrinsics[:,1,1] = intrinsics[:,1,1] * size[0] / orisize[0][0]
    intrinsics[:,1,2] = intrinsics[:,1,2] * size[0] / orisize[0][0]

    return extrinsics, intrinsics



def infer(
    config_file, image_root, extrinsic_root, intrinsic_root, action_path, prompt, save_path,
    seed=42, device="cuda", default_fps=30
):

    args = load_config(config_file)

    # 根据 action_chunk 与 chunk 的比例调节输出 fps，使时间尺度一致
    if "action_chunk" in args.data["train"]:
        args.data['train']['chunk']
        video_fps = default_fps // (args.data['train']['action_chunk'] // args.data['train']['chunk'])
    else:
        video_fps = default_fps

    tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe = prepare_model(args, device=device)

    valid_cams = [_+"_color" for _ in args.data["train"]["valid_cam"]]

    # 加载多视角历史帧；obs: v,c,t,h,w
    obs, ori_sizes = load_images(args, image_root, valid_cams, size=(args.data["train"]["sample_size"][1], args.data["train"]["sample_size"][0]))

    v,c,t,h,w = obs.shape

    SPATIAL_DOWN_RATIO = vae.spatial_compression_ratio
    TEMPORAL_DOWN_RATIO = vae.temporal_compression_ratio

    ### extrinsics: v,t,4,4
    ### intrinsics: v,3,3
    # 读取并缩放相机参数
    extrinsics, intrinsics = load_cam_infos(extrinsic_root, intrinsic_root, args.data["train"]["valid_cam"], orisize=ori_sizes, size=(args.data["train"]["sample_size"]))
    ### actions   : t,c
    actions = np.load(action_path)

    extrinsics = torch.FloatTensor(extrinsics)
    intrinsics = torch.FloatTensor(intrinsics)
    actions = torch.FloatTensor(actions)

    os.makedirs(save_path, exist_ok=True)

    
    # 基于动作+相机位姿生成轨迹特征 (c,v,t,h,w)，并映射到 [-1,1]
    trajs = get_traj_maps(
        actions, torch.linalg.inv(extrinsics), extrinsics, intrinsics, args.data["train"]["sample_size"], radius_gen_func=simple_radius_gen_func
    ) # trajs: c,v,t,h,w

    trajs = trajs * 2 - 1
    ori_trajs = trajs.clone()

    # 读取 GT 原始视频全序列，用于可视化对比：
    # 若提供 data_root/task_id/episode_id，则从原始 mp4 读取；否则回退到 image_root/{cam}/idx.png
    gt_video_full = None
    if getattr(args, "data_root", None) and getattr(args, "task_id", None) and getattr(args, "episode_id", None):
        try:
            gt_video_full = load_gt_sequence_from_mp4(
                args.data_root, args.task_id, args.episode_id,
                valid_cams,
                size=(args.data["train"]["sample_size"][1], args.data["train"]["sample_size"][0]),
                n_frames=trajs.shape[2]
            )
        except Exception as e:
            print(f"[warning] 读取原始 mp4 失败，回退到帧序列: {e}")
    if gt_video_full is None:
        gt_video_full = load_gt_sequence_from_frames(
            image_root, valid_cams,
            size=(args.data["train"]["sample_size"][1], args.data["train"]["sample_size"][0]),
            n_frames=trajs.shape[2]
        )

    # save_video(
    #     rearrange(trajs, 'c v t h w -> c t h (v w)', v=v),
    #     os.path.join(save_path, "trajs.mp4"),
    #     fps=video_fps
    # )

    # 构造每个像素的射线张量，后续与轨迹一起作为条件
    rays_o, rays_d = get_ray_maps(
        intrinsics.unsqueeze(dim=1).repeat(1,extrinsics.shape[1],1,1).reshape(-1,3,3), extrinsics.reshape(-1,4,4), args.data["train"]["sample_size"][0], args.data["train"]["sample_size"][1]
    )
    rays = torch.cat((rays_o, rays_d), dim=-1).reshape(trajs.shape[1], trajs.shape[2], rays_o.shape[1], rays_o.shape[2], -1)
    rays = rays.permute(4,0,1,2,3) # rays: c,v,t,h,w

    # c,v,t,h,w
    # 条件张量：轨迹 + 射线
    cond_to_concat = torch.cat((trajs, rays), dim=0)

    breakpoint()

    # 负向提示词，抑制低质视觉
    negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


    # 计算需要生成的 chunk 数；已有 n_previous 帧作为条件
    nall = trajs.shape[2]
    nchunk = int(np.ceil((nall-args.data['train']['n_previous'])/args.data['train']['chunk']))
    
    # 初始化生成结果：先放入历史观测
    videos = obs.clone()
    mem_idxes = list(range(args.data['train']['n_previous']))

    ### init conditions
    # 初始化第一段的条件：前 n_previous 帧 + 随后的 chunk 帧
    ichunk_cond_to_concat = torch.cat((
        cond_to_concat[:,:,:args.data['train']['n_previous']],
        cond_to_concat[:,:,args.data['train']['n_previous']:args.data['train']['n_previous']+args.data['train']['chunk']]
    ), dim=2)

    trajs = ichunk_cond_to_concat[:3].clone()

    breakpoint()

    # 逐块生成视频
    for ichunk in range(nchunk):

        preds = pipe.infer(
            video=obs.permute(0,2,1,3,4).to(device), # -> v, t, c, h, w
            cond_to_concat=rearrange(ichunk_cond_to_concat, "c v t h w -> v c t h w"), 
            prompt=[prompt, ],
            negative_prompt=negative_prompt,
            height=h, width=w, n_view=v,
            num_frames=args.data['train']['chunk'],
            num_inference_steps=args.num_inference_step,
            # decode_timestep=0.03,
            # decode_noise_scale=0.025,
            n_prev=args.data['train']['n_previous'],
            guidance_scale=1.0,
            merge_view_into_width=False,
            output_type="pt",
            postprocess_video=False,
        )['frames'] # preds: v c t h w , range -1 to 1 (could exceed range)

        # 将当前块结果附加到完整视频
        videos = torch.cat((videos, preds.data.cpu()), dim=2) # v c t h w

        videos = torch.clamp(videos, min=-1, max=1)

        if ichunk < nchunk-1:
            ### 更新记忆帧与下一块的条件
            ncur = videos.shape[2]
            mem_idxes = list(np.linspace(0, ncur-1, args.data['train']['n_previous']).astype(np.int16))

            obs = videos[:,:,mem_idxes].clone()
            ichunk_cond_to_concat = torch.cat((
                cond_to_concat[:,:,mem_idxes],
                cond_to_concat[:,:,args.data['train']['n_previous']+(ichunk+1)*args.data['train']['chunk']:args.data['train']['n_previous']+(ichunk+2)*args.data['train']['chunk']]
            ),dim=2)

            if ichunk_cond_to_concat.shape[2]<args.data['train']['chunk']+args.data['train']['n_previous']:
                ichunk_cond_to_concat = torch.cat([ichunk_cond_to_concat,] + [ichunk_cond_to_concat[:,:,-1:],]*(args.data['train']['chunk']-ichunk_cond_to_concat.shape[2]-args.data['train']['n_previous']), dim=2)

    # # 输出：时间维上拼接生成视频与轨迹可视化；空间上多视角横向排列
    # video_to_save = torch.cat((rearrange(videos[:,:,:ori_trajs.shape[2]], 'v c t h w -> c t h (v w)', v=v), rearrange(ori_trajs, 'c v t h w -> c t h (v w)', v=v),), dim=2)
    
    # 输出三行：1) 数据集原始 GT 视频；2) rollout 生成结果；3) 轨迹可视化（GT 轨迹）
    rollout_video_row = rearrange(videos[:,:,:ori_trajs.shape[2]], 'v c t h w -> c t h (v w)', v=v)
    traj_video_row = rearrange(ori_trajs, 'c v t h w -> c t h (v w)', v=v)
    gt_raw_row = rearrange(gt_video_full[:,:,:ori_trajs.shape[2]], 'v c t h w -> c t h (v w)', v=v)
    video_to_save = torch.cat((gt_raw_row, rollout_video_row, traj_video_row), dim=2)


    save_video(
        video_to_save,
        os.path.join(save_path, "video.mp4"),
        fps=video_fps
    )


def args_parser():
    # 命令行参数：配置路径、观测帧/相机参数/动作路径，以及输出路径与可选提示词
    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--image_root', type=str, required=True, help='Path to observation images')
    parser.add_argument('--extrinsic_root', type=str, required=True, help='Path to extrinsics')
    parser.add_argument('--intrinsic_root', type=str, required=True, help='Path to intrinsics')
    parser.add_argument('--action_path', type=str, required=True, help='Path to actions')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save outputs, used in inference stage only')
    parser.add_argument('--prompt', type=str, default="best quality, consistent and smooth motion, realistic, clear and distinct.")
    parser.add_argument('--data_root', type=str, default=None, help='(可选) 原始数据集根目录，用于从 mp4 读取 GT 视频')
    parser.add_argument('--task_id', type=str, default=None, help='(可选) 任务 ID，对应原始数据集层级')
    parser.add_argument('--episode_id', type=str, default=None, help='(可选) episode ID，对应原始数据集层级')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    ### For simplicity, this script directly load extrinsics, intrinsics and actions from .npy files.
    ### We also provide a conversion script `gesim_video_gen_examples/get_example_gesim_inputs.py` to demonstrate how to generate these .npy files.

    args = args_parser()
    print(args)

    infer(
        args.config_file, args.image_root, args.extrinsic_root, args.intrinsic_root, args.action_path, args.prompt, args.output_path
    )

import torch
import torch.nn.functional as F
from einops import rearrange


def resize_traj_and_ray(traj_n_ray, mem_size, future_size, height, width):
    '''
    traj_n_ray: (b*v) c t h w
      - time `t` is in raw-frame domain (memory + future raw), where memory frames are kept as-is (no temporal downsample),
        and future part will be temporally resampled to `future_size` (latent future length).
    '''
    if traj_n_ray.ndim != 5:
        raise ValueError(f"Expected traj_n_ray to be 5D (bv,c,t,h,w), got shape {tuple(traj_n_ray.shape)}")

    bv, c, t_raw, h0, w0 = traj_n_ray.shape
    if t_raw < mem_size:
        raise ValueError(f"traj_n_ray time length {t_raw} < mem_size {mem_size}")

    # memory part: spatial resize only
    mem = traj_n_ray[:, :, :mem_size]  # (bv,c,mem,h0,w0)
    mem = rearrange(mem, 'bv c t h w -> (bv t) c h w')
    mem = F.interpolate(mem, (height, width), mode='bilinear', align_corners=False)
    mem = rearrange(mem, '(bv t) c h w -> bv c t h w', bv=bv, t=mem_size)

    # future part: temporal + spatial resize
    fut = traj_n_ray[:, :, mem_size:]  # (bv,c,t_fut_raw,h0,w0)
    if fut.shape[2] == 0:
        # allow empty future (degenerate), return only memory resized + empty future
        out = mem[:, :, :mem_size]
        return out

    fut = F.interpolate(fut, (future_size, height, width), mode='trilinear', align_corners=False)

    out = torch.cat([mem, fut], dim=2)  # (bv,c,mem+future_size,h,w)
    return out

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import torch.utils.checkpoint
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin


class RMSNorm(nn.Module):
    """Simple RMSNorm with only weight (no bias), matching checkpoint shapes like (hidden,)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class _Conv2dWrap(nn.Module):
    """A wrapper so checkpoint keys match `*.conv.weight/bias`."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SanaSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim}) must be divisible by num_heads({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, L, C)
        b, l, c = x.shape
        qkv = self.qkv(x)  # (B, L, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        # attn_mask: broadcastable to (B, H, L, S); for self-attn S=L
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, l, c)
        out = self.proj(out)
        return out


class SanaCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim}) must be divisible by num_heads({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim, bias=True)
        self.kv_linear = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, L, C)
        # encoder_hidden_states: (B, S, C)
        b, l, c = x.shape
        s = encoder_hidden_states.shape[1]

        q = self.q_norm(self.q_linear(x))
        kv = self.kv_linear(encoder_hidden_states)
        k, v = kv.chunk(2, dim=-1)
        k = self.k_norm(k)

        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=encoder_attention_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, l, c)
        out = self.proj(out)
        return out


class SanaMLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 6):
        super().__init__()
        mid = expansion * dim  # 6C

        self.inverted_conv = _Conv2dWrap(nn.Conv2d(dim, mid, kernel_size=1, bias=True))
        self.depth_conv = _Conv2dWrap(nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=True))
        # SwiGLU halves channels: 6C -> 3C
        self.point_conv = _Conv2dWrap(nn.Conv2d(mid // 2, dim, kernel_size=1, bias=False))

        # temporal mixing conv on (B, C, T, HW)
        self.t_conv = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), bias=False)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        # x: (B, L, C) with L=t*h*w
        b, l, c = x.shape
        x_ = x.view(b, t, h, w, c).permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)  # (B*T, C, H, W)

        x_ = self.inverted_conv(x_)
        x_ = self.act(x_)
        x_ = self.depth_conv(x_)

        # SwiGLU
        x1, x2 = x_.chunk(2, dim=1)
        x_ = x1 * self.act(x2)

        x_ = self.point_conv(x_)  # (B*T, C, H, W)
        x_ = x_.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)

        # temporal mixing: treat spatial as width=H*W
        x_ = x_.view(b, c, t, h * w)
        x_ = self.t_conv(x_)
        x_ = x_.view(b, c, t, h, w).permute(0, 2, 3, 4, 1).contiguous().view(b, l, c)
        return x_


class SanaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SanaSelfAttention(dim, num_heads=num_heads)
        self.cross_attn = SanaCrossAttention(dim, num_heads=num_heads)
        self.mlp = SanaMLP(dim, expansion=6)

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        t: int,
        h: int,
        w: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, L, C), mod: (B, 6, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.unbind(dim=1)

        x_norm = self.norm(x)
        x_msa_in = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.attn(x_msa_in, attn_mask=None)

        if encoder_hidden_states is not None:
            cross_out = self.cross_attn(
                x_msa_in,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            attn_out = attn_out + cross_out

        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm = self.norm(x)
        x_mlp_in = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_mlp_in, t=t, h=h, w=w)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class SanaFinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: Tuple[int, int], out_channels: int):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, dim))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels * patch_size[0] * patch_size[1], bias=True)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C), mod: (B, 2, C)
        shift, scale = mod.unbind(dim=1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings. timesteps: (B,) float/int.
    Returns (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(float(max_period), device=timesteps.device)) * torch.arange(0, half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SanaTimestepEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(timesteps)


class MultiViewSANATransformer3DModel(ModelMixin, ConfigMixin):
    """
    Minimal SANA Video DiT-like transformer compatible with GE-Sim training loop:
    - Accepts `hidden_states` as (B, L, C_in) tokens or (B, C_in, T, H, W)
    - Returns `Transformer2DModelOutput(sample={'video': pred_tokens})`
    - Loads SANA official `.pth` checkpoint keys like `blocks.N.attn.qkv.weight`
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 16,
        cond_channels: int = 9,
        out_channels: int = 16,
        hidden_dim: int = 2240,
        num_layers: int = 20,
        num_heads: int = 20,
        patch_size: Tuple[int, int] = (2, 2),
        timestep_embed_dim: int = 256,
        text_embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size

        # input patch embed (only for latent channels to maximize checkpoint reuse)
        self.x_embedder = nn.Module()
        self.x_embedder.proj = nn.Conv3d(
            latent_channels, hidden_dim, kernel_size=(1, patch_size[0], patch_size[1]), stride=(1, patch_size[0], patch_size[1]), bias=True
        )

        # optional extra embeds (cond / condition_mask)
        self.cond_embedder = nn.Module()
        self.cond_embedder.proj = nn.Conv3d(
            cond_channels, hidden_dim, kernel_size=(1, patch_size[0], patch_size[1]), stride=(1, patch_size[0], patch_size[1]), bias=True
        )
        self.mask_embedder = nn.Module()
        self.mask_embedder.proj = nn.Conv3d(
            1, hidden_dim, kernel_size=(1, patch_size[0], patch_size[1]), stride=(1, patch_size[0], patch_size[1]), bias=True
        )

        # learned absolute pos embed (will be interpolated in 1D if token length mismatches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1800, hidden_dim))

        # timestep embeddings: sinus(256) -> hidden_dim -> hidden_dim, then to 6*hidden_dim
        self.t_embedder = SanaTimestepEmbedder(timestep_embed_dim, hidden_dim)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

        # text projection (checkpoint expects encoder_hidden_states dim == hidden_dim)
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim, bias=False)
        self.attention_y_norm = RMSNorm(hidden_dim)

        self.blocks = nn.ModuleList([SanaBlock(hidden_dim, num_heads=num_heads) for _ in range(num_layers)])

        self.final_layer = SanaFinalLayer(hidden_dim, patch_size=patch_size, out_channels=out_channels)
        self.final_modulation = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)

        # Important: GE-Sim training/pipeline code treats token inputs as "unpacked".
        # Setting this flag makes `CustomPipeline` use patch_size=1 in its pack/unpack helpers.
        self.unpack_in_forward = True

        # Trainer expects this to work. We'll implement real activation checkpointing for memory savings.
        self.gradient_checkpointing = False
        self._supports_gradient_checkpointing = True

    def enable_gradient_checkpointing(self, *args, **kwargs) -> None:
        self.gradient_checkpointing = True

    def _checkpoint(self, fn, *inputs):
        """
        Compatibility wrapper across torch versions.
        Prefer non-reentrant checkpointing when available (saves memory and avoids some autograd edge cases).
        """
        sig = inspect.signature(torch.utils.checkpoint.checkpoint)
        if "use_reentrant" in sig.parameters:
            return torch.utils.checkpoint.checkpoint(fn, *inputs, use_reentrant=False)
        return torch.utils.checkpoint.checkpoint(fn, *inputs)

    def _get_pos_embed(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pe = self.pos_embed.to(device=device, dtype=dtype)
        if pe.shape[1] == seq_len:
            return pe
        # 1D interpolate on token axis (no need to know original grid factorization)
        pe_t = pe.transpose(1, 2)  # (1, C, L0)
        pe_t = F.interpolate(pe_t, size=seq_len, mode="linear", align_corners=False)
        return pe_t.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        n_view: int = 1,
        return_video: bool = True,
        **kwargs,
    ):
        # Accept flattened tokens: (B, L, C_in)
        input_is_flat = hidden_states.ndim == 3
        if input_is_flat:
            if num_frames is None or height is None or width is None:
                raise ValueError("When `hidden_states` is flattened, must provide num_frames/height/width.")
            b, l, c_in = hidden_states.shape
            if l != num_frames * height * width:
                raise ValueError(f"hidden_states length {l} != num_frames*height*width ({num_frames*height*width})")
            x_5d = rearrange(hidden_states, "b (t h w) c -> b c t h w", t=num_frames, h=height, w=width)
        elif hidden_states.ndim == 5:
            x_5d = hidden_states
            b, c_in, num_frames, height, width = x_5d.shape
        else:
            raise ValueError(f"Unsupported hidden_states shape: {tuple(hidden_states.shape)}")

        # split channels: latent + optional cond
        latent = x_5d[:, : self.latent_channels]
        cond = None
        if c_in > self.latent_channels:
            cond = x_5d[:, self.latent_channels : self.latent_channels + self.cond_channels]

        # patch embed
        x = self.x_embedder.proj(latent)  # (B, C, T, H', W')
        if cond is not None and self.cond_channels > 0:
            x = x + self.cond_embedder.proj(cond)

        if condition_mask is not None:
            if condition_mask.ndim == 2:
                # (B, L) -> (B, 1, T, H, W)
                cm = rearrange(condition_mask, "b (t h w) -> b 1 t h w", t=num_frames, h=height, w=width)
            elif condition_mask.ndim == 5:
                cm = condition_mask
            else:
                raise ValueError(f"Unsupported condition_mask shape: {tuple(condition_mask.shape)}")
            cm = cm.to(dtype=x.dtype, device=x.device)
            # downsample to patch grid via strided conv (same as patch stride)
            x = x + self.mask_embedder.proj(cm)

        # flatten to tokens
        b, c, t_p, h_p, w_p = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(b, t_p * h_p * w_p, c)  # (B, Lp, C)
        x = x + self._get_pos_embed(x.shape[1], dtype=x.dtype, device=x.device)

        # timestep -> modulation
        if timestep.ndim == 2:
            # pixel-wise timestep: (B, L_raw); use per-video max as global t
            # (masked memory tokens are 0; max keeps the "future/noisy" level)
            timestep_ = timestep.float().amax(dim=-1)
        elif timestep.ndim == 1:
            timestep_ = timestep.float()
        else:
            timestep_ = timestep.flatten().float()

        t_emb = _sinusoidal_timestep_embedding(timestep_, dim=self.config.timestep_embed_dim)
        t_emb = self.t_embedder(t_emb)  # (B, C)
        block_mod = self.t_block(t_emb).view(b, 6, c)  # (B, 6, C)

        # text
        enc = None
        enc_mask = None
        if encoder_hidden_states is not None:
            if encoder_hidden_states.shape[-1] != c:
                encoder_hidden_states = self.text_proj(encoder_hidden_states)
            enc = self.attention_y_norm(encoder_hidden_states)

            if encoder_attention_mask is not None:
                # expect (B, S) with 1 for keep
                if encoder_attention_mask.dtype != torch.float32 and encoder_attention_mask.dtype != torch.float16 and encoder_attention_mask.dtype != torch.bfloat16:
                    mask = encoder_attention_mask.to(dtype=torch.float32)
                else:
                    mask = encoder_attention_mask
                # additive mask: 0 for keep, -inf for mask
                mask = (1.0 - mask) * -1e9
                enc_mask = mask[:, None, None, :]  # (B,1,1,S)

            # GE-Sim uses multi-view: hidden_states batch is (b*v) but prompt embeds are usually batch=b.
            # Also for CFG in pipeline, hidden_states can be (2*b*v) while text is (2*b) or (b).
            # Make encoder batch match token batch by repeating along batch dimension.
            if enc.shape[0] != b:
                if b % enc.shape[0] != 0:
                    raise ValueError(
                        f"encoder_hidden_states batch {enc.shape[0]} does not match hidden_states batch {b} "
                        f"and is not a divisor. Please ensure prompts are repeated per view (n_view) and CFG."
                    )
                rep = b // enc.shape[0]
                enc = enc.repeat_interleave(rep, dim=0)
                if enc_mask is not None:
                    enc_mask = enc_mask.repeat_interleave(rep, dim=0)

        for i, blk in enumerate(self.blocks):
            mod_i = block_mod + blk.scale_shift_table.unsqueeze(0)
            if self.training and self.gradient_checkpointing:
                def _blk_forward(x_in):
                    return blk(
                        x_in,
                        mod=mod_i,
                        t=t_p,
                        h=h_p,
                        w=w_p,
                        encoder_hidden_states=enc,
                        encoder_attention_mask=enc_mask,
                    )

                x = self._checkpoint(_blk_forward, x)
            else:
                x = blk(x, mod=mod_i, t=t_p, h=h_p, w=w_p, encoder_hidden_states=enc, encoder_attention_mask=enc_mask)

        # final modulation (2,C)
        final_mod = self.final_modulation(t_emb).view(b, 2, c) + self.final_layer.scale_shift_table.unsqueeze(0)
        x = self.final_layer(x, mod=final_mod)  # (B, Lp, out_ch * p_h * p_w)

        # unpatchify back to latent grid
        p_h, p_w = self.patch_size
        x = x.view(b, t_p, h_p, w_p, self.out_channels, p_h, p_w)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous().view(b, self.out_channels, t_p, h_p * p_h, w_p * p_w)

        if input_is_flat:
            x = rearrange(x, "b c t h w -> b (t h w) c")

        out: Dict[str, Any] = {"video": x}
        if not return_dict:
            return (out,)
        return Transformer2DModelOutput(sample=out)



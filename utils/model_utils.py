from typing import Dict, Optional, Union
import json
import os
import sys

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import save_model, load_file, save_file


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def count_model_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def load_index_file(index_filename):
    checkpoint_folder = os.path.split(index_filename)[0]
    with open(index_filename) as f:
        index = json.loads(f.read())

    if "weight_map" in index:
        index = index["weight_map"]
    checkpoint_files = sorted(list(set(index.values())))
    checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]
    state_dict = {}
    for checkpoint_file in checkpoint_files:
        state_dict.update(load_file(checkpoint_file))
    return state_dict

def _find_mismatched_keys(
    state_dict,
    model_state_dict,
    loaded_keys,
):
    mismatched_keys = []
    for checkpoint_key in loaded_keys:
        model_key = checkpoint_key

        if (
            model_key in model_state_dict
            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
        ):
            mismatched_keys.append(
                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
            )
            del state_dict[checkpoint_key]

    return mismatched_keys

def load_checkpoints(model, pretrained_ckpt, strict=False, ignore_mismatched_sizes=True):
    """
    Load safetensors model state dict file.
    """

    def _extract_state_dict(obj):
        # Common training checkpoints wrap weights under various keys.
        if not isinstance(obj, dict):
            return None
        for k in ["state_dict", "model", "ema", "params", "weights", "net", "module"]:
            v = obj.get(k, None)
            if isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
                return v
        # Some checkpoints store {'model': {'state_dict': ...}}
        for k in ["model", "ema", "net", "module"]:
            v = obj.get(k, None)
            if isinstance(v, dict):
                vv = v.get("state_dict", None)
                if isinstance(vv, dict) and vv and all(isinstance(x, torch.Tensor) for x in vv.values()):
                    return vv
        return None

    def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    def _score_match(sd: Dict[str, torch.Tensor], model_keys: set) -> int:
        # count direct key matches
        return sum(1 for k in sd.keys() if k in model_keys)

    def _normalize_state_dict_keys(sd: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
        """
        Try to align checkpoint keys to current model's keys by stripping common prefixes.
        This is especially important for original-repo `.pth` checkpoints (e.g. SANA).
        """
        model_keys = set(model.state_dict().keys())
        if not sd:
            return sd

        # 1) unwrap DistributedDataParallel prefix
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}

        # 2) try common container prefixes. We choose the best match w.r.t model keys.
        candidates = [sd]
        prefixes = [
            "model.",
            "ema.",
            "diffusion_model.",
            "transformer.",
            "unet.",
            "net.",
            "generator.",
        ]
        for p in prefixes:
            if any(k.startswith(p) for k in sd.keys()):
                candidates.append(_strip_prefix(sd, p))

        # also allow chained stripping like "model.diffusion_model."
        chained = ["model.diffusion_model.", "model.transformer.", "ema.diffusion_model.", "ema.transformer."]
        for p in chained:
            if any(k.startswith(p) for k in sd.keys()):
                candidates.append(_strip_prefix(sd, p))

        # pick best by match count
        best = max(candidates, key=lambda x: _score_match(x, model_keys))
        return best

    # In this case we have many shards to load
    if os.path.isdir(pretrained_ckpt):
        state_dict = load_index_file(os.path.join(pretrained_ckpt, "diffusion_pytorch_model.safetensors.index.json"))
    # torch checkpoint (.pth/.pt) - common for original repos
    elif str(pretrained_ckpt).endswith((".pth", ".pt")):
        obj = torch.load(pretrained_ckpt, map_location="cpu")
        extracted = _extract_state_dict(obj)
        if extracted is not None:
            state_dict = extracted
        elif isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj
        else:
            raise ValueError(f"Unsupported checkpoint object type: {type(obj)} from {pretrained_ckpt}")
    else:
        # in this case we need give the file path
        state_dict = load_file(pretrained_ckpt)

    # normalize keys (prefix stripping etc.)
    # if isinstance(state_dict, dict):
        state_dict = _normalize_state_dict_keys(state_dict, model)

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        if ignore_mismatched_sizes:
            model_state_dict = model.state_dict()
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                list(state_dict.keys()),
            )
        else:
            mismatched_keys = []
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(">>> mismatched_keys: %s" % mismatched_keys)
        print(">>> missing: %s" % missing)
        print(">>> unexpected: %s" % unexpected)
    print(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)



def load_condition_models(
    tokenizer_class,
    textenc_class,
    model_id: str = "a-r-r-o-w/LTX-Video-0.9.1-diffusers",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    load_weights: bool = True,
    **kwargs,
) -> Dict[str, nn.Module]:
    tokenizer = tokenizer_class.from_pretrained(
        model_id,
        subfolder="tokenizer",
        revision=revision,
        cache_dir=cache_dir
    )

    if load_weights:
        text_encoder = textenc_class.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=text_encoder_dtype,
            revision=revision,
            cache_dir=cache_dir
        )
    else:
        # logger.warning('You are not lodding the checkpoint of the text Embedder, please check the code!!!')
        config = textenc_class.config_class.from_pretrained(
            model_id,
            subfolder="text_encoder",
            revision=revision,
            cache_dir=cache_dir
        )
        text_encoder = textenc_class(config)  # 仅初始化模型，不加载权重

    return {"tokenizer": tokenizer, "text_encoder": text_encoder}


def load_latent_models(
    model_cls,
    model_id,
    vae_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    vae = model_cls.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}


def load_diffusion_model(model_cls, model_dir, load_weights=True, **kwargs):
    model = model_cls(**kwargs)
    print(model_dir)
    if load_weights:
        load_checkpoints(model, pretrained_ckpt=model_dir)
    return model


def load_vae_models(model_cls, model_dir, load_weights=True):
    with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
        vae_kwargs = json.load(f)
    model = model_cls(**vae_kwargs)
    if load_weights:
        load_checkpoints(model, pretrained_ckpt=os.path.join(model_dir, 'diffusion_pytorch_model.safetensors'))
    else:
        print('You are not loading the weights of the vae model, please check your code.')
        pass
    return model


def forward_pass(
    model,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    num_frames: int,
    height: int,
    width: int,
    n_view: int = 1,
    frame_rate = 30,
    temporal_compression_ratio = 8,
    spatial_compression_ratio = 32,
    **kwargs,
) -> torch.Tensor:
    latent_frame_rate = frame_rate / temporal_compression_ratio
    rope_interpolation_scale = [1 / latent_frame_rate, spatial_compression_ratio, spatial_compression_ratio]
    
    denoised_latents = model(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        encoder_attention_mask=prompt_attention_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        n_view=n_view,
        rope_interpolation_scale=rope_interpolation_scale,
        return_dict=False,
        **kwargs,
    )[0]
    return {"latents": denoised_latents}

# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

"""
GE-Sim(Cosmos2) 推理 Pipeline（多视角 Video-to-World / Video Rollout）

本文件基于 diffusers 的 Cosmos2 Video2World pipeline 改造，核心目标是：
- 输入：多视角“记忆帧”(memory video) + 文本 prompt + 额外条件(cond_to_concat：轨迹(traj) + 射线(ray) + padding mask 等)
- 输出：未来一段视频（rollout），可选择输出为 torch / numpy / PIL 等格式

在具身智能 World Model 语境下：
- 多视角 n_view：同一时刻来自多个相机的观测（例如 head / hand_left / hand_right）
- 记忆帧 n_prev：历史观测帧，用于条件（模型只“生成未来”，不重建这部分）
- 未来帧 num_frames：本次要生成的未来帧数量（对应 chunk size）
- cond_to_concat：与视频 latent 在 channel 维拼接的几何条件（轨迹热力图、射线图等）

张量约定（与仓库其他代码保持一致）：
- video 输入通常是 (b*v, t, c, h, w) 或 (b*v, c, t, h, w)（具体取决于 preprocess）
- pipeline 内部会统一到 (b*v, c, t, h, w)
- VAE latent 一般是 (b*v, z_dim, t_lat, h_lat, w_lat)

注意：
本文件同时从 `models.pipeline.custom_pipeline` 导入了 `retrieve_timesteps/retrieve_latents`，
但下方又定义了同名函数（来自 diffusers 示例）。Python 以“后定义覆盖先导入”为准，
因此实际使用的是本文件内定义的版本。这里保留该结构是为了与上游代码对齐。
"""

### This file is modified from https://github.com/huggingface/diffusers/blob/f064b3bf73e479051ed4255d98afad4259a6f012/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py


import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import os.path as osp
import torch
from einops import rearrange

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from models.pipeline.gesim_cosmos2_pipeline_utils.pipeline_output import CosmosPipelineOutput
    
from models.pipeline.custom_pipeline import retrieve_timesteps, retrieve_latents

from models.pipeline.gesim_cosmos2_pipeline_utils.video_processor import VideoProcessor

from utils.geometry_utils import resize_traj_and_ray


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class GeSimCosmos2Pipeline(DiffusionPipeline):
    r"""
    Pipeline for video-to-world generation using [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. Cosmos uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-11b](https://huggingface.co/google-t5/t5-11b) variant.
        tokenizer (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CosmosTransformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # We mark safety_checker as optional here to get around some test failures, but it is not really optional
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        text_encoder,
        tokenizer,
        transformer,
        vae,
        scheduler,
        safety_checker=None,
    ):
        super().__init__()

        # if safety_checker is None:
        #     safety_checker = CosmosSafetyChecker()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.sigma_max = 80.0
        self.sigma_min = 0.002
        self.sigma_data = 1.0
        self.final_sigmas_type = "sigma_min"
        if self.scheduler is not None:
            self.scheduler.register_to_config(
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                sigma_data=self.sigma_data,
                final_sigmas_type=self.final_sigmas_type,
            )

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_length=True,
            return_offsets_mapping=False,
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=prompt_attention_mask
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        lengths = prompt_attention_mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            prompt_embeds[i, length:] = 0

        return prompt_embeds

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        将 prompt 编码为文本 encoder hidden states（Cosmos2 使用 T5）。

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 1) 正向 prompt
        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        # 2) 负向 prompt（CFG 用）
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds


    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def infer(
        self,
        image: PipelineImageInput = None,
        video: List[PipelineImageInput] = None,  # 输入视频：常见形状 (b*v, t, c, h, w)（时间在前）
        cond_to_concat: List[PipelineImageInput] = None,  # 额外条件：形状 (b*v, c_cond, t_cond, h, w)，t_cond 可与 video 不同
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93, # 本次生成的未来帧数（chunk size，注意：不包含 memory）
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        fps: int = 16,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        sigma_conditioning: float = 0.0001,
        n_view: int = 3,   # 视角数（相机数）
        n_prev: int = 4,   # 记忆帧数（作为条件输入，不生成）
        merge_view_into_width: bool = False,
        postprocess_video: bool = True,
        show_progress: bool = True,
        **kwargs,
    ):
        r"""
        GE-Sim rollout 推理入口。

        核心思路：
        - 只把“记忆帧”编码为 conditioning_latents
        - 未来帧 latents 从高斯噪声初始化，通过扩散反推逐步去噪
        - 在 Transformer 输入通道维拼接 cond_to_concat（轨迹/射线等），让模型具备几何约束

        关键形状（简化）：
        - memory video:            (b*v, c, n_prev, h, w)
        - future latents(noise):   (b*v, z_dim, t_lat, h_lat, w_lat)
        - conditioning_latents:    (b*v, z_dim, n_prev, h_lat, w_lat)
        - cond_to_concat_resized:  (b*v, c_cond, n_prev+t_lat, h_lat, w_lat)

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*):
                The image to be used as a conditioning input for the video generation.
            video (`List[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*):
                The video to be used as a conditioning input for the video generation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `704`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `93`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `35`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `7.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`.
            fps (`int`, defaults to `16`):
                The frames per second of the generated video.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`CosmosPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
                the prompt is shorter than this length, it will be padded.
            sigma_conditioning (`float`, defaults to `0.0001`):
                The sigma value used for scaling conditioning latents. Ideally, it should not be changed or should be
                set to a small value close to zero.

        Examples:

        Returns:
            [`~CosmosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`CosmosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # if self.safety_checker is None:
        #     raise ValueError(
        #         f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
        #         "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
        #         f"Please ensure that you are compliant with the license agreement."
        #     )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1) 参数检查：分辨率必须能被下采样整除等
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # if self.safety_checker is not None:
        #     self.safety_checker.to(device)
        #     if prompt is not None:
        #         prompt_list = [prompt] if isinstance(prompt, str) else prompt
        #         for p in prompt_list:
        #             if not self.safety_checker.check_text_safety(p):
        #                 raise ValueError(
        #                     f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
        #                     f"prompt abides by the NVIDIA Open Model License Agreement."
        #                 )
        #     self.safety_checker.to("cpu")

        # 2) batch_size（这里的 batch 指“文本 prompt 的 batch”，后面会乘上 n_view）
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3) 文本编码：得到 prompt_embeds 与 negative_prompt_embeds（CFG）
        assert num_videos_per_prompt == 1  # TODO: only support num_videos_per_prompt=1 for now
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        # 4) 构造扩散时间表（Cosmos 使用 sigma schedule）
        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
        if self.scheduler.config.final_sigmas_type == "sigma_min":
            # Replace the last sigma (which is zero) with the minimum sigma value
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        # 5) 预处理视频输入，并准备 future latents（纯噪声初始化）与 conditioning_latents（由 memory 编码）
        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        if image is not None:
            video = self.video_processor.preprocess(image, height, width).unsqueeze(2)
        else:
            # 输入 video 常见是 (b*v, t, c, h, w)，预处理后统一为 (b*v, c, t, h, w)
            video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=vae_dtype)

        # num_channels_latents = self.transformer.config.in_channels - 1
        num_channels_latents = self.vae.z_dim
        # 只取前 n_prev 帧作为记忆条件（多出来的帧会被丢弃）
        if video.shape[2] > n_prev:  # pyright: ignore
            video = video[:, :, :n_prev]  # pyright: ignore
        # prepare_latents：返回 future latents（噪声）、memory latents（conditioning_latents）与 mask/indicator
        latents, conditioning_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask = self.prepare_latents(
            video,  # memory only! (b v) c t h w
            batch_size * n_view,
            num_channels_latents,
            height,
            width,
            num_frames,
            self.do_classifier_free_guidance,
            torch.float32,
            device,
            generator,
            latents,
        )  # indicator and mask here are all about memory video, action condition not included
           # latents here only contains future
        unconditioning_latents = None

        cond_mask = cond_mask.to(transformer_dtype)
        if self.do_classifier_free_guidance:
            uncond_mask = uncond_mask.to(transformer_dtype)
            unconditioning_latents = conditioning_latents

        # padding_mask：Cosmos2 的输入接口需要（这里全 0，相当于不 mask）
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        sigma_conditioning = torch.tensor(sigma_conditioning, dtype=torch.float32, device=device)
        t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

        # 6) 去噪循环：每个 timestep 计算 transformer 噪声预测并更新 latents
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if not show_progress:
            self.set_progress_bar_config(disable=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]

                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                # timestep = current_t.view(1, 1, 1, 1, 1).expand(
                #     latents.size(0), -1, latents.size(2), -1, -1
                # )  # [B, 1, T, 1, 1]
                # timestep 张量会扩展到 (B, 1, T_total, 1, 1)，其中 T_total=memory_lat + future_lat
                timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2)+conditioning_latents.size(2), -1, -1
                )  # [B, 1, T, 1, 1]
                # timestep = timestep * 1000  # LTX, timestep ranges from 1 to 1000

                # cond_latent：将 future latents 乘上缩放系数，随后与 memory latents 拼接形成完整时序
                cond_latent = latents * c_in  # all noise with a scale factor
                # replace :n_prev frames with clean video latents
                # cond_latent = cond_indicator * conditioning_latents + (1 - cond_indicator) * cond_latent
                # frame 维拼接：memory 在前，future 在后
                cond_latent = torch.cat([conditioning_latents, cond_latent], dim=2)

                n_fut = (num_frames - 1) // self.vae_scale_factor_temporal + 1

                # 将几何条件 resize 到 latent 空间分辨率与时间长度（memory + future_lat）
                cond_to_concat = cond_to_concat.to(device=cond_latent.device, dtype=cond_latent.dtype)
                cond_to_concat_resze = resize_traj_and_ray(cond_to_concat, 
                    mem_size=n_prev, future_size=n_fut,
                    height=cond_latent.shape[-2], width=cond_latent.shape[-1]
                )

                # channel 维拼接：把 traj/ray 等条件与视频 latent 拼到一起送入 transformer
                cond_latent = torch.cat([cond_latent, cond_to_concat_resze], dim=1)
                cond_latent = cond_latent.to(transformer_dtype)  # (b v) c t h w
                cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                cond_timestep = cond_timestep.to(transformer_dtype)

                # transformer 输出噪声预测（这里取 video 分支），并去掉 memory 部分
                noise_pred = self.transformer(
                    hidden_states=cond_latent,
                    timestep=cond_timestep,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps,
                    condition_mask=cond_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                    height=cond_latent.shape[-2],
                    width=cond_latent.shape[-1],
                    n_view=n_view,
                    num_frames=n_fut+n_prev,
                    return_video=True,
                )[0]['video']

                noise_pred = noise_pred[:, :, n_prev:]  # remove memory
                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(transformer_dtype)
                # noise_pred = cond_indicator * conditioning_latents + (1 - cond_indicator) * noise_pred

                # CFG：再跑一次 negative prompt 的预测，做 guidance 合成
                if self.do_classifier_free_guidance:
                    uncond_latent = latents * c_in
                    # replace :n_prev frames with clean video latents
                    # uncond_latent = uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * uncond_latent
                    uncond_latent = torch.cat([conditioning_latents, uncond_latent], dim=2)  # frame
                    # 这里使用未 resize 的 cond_to_concat（与 cond 分支略不同）；保持原作者实现不改动
                    uncond_latent = torch.cat([uncond_latent, cond_to_concat.to(device=cond_latent.device, dtype=cond_latent.dtype)], dim=1)
                    uncond_latent = uncond_latent.to(transformer_dtype)
                    uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                    uncond_timestep = uncond_timestep.to(transformer_dtype)
                    
                    noise_pred_uncond = self.transformer(
                        hidden_states=uncond_latent,
                        timestep=uncond_timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        fps=fps,
                        condition_mask=uncond_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                        n_view=n_view
                    )[0]
                    noise_pred_uncond = noise_pred_uncond[:, :, n_prev:]  # remove memory
                    noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(transformer_dtype)
                    # noise_pred_uncond = (
                    #     uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                    # )
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                # scheduler.step 需要的“velocity/noise”形式
                noise_pred = (latents - noise_pred) / current_sigma
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # 7) 解码：将 latents 反标准化并经 VAE decode 得到像素空间视频
        if not output_type == "latent":
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean  # config.sigma_data=1.0

            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            if merge_view_into_width:
                video = rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)  # should be vw not wv !!!

            # if self.safety_checker is not None:
            #     self.safety_checker.to(device)
            #     video = self.video_processor.postprocess_video(video, output_type="np")
            #     video = (video * 255).astype(np.uint8)
            #     video_batch = []
            #     for vid in video:
            #         vid = self.safety_checker.check_video_safety(vid)
            #         video_batch.append(vid)
            #     video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
            #     video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
            #     video = self.video_processor.postprocess_video(video, output_type=output_type)
            #     self.safety_checker.to("cpu")
            # else:
            # 后处理：输出类型转换、范围裁剪等
            if postprocess_video:
                video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)



    def prepare_latents(
        self,
        video: torch.Tensor,  # (b v) c t h w
        batch_size: int,  # (b v)
        num_channels_latents: 16,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,  # memory not included
        do_classifier_free_guidance: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        构造扩散所需的初始 latent 与 mask。

        输入：
        - video: (b*v, c, n_prev, h, w) 的记忆帧（像素空间，已 preprocess 到 [-1,1]）

        输出：
        - latents:              (b*v, z_dim, t_lat, h_lat, w_lat) 未来段的“纯噪声”初始化 latent
        - init_latents:         (b*v, z_dim, n_prev, h_lat, w_lat) 由 VAE 编码得到的记忆帧 latent
        - cond_indicator/mask:  指示哪些时间步是 conditioning（记忆帧）

        说明：
        Cosmos2 的时间下采样会使 latent 帧数 t_lat = floor((num_frames-1)/temporal_down)+1。
        """
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_cond_frames = video.size(2)
        num_cond_latent_frames = num_cond_frames  # encode memory separately
        init_latents = [retrieve_latents(self.vae.encode(video[:, :, it].unsqueeze(2)), generator) for it in range(video.size(2))]

        init_latents = torch.cat(init_latents, dim=2).to(dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        init_latents = (init_latents - latents_mean) / latents_std * self.scheduler.config.sigma_data  # sigma_data=1.0

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.config.sigma_max  # sigma_max = 80.0

        padding_shape = (batch_size, 1, num_cond_latent_frames+num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        cond_indicator = latents.new_zeros(1, 1, num_cond_latent_frames+num_latent_frames, 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

        uncond_indicator = uncond_mask = None  # equals cond_indicator and cond_mask
        if do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, num_cond_latent_frames+num_latent_frames, 1, 1)
            uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding

        return latents, init_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask

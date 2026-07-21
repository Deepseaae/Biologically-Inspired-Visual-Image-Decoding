"""
This file is for training priorunet using local models.
Modified: removed image encoding part, directly use provided image features.
Added: Multi IP-Adapter + dynamic scale support.
"""
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
from pathlib import Path
import sys
import os
from PIL import Image

# Add local path to system path to ensure local modules can be imported
sys.path.append(str(Path(__file__).parent))

from config_manager import MODEL_PATHS


@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def generate_ip_adapter_embeds_dynamic(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_embeds: Optional[List[torch.FloatTensor]] = None,  # Changed to list
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        scale_schedule_fn: Optional[Callable[[int], List[Dict]]] = None,  # New: function returning scale list based on timestep
        **kwargs,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
            Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
            Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        denoising_end (`float`, *optional*):
            When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
            completed before it is intentionally prematurely terminated. As a result, the returned sample will
            still retain a substantial amount of noise as determined by the discrete timesteps selected by the
            scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
            "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
            Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
        guidance_scale (`float`, *optional*, defaults to 5.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        ip_adapter_embeds: (`FloatTensor`, *optional*): Optional image embeddings to work with IP Adapters.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
            of a plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.
        original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
            `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
            explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
            `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
            `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            For most cases, `target_size` should be set to the desired height and width of the generated image. If
            not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
            section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a specific image resolution. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a target image resolution. It should be as same
            as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """
    # Keep original parameter handling (callback etc.)
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )

    # 0. Default height and width to unet
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    # Handle multiple IP-Adapter embeddings
    if ip_adapter_embeds is not None:
        # 1) Ensure it is a list (adapter dimension)
        if not isinstance(ip_adapter_embeds, list):
            ip_adapter_embeds = [ip_adapter_embeds]

        image_embeds_list = []

        # 2) Process each adapter
        for embeds in ip_adapter_embeds:
            embeds = embeds.to(device=device, dtype=prompt_embeds.dtype)

            # Ensure (B, 1, D)
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(1)
            elif embeds.dim() != 3:
                raise ValueError(f"ip_adapter_embeds must be 2D or 3D, got shape={tuple(embeds.shape)}")

            # 3) CFG: construct [uncond, cond] along batch dimension
            # prompt_embeds is already torch.cat([negative, positive], dim=0)
            # So image_embeds must also align: first B are zeros, last B are embeds
            if self.do_classifier_free_guidance:
                negative_embeds = torch.zeros_like(embeds)
                embeds = torch.cat([negative_embeds, embeds], dim=0)  # (2B, 1, D)

            image_embeds_list.append(embeds)

        # Put the list into added_cond_kwargs
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
            "image_embeds": image_embeds_list,
        }
    else:
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    # 8.1 Apply denoising_end
    if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                self.scheduler.config.num_train_timesteps
                - (self.denoising_end * self.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # Dynamically adjust IP-Adapter scales if schedule function is provided
            if scale_schedule_fn is not None:
                # Get scale list based on current timestep t
                scales = scale_schedule_fn(t.item())
                self.set_ip_adapter_scale(scales)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            num_adapters = len(self.unet.encoder_hid_proj.image_projection_layers)

            print("Current timestep:", t.item())
            print(
                f"image_embeds={len(added_cond_kwargs['image_embeds'])}, "
                f"adapters={num_adapters}"
            )

            assert len(added_cond_kwargs["image_embeds"]) == num_adapters, \
                f"image_embeds={len(added_cond_kwargs['image_embeds'])}, adapters={num_adapters}"


            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )
                add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if not output_type == "latent":
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusionXLPipelineOutput(images=image)


class Generator4Embeds:
    """Image generator using local models (supports multiple IP-Adapters + dynamic scales)"""

    def __init__(self, num_inference_steps=1, device='cuda', 
                 sdxl_local_path=None, ip_adapter_local_path=None,
                 num_adapters=4):  # New: number of adapters
        self.num_inference_steps = num_inference_steps
        self.dtype = torch.float16
        self.device = device
        
        # Set model paths
        if sdxl_local_path:
            self.sdxl_path = sdxl_local_path
        else:
            self.sdxl_path = MODEL_PATHS.get_sdxl_path()
            
        if ip_adapter_local_path:
            self.ip_adapter_path = ip_adapter_local_path
        else:
            self.ip_adapter_path = MODEL_PATHS.ip_adapter_path
        
        print(f"Using SDXL-Turbo model path: {self.sdxl_path}")
        print(f"Using IP-Adapter model path: {self.ip_adapter_path}")
        print(f"Loading {num_adapters} IP-Adapters")
        
        # Load SDXL-Turbo from local path
        print("Loading SDXL-Turbo model...")
        pipe = DiffusionPipeline.from_pretrained(
            self.sdxl_path, 
            torch_dtype=torch.float16, 
            variant="fp16",
            local_files_only=True
        )
        pipe.to(device)
        # Bind dynamic generation function
        pipe.generate_ip_adapter_embeds_dynamic = generate_ip_adapter_embeds_dynamic.__get__(pipe)
        
        # Load multiple IP-Adapters (using the same weight file, but can be set independently)
        # Here assume all adapters use the same weight file (can be modified to different files)
        weight_names = ["ip-adapter_sdxl_vit-h.safetensors"] * num_adapters
        try:
            pipe.load_ip_adapter(
                self.ip_adapter_path,
                subfolder="sdxl_models",
                weight_name=weight_names,
                torch_dtype=torch.float16,
                local_files_only=True
            )
        except Exception as e:
            print(f"Failed to load IP-Adapter: {e}")
            print("Attempting to load weight file directly...")
            pipe.load_ip_adapter(
                self.ip_adapter_path,
                torch_dtype=torch.float16,
                local_files_only=True
            )

        # Set initial scales (can be dynamically overridden during generation)
        # Here set default values; they will be specified via scale_schedule_fn during generation
        default_scales = [1.0] * num_adapters
        pipe.set_ip_adapter_scale(default_scales)
        
        self.pipe = pipe
        self.num_adapters = num_adapters

    def generate(self, image_embeds_list, text_prompt='', generator=None, scale_schedule_fn=None):
        """Generate an image from multiple image embeddings, with dynamic scale scheduling
        
        Args:
            image_embeds_list: List[torch.Tensor], each of shape (N, D) or (D,)
            text_prompt: text prompt
            generator: torch.Generator
            scale_schedule_fn: function that takes timestep t (int) and returns a scale list (length num_adapters)
        """
        if image_embeds_list is None or len(image_embeds_list) == 0:
            raise ValueError("image_embeds_list cannot be empty")
        
        # Ensure each embedding is a tensor with correct dimensions
        processed_embeds = []
        for embeds in image_embeds_list:
            if not isinstance(embeds, torch.Tensor):
                embeds = torch.tensor(embeds)
            if len(embeds.shape) == 1:
                embeds = embeds.unsqueeze(0)  # [D] -> [1, D]
            elif len(embeds.shape) > 2:
                embeds = embeds.view(embeds.shape[0], -1)
            
            # Check if dimension matches IP-Adapter requirements (e.g., 1024)
            expected_dim = 1024
            if embeds.shape[1] != expected_dim:
                print(f"Warning: image embedding dimension is {embeds.shape[1]}, expected {expected_dim}")
                if embeds.shape[1] < expected_dim:
                    padding = torch.zeros(embeds.shape[0], expected_dim - embeds.shape[1], 
                                         device=embeds.device, dtype=embeds.dtype)
                    embeds = torch.cat([embeds, padding], dim=1)
                else:
                    embeds = embeds[:, :expected_dim]
            processed_embeds.append(embeds.to(device=self.device, dtype=self.dtype))
        
        # Use dynamic generation function
        try:
            image = self.pipe.generate_ip_adapter_embeds_dynamic(
                prompt=text_prompt,
                ip_adapter_embeds=processed_embeds,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=0.0,
                generator=generator,
                scale_schedule_fn=scale_schedule_fn,
            ).images[0]
        except Exception as e:
            print(f"Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            image = Image.new('RGB', (512, 512), color='black')
        
        return image

    def batch_generate(self, image_embeds_batch_list, text_prompt='', generator=None, scale_schedule_fn=None):
        """Batch generate images, each sample corresponds to multiple embeddings
        
        Args:
            image_embeds_batch_list: List[List[Tensor]], outer list length = number of samples, inner list length = num_adapters
        """
        if not image_embeds_batch_list:
            return []
        
        images = []
        for i, embeds_list in enumerate(image_embeds_batch_list):
            print(f"Generating image {i+1}/{len(image_embeds_batch_list)}...")
            try:
                image = self.generate(embeds_list, text_prompt, generator, scale_schedule_fn)
                images.append(image)
            except Exception as e:
                print(f"Failed to generate image {i+1}: {e}")
                images.append(None)
        return images


class LocalModelManager:
    """Local model manager for checking and managing models"""
    
    def __init__(self, base_path=""):
        self.base_path = Path(base_path)
        self.model_paths = MODEL_PATHS
    
    def check_models(self):
        """Check if all models are available"""
        print("\n" + "="*50)
        print("Model integrity check")
        print("="*50)
        
        results = {
            "sdxl_turbo": self._check_sdxl_turbo(),
            "ip_adapter": self._check_ip_adapter(),
        }
        
        all_ok = all(results.values())
        
        print("\n" + "="*50)
        if all_ok:
            print("All models passed the check!")
        else:
            print("Some models have issues, please check the warnings above.")
        print("="*50)
        
        return all_ok
    
    def _check_sdxl_turbo(self):
        """Check SDXL-Turbo model"""
        print("\nChecking SDXL-Turbo model:")
        
        if not self.model_paths.sdxl_turbo_path.exists():
            print("  Model directory does not exist.")
            return False
        
        # Check key files
        required_files = ["model_index.json", "scheduler/scheduler_config.json"]
        
        for file in required_files:
            file_path = Path(self.model_paths.get_sdxl_path()) / file
            if not file_path.exists():
                print(f"  ✗ Missing file: {file}")
                return False
        
        print("  ✓ SDXL-Turbo model is complete.")
        return True
    
    def _check_ip_adapter(self):
        """Check IP-Adapter model"""
        print("\nChecking IP-Adapter model:")
        
        if not self.model_paths.ip_adapter_path.exists():
            print("  ✗ Model directory does not exist.")
            return False
        
        # Check weight files
        weights_path = Path(self.model_paths.get_ip_adapter_sdxl_weights())
        if not weights_path.exists():
            # Try other possible weight files
            possible_weights = ["ip-adapter_sdxl.bin", "ip-adapter_sdxl_vit-h.bin", "ip-adapter-plus_sdxl_vit-h.bin"]
            found = False
            for weight in possible_weights:
                if (self.model_paths.ip_adapter_path / weight).exists():
                    print(f"   Found weight file: {weight}")
                    found = True
                    break
            
            if not found:
                print("   No weight file found.")
                return False
        else:
            print(f"  IP-Adapter weight file: {weights_path.name}")
        
        return True


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    model_manager = LocalModelManager()
    model_manager.check_models()
    
    print("\nTesting local model generation (multi IP-Adapter)...")
    try:
        generator = Generator4Embeds(num_inference_steps=4, device='cuda', num_adapters=4)
        
        # Create 4 random features (simulating RGB, LMS, DKL, CIELUV)
        test_feature_rgb = torch.randn(1, 1024)
        test_feature_lms = torch.randn(1, 1024)
        test_feature_dkl = torch.randn(1, 1024)
        test_feature_cieluv = torch.randn(1, 1024)
        

        def example_schedule(t, t_max=1000):
            progress = 1.0 - t / t_max  # correct t direction

            if progress < 0.2:  # early
                return [1.0, 0.0, 0.0, 0.0]

            elif progress < 0.6:  # middle
                w = (progress - 0.2) / 0.4
                return [1.0 - 0.6 * w, 0.4 * w, 0.4 * w, 0.0]

            else:  # late
                w = (progress - 0.6) / 0.4
                return [0.4, 0.0, 0.0, 0.5 * w]

        
        print("Generating image...")
        generated_image = generator.generate(
            [test_feature_rgb, test_feature_lms, test_feature_dkl, test_feature_cieluv],
            text_prompt="a beautiful photo",
            scale_schedule_fn=example_schedule
        )
        
        if hasattr(generated_image, 'save'):
            generated_image.save("multi_adapter_test_output.png")
            print(f"✓ Image generation completed, saved as: multi_adapter_test_output.png")
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
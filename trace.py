from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Union
import math

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F

from utils import cache_dir, auto_autocast
from experiment import GenerationExperiment
from heatmap import RawHeatMapCollection, GlobalHeatMap
from hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator, UNetCrossAttentionLocatorXL


__all__ = ['trace', 'DiffusionHeatMapHooker', 'GlobalHeatMap']


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
            self,
            pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
            low_memory: bool = False,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: str = None
    ):
        self.all_heat_maps = RawHeatMapCollection()
        self.pipeline_type = type(pipeline)
        self.is_sdxl = isinstance(pipeline, StableDiffusionXLPipeline)
        
        # Calculate latent dimensions based on pipeline type
        h = (pipeline.unet.config.sample_size * pipeline.vae_scale_factor)
        if self.is_sdxl:
            # SDXL uses 128x128 latent space for 1024x1024 images
            self.latent_hw = 16384  # 128x128
        else:
            # SD v1.4 uses 64x64 or 96x96 latent space
            self.latent_hw = 4096 if h == 512 or h == 1024 else 9216  # 64x64 or 96x96
        
        locate_middle = load_heads or save_heads
        
        # Use appropriate locator based on pipeline type
        if self.is_sdxl:
            self.locator = UNetCrossAttentionLocatorXL(
                restrict={0} if low_memory else None, 
                locate_middle_block=locate_middle,
                locate_transformer_blocks=True
            )
        else:
            self.locator = UNetCrossAttentionLocator(
                restrict={0} if low_memory else None, 
                locate_middle_block=locate_middle
            )
        
        self.last_prompt: str = ''
        self.last_image: Image = None
        self.time_idx = 0
        self._gen_idx = 0

        modules = [
            UNetCrossAttentionHooker(
                x,
                self,
                layer_idx=idx,
                latent_hw=self.latent_hw,
                load_heads=load_heads,
                save_heads=save_heads,
                data_dir=data_dir,
                is_sdxl=self.is_sdxl
            ) for idx, x in enumerate(self.locator.locate(pipeline.unet))
        ]

        modules.append(PipelineHooker(pipeline, self))

        # SDXL always has image processor
        if self.is_sdxl:
            modules.append(ImageProcessorHooker(pipeline.image_processor, self))

        super().__init__(modules)
        self.pipe = pipeline

    def time_callback(self, *args, **kwargs):
        self.time_idx += 1

    @property
    def layer_names(self):
        return self.locator.layer_names

    def to_experiment(self, path, seed=None, id='.', subtype='.', **compute_kwargs):
        # type: (Union[Path, str], int, str, str, Dict[str, Any]) -> GenerationExperiment
        """Exports the last generation call to a serializable generation experiment."""

        # Use appropriate tokenizer for SDXL (use first tokenizer for compatibility)
        tokenizer = self.pipe.tokenizer if hasattr(self.pipe, 'tokenizer') else self.pipe.tokenizer

        return GenerationExperiment(
            self.last_image,
            self.compute_global_heat_map(**compute_kwargs).heat_maps,
            self.last_prompt,
            seed=seed,
            id=id,
            subtype=subtype,
            path=path,
            tokenizer=tokenizer,
        )

    def compute_global_heat_map(self, prompt=None, factors=None, head_idx=None, layer_idx=None, normalize=False):
        # type: (str, List[float], int, int, bool) -> GlobalHeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.all_heat_maps

        if prompt is None:
            prompt = self.last_prompt

        if factors is None:
            if self.is_sdxl:
                # SDXL has different spatial factors due to higher resolution
                factors = {0, 1, 2, 4, 8, 16, 32, 64, 128}
            else:
                factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)

        all_merges = []
        x = int(np.sqrt(self.latent_hw))

        with auto_autocast(dtype=torch.float32):
            for (factor, layer, head), heat_map in heat_maps:
                if factor in factors and (head_idx is None or head_idx == head) and (layer_idx is None or layer_idx == layer):
                    heat_map = heat_map.unsqueeze(1)
                    # The clamping fixes undershoot.
                    all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

            try:
                maps = torch.stack(all_merges, dim=0)
            except RuntimeError:
                if head_idx is not None or layer_idx is not None:
                    raise RuntimeError('No heat maps found for the given parameters.')
                else:
                    raise RuntimeError('No heat maps found. Did you forget to call `with trace(...)` during generation?')

            maps = maps.mean(0)[:, 0]
            
            # Handle tokenization for SDXL (use first tokenizer)
            tokenizer = self.pipe.tokenizer if hasattr(self.pipe, 'tokenizer') else self.pipe.tokenizer
            tokenized_length = len(tokenizer.tokenize(prompt)) + 2  # 1 for SOS and 1 for padding
            maps = maps[:tokenized_length]

            if normalize:
                maps = maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities

        return GlobalHeatMap(tokenizer, prompt, maps)


class ImageProcessorHooker(ObjectHooker[VaeImageProcessor]):
    def __init__(self, processor: VaeImageProcessor, parent_trace: 'trace'):
        super().__init__(processor)
        self.parent_trace = parent_trace

    def _hooked_postprocess(hk_self, _: VaeImageProcessor, *args, **kwargs):
        images = hk_self.monkey_super('postprocess', *args, **kwargs)
        hk_self.parent_trace.last_image = images[0]

        return images

    def _hook_impl(self):
        self.monkey_patch('postprocess', self._hooked_postprocess)


class PipelineHooker(ObjectHooker[Union[StableDiffusionPipeline, StableDiffusionXLPipeline]]):
    def __init__(self, pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline], parent_trace: 'trace'):
        super().__init__(pipeline)
        self.heat_maps = parent_trace.all_heat_maps
        self.parent_trace = parent_trace
        self.is_sdxl = isinstance(pipeline, StableDiffusionXLPipeline)

    def _hooked_run_safety_checker(hk_self, self: StableDiffusionPipeline, image, *args, **kwargs):
        image, has_nsfw = hk_self.monkey_super('run_safety_checker', image, *args, **kwargs)

        if self.image_processor:
            if torch.is_tensor(image):
                images = self.image_processor.postprocess(image, output_type='pil')
            else:
                images = self.image_processor.numpy_to_pil(image)
        else:
            images = self.numpy_to_pil(image)

        hk_self.parent_trace.last_image = images[len(images)-1]

        return image, has_nsfw

    def _hooked_check_inputs(hk_self, _: Union[StableDiffusionPipeline, StableDiffusionXLPipeline], prompt: Union[str, List[str]], *args, **kwargs):
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError('Only single prompt generation is supported for heat map computation.')
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt

        hk_self.heat_maps.clear()
        hk_self.parent_trace.last_prompt = last_prompt

        return hk_self.monkey_super('check_inputs', prompt, *args, **kwargs)

    def _hooked_encode_prompt(hk_self, pipeline, prompt, *args, **kwargs):
        """Hook for SDXL's encode_prompt method to capture the prompt."""
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError('Only single prompt generation is supported for heat map computation.')
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt

        hk_self.heat_maps.clear()
        hk_self.parent_trace.last_prompt = last_prompt

        return hk_self.monkey_super('encode_prompt', prompt, *args, **kwargs)

    def _hook_impl(self):
        # SD v1.4 has run_safety_checker, SDXL does not
        self.monkey_patch('run_safety_checker', self._hooked_run_safety_checker, strict=False)
        
        if self.is_sdxl:
            # SDXL uses encode_prompt instead of check_inputs
            self.monkey_patch('encode_prompt', self._hooked_encode_prompt, strict=False)
        else:
            # SD v1.4 uses check_inputs
            self.monkey_patch('check_inputs', self._hooked_check_inputs, strict=False)


class UNetCrossAttentionHooker(ObjectHooker[Attention]):
    def __init__(
            self,
            module: Attention,
            parent_trace: 'trace',
            context_size: int = 77,
            layer_idx: int = 0,
            latent_hw: int = 9216,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: Union[str, Path] = None,
            is_sdxl: bool = False,
    ):
        super().__init__(module)
        self.heat_maps = parent_trace.all_heat_maps
        self.is_sdxl = is_sdxl
        
        # SDXL has different context sizes due to dual text encoders
        if is_sdxl:
            # SDXL context size can vary, typically 77 for each encoder, concatenated = 154
            # But can be different based on the specific implementation
            self.context_size = 154 if context_size == 77 else context_size
        else:
            self.context_size = context_size
            
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

        self.load_heads = load_heads
        self.save_heads = save_heads
        self.trace = parent_trace

        if data_dir is not None:
            data_dir = Path(data_dir)
        else:
            data_dir = cache_dir() / 'heads'

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.

        Returns:
            `torch.Tensor`: the heat maps across heads with shape (heads, tokens, height, width).
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                
                # Handle different guidance configurations
                if self.is_sdxl:
                    # SDXL guidance handling - may have different batch structures
                    if map_.size(0) > 2:
                        # Filter out unconditional part (keep guided part)
                        map_ = map_[map_.size(0) // 2:]
                    # For SDXL, we might not need additional filtering in some cases
                else:
                    # Original SD v1.4 logic
                    # For Instruct Pix2Pix, divide the map into three parts: text condition, image condition and unconditional,
                    # and only keep the text condition part, which is first of the three parts(as per diffusers implementation).
                    if map_.size(0) == 24:
                        map_ = map_[:((map_.size(0) // 3)+1)]  # Filter out unconditional and image condition
                    else:
                        map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                        
                maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        return maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)

    def _save_attn(self, attn_slice: torch.Tensor):
        torch.save(attn_slice, self.data_dir / f'{self.trace._gen_idx}.pt')

    def _load_attn(self) -> torch.Tensor:
        return torch.load(self.data_dir / f'{self.trace._gen_idx}.pt')

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs,  # SDXL may pass additional arguments
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # DAAM save heads
        if self.save_heads:
            self._save_attn(attention_probs)
        elif self.load_heads:
            attention_probs = self._load_attn()

        # compute shape factor
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1]))
        self.trace._gen_idx += 1

        # Check if this is cross-attention (not self-attention)
        # SDXL may have different sequence lengths due to dual text encoders
        expected_context_sizes = [self.context_size]
        if self.is_sdxl:
            # Add other possible context sizes for SDXL
            expected_context_sizes.extend([77, 154, 231])  # Common SDXL context sizes
        
        # skip if too large or if it's not cross-attention
        if (attention_probs.shape[-1] in expected_context_sizes and 
            factor != 8 and  # Skip 8x8 resolution typically
            factor > 0):     # Ensure valid factor
            
            # shape: (batch_size, spatial_res, spatial_res, context_size)
            maps = self._unravel_attn(attention_probs)

            for head_idx, heatmap in enumerate(maps):
                self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _hook_impl(self):
        self.original_processor = self.module.processor
        self.module.set_processor(self)

    def _unhook_impl(self):
        self.module.set_processor(self.original_processor)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
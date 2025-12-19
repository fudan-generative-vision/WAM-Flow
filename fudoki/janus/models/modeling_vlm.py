# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from fudoki.janus.models.clip_encoder import CLIPVisionTower
from fudoki.janus.models.projector import MlpProjector

from icecream import ic

class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x

class number_head(torch.nn.Module):
    def __init__(self, d_model=2048, dim_feedforward=5632, numhead_bias=True):
        super().__init__()
        self.num_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_feedforward, bias=numhead_bias),
            torch.nn.GELU(),
            torch.nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )

    def forward(self, x):
        x = self.num_head(x)
        return x

def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from fudoki.janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)
        # Convert causal attention to full attention
        self._convert_to_full_attention()

    def add_number_head(self):
        self.language_model.num_head = number_head(
            d_model=self.language_model.config.hidden_size,
            dim_feedforward=self.language_model.config.intermediate_size,
            numhead_bias=True,
        )

    def _convert_to_full_attention(self):
        """Convert all causal attention layers to full attention using BlockDiagonalMask"""
        import types
        import xformers
        import xformers.ops.fmha as fmha
        from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
        from transformers.cache_utils import Cache
        import torch.nn.functional as F
        from typing import Optional, Tuple
        
        for layer_idx, layer in enumerate(self.language_model.model.layers):
            if hasattr(layer, 'self_attn'):
                def full_attention_forward(
                    self,
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value: Optional[Cache] = None,
                    output_attentions: bool = False,
                    use_cache: bool = False,
                    cache_position: Optional[torch.LongTensor] = None,
                    **kwargs,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                    bsz, q_len, _ = hidden_states.size()

                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)

                    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                    cos, sin = self.rotary_emb(value_states, position_ids)
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)

                    query_states = query_states.contiguous().transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads, self.head_dim).contiguous()
                    key_states = key_states.contiguous().transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).contiguous()
                    value_states = value_states.contiguous().transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).contiguous()
                    # attn_output = xformers.ops.memory_efficient_attention(query_states, key_states, value_states, p=self.attention_dropout, attn_bias=None)

                    attn_bias, flattened_q = fmha.BlockDiagonalMask.from_tensor_list([query_states[bs, :attention_mask[bs].sum()][None] for bs in range(bsz)])
                    _, flattened_k = fmha.BlockDiagonalMask.from_tensor_list([key_states[bs, :attention_mask[bs].sum()][None] for bs in range(bsz)])
                    _, flattened_v = fmha.BlockDiagonalMask.from_tensor_list([value_states[bs, :attention_mask[bs].sum()][None] for bs in range(bsz)])


                    output = xformers.ops.memory_efficient_attention(flattened_q, flattened_k, flattened_v, p=self.attention_dropout, attn_bias=attn_bias)
                    output = attn_bias.split(output)
                    attn_output = hidden_states.clone().reshape(bsz, q_len, self.num_heads, self.head_dim)
                    for bs in range(bsz):
                        attn_output[bs, :attention_mask[bs].sum()] = output[bs]

                    attn_output = attn_output.reshape(bsz, q_len, -1)
                    attn_output = self.o_proj(attn_output)

                    if not output_attentions:
                        attn_weights = None
                    else:
                        query, key = flattened_q, flattened_k
                        scale = 1.0 / query.shape[-1] ** 0.5
                        query = query * scale
                        query = query.transpose(1, 2)
                        key = key.transpose(1, 2)
                        attn = query @ key.transpose(-2, -1)
                        attn = attn.softmax(-1)
                        attn_weights = attn

                    return attn_output, attn_weights, past_key_value
                
                # Bind the modified forward method
                layer.self_attn.forward = types.MethodType(full_attention_forward, layer.self_attn)
                
        if hasattr(self.language_model.model, '_update_causal_mask'):
            def _update_causal_mask(
                self,
                attention_mask: torch.Tensor,
                input_tensor: torch.Tensor,
                cache_position: torch.Tensor,
                past_key_values: Optional[Cache] = None,
                output_attentions: bool = False,
            ):
                # Simply return the original attention mask without any causal modifications
                return attention_mask
            
            # Override the method in the language model
            self.language_model.model._update_causal_mask = types.MethodType(
                _update_causal_mask, self.language_model.model
            )
        
    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))

    def get_fsdp_wrap_module_list(self):
        return list(self.language_model.model.layers)

    def token_drop(self, mmsamples, datainfo, uncond_prob=0.1, uncond_id=100015):
        batch_size = mmsamples.shape[0]
        drop_ids = torch.rand(batch_size, device=mmsamples.device) < uncond_prob
        uncondition_context = mmsamples.clone()
        generation_mask = (datainfo['generation_or_understanding_mask'] == 1)
        for b in range(batch_size):
            if drop_ids[b]:
                if generation_mask[b]:
                    nz = datainfo['text_token_mask'][b].nonzero()
                    if nz.numel() > 0:
                        text_nonzero_idx_begin = nz[0, 0] 
                        text_nonzero_idx_end = nz[-1, 0]
                        uncondition_context[b, text_nonzero_idx_begin:text_nonzero_idx_end+1] = uncond_id
        return uncondition_context

    def forward(self, mmsamples, datainfo):
        if self.training:
            mmsamples = self.token_drop(mmsamples, datainfo, uncond_prob=0.1, uncond_id=100015)
        for b_index in range(mmsamples.shape[0]):
            mask = datainfo['image_token_mask'][b_index] == 1 
            indices = torch.nonzero(mask, as_tuple=False) 
        
            if datainfo['generation_or_understanding_mask'][b_index] == 1:
                imgsamples = mmsamples[b_index, indices[:, 0]]
                img_embeds = self.prepare_gen_img_embeds(imgsamples.unsqueeze(0))
                inputs_embeds = self.language_model.get_input_embeddings()(mmsamples)
                inputs_embeds[b_index, indices[:, 0]] = img_embeds.reshape(-1, img_embeds.shape[-1])
            elif datainfo['generation_or_understanding_mask'][b_index] == 0:
                imgsamples = datainfo['understanding_img']
                if datainfo['has_understanding_img'][b_index] == 1:
                    imgs = imgsamples[b_index]
                    if imgs.dim() == 3:
                        imgs = imgs.unsqueeze(0)
                    img_embeds = self.aligner(self.vision_model(imgs))
                    inputs_embeds = self.language_model.get_input_embeddings()(mmsamples)
                    inputs_embeds[b_index, indices[:, 0]] = img_embeds.reshape(-1, img_embeds.shape[-1])
                else:
                    inputs_embeds = self.language_model.get_input_embeddings()(mmsamples)
    
        outputs = self.language_model.model(inputs_embeds=inputs_embeds, use_cache=False, attention_mask=datainfo['attention_mask'])
        hidden_states = outputs.last_hidden_state
        
        img_logits = self.gen_head(hidden_states)
        txt_logits = self.language_model.lm_head(hidden_states)

        img_logits = torch.cat([torch.zeros((img_logits.shape[0], 1, img_logits.shape[2]), device=img_logits.device), img_logits[:, :-1, :]], dim=1)
        txt_logits = torch.cat([torch.zeros((txt_logits.shape[0], 1, txt_logits.shape[2]), device=txt_logits.device), txt_logits[:, :-1, :]], dim=1)
        
        return img_logits, txt_logits


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

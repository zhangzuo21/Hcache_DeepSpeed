# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Optional, Tuple

import torch
from torch.cuda import Stream, Event

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from ...allocator import empty_from
from ...inference_utils import ActivationType, DtypeEnum
from .. import *
from ...modules.configs import *
from ...modules.interfaces import *
from ...ragged import RaggedBatchWrapper

from .container import Llama2NonTransformerContainer, Llama2TransformerContainer


class Llama2InferenceModel(DSTransformerModelBase):
    """
    Inference model implementation for ragged batching for Llama-2 models.
    """

    _non_transformer: Optional[Llama2NonTransformerContainer]
    """
    Embed + unembed container. Specializing the type annotation.
    """

    _transformer: Optional[Iterable[Llama2TransformerContainer]]
    """
    Per-layer transformer container. Specializing the type annotation.
    """
    """
    Properties ineherited from `DSInferenceModelBase`
    """

    @property
    def max_sequence_length(self) -> int:
        return self._config.max_seq_length

    """
    Properties ineherited from `DSTransformerModelBase`
    """

    @property
    def num_layers(self) -> int:
        return self._config.num_hidden_layers

    @property
    def model_dim(self) -> int:
        return self._config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size

    @property
    def head_size(self) -> int:
        return self.model_dim // self.n_heads

    @property
    def n_heads(self) -> int:
        return self._config.num_attention_heads

    @property
    def intermediate_dim(self) -> int:
        return self._config.intermediate_size

    @property
    def n_heads_kv(self) -> int:
        return self._config.num_key_value_heads

    @property
    def activation_dtype(self) -> DtypeEnum:
        if self._config.torch_dtype == torch.float16:
            return DtypeEnum.fp16
        elif self._config.torch_dtype == torch.bfloat16:
            return DtypeEnum.bf16
        else:
            raise NotImplementedError("Only fp16 and bf16 are supported")

    @property
    def mlp_activation_fn(self) -> ActivationType:
        activation = self._config.hidden_act.lower()
        # llama model family is special and is always gated so force gated versions of relu, gelu, silu
        if activation == "gelu":
            return ActivationType.GEGLU
        elif activation == "relu":
            return ActivationType.ReGLU
        elif activation == "gegelu":
            return ActivationType.GEGLU
        elif activation == "silu":
            return ActivationType.SiGLU
        else:
            raise NotImplementedError(f"Activation {activation} not supported")

    @property
    def norm_type(self) -> NormTypeEnum:
        return NormTypeEnum.RMSNorm

    @property
    def positional_embedding_type(self) -> PositionalEmbeddingType:
        return PositionalEmbeddingType.rotate_half

    @property
    def positional_embedding_config(self) -> Optional[RotateHalfConfig]:
        return RotateHalfConfig(theta_base=self._config.rope_theta)

    """
    Forward implementations
    """

    def _forward_embed(self, ragged_batch: RaggedBatchWrapper) -> torch.Tensor:
        """
        Performs the embedding lookup prior to running the transformer of the model.

        Arguments:
            ragged_batch (RaggedBatchWrapper): The batch to embed.

        Returns:
            torch.Tensor: The embedded batch.
        """
        embed = self.embed(ragged_batch, self._non_transformer.word_emb)

        if embed.shape[-1] != self.model_dim:
            raise ValueError(f"Embedding output shape {embed.shape} does not match model_dim {self.model_dim}")

        return embed

    def _forward_transformer_layer(self, layer_idx: int, residual: torch.Tensor, hidden_states: torch.Tensor,
                                   ragged_batch_info: RaggedBatchWrapper) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one (slightly offset) layer of the transformer. This implementation does a peak-ahead
        optimization to fuse the layer norm of the next layer into the current layer.

        Arguments:
            layer_idx (int): The index of the layer to execute.
            residual (torch.Tensor): The residual tensor from the previous layer.
            hidden_states (torch.Tensor): The hidden states from the previous layer. This is the
                hidden states after pre normalization.
            ragged_batch_info (RaggedBatchWrapper): The batch metadata.
        """
        # TODO(cmikeh2): Distribute ragged_batch_info to all modules
        # pinned_hidden_state = torch.empty_like(hidden_states, device='cpu', pin_memory=True)
        # pinned_hidden_state.copy_(hidden_states, non_blocking=True)

        cur_params = self._transformer[layer_idx]
        kv_cache = self.state_manager.get_cache(layer_idx)

        hidden_states = self.qkv(hidden_states, cur_params.qkv_w, b=None)
        hidden_states = self.attn(hidden_states, kv_cache, ragged_batch_info)
        hidden_states = self.attn_out(hidden_states, cur_params.attn_out_w, b=None)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        residual, hidden_states = self.norm(residual, hidden_states, cur_params.mlp_norm_gamma, beta=None)

        # Should be configurable in the future
        hidden_states = self.mlp_1(hidden_states, cur_params.mlp_1_w, b=None)
        hidden_states = self.mlp_2(hidden_states, cur_params.mlp_2_w, b=None)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        if layer_idx != self.num_layers - 1:
            next_params = self._transformer[layer_idx + 1]
            residual, hidden_states = self.norm(residual, hidden_states, next_params.attn_norm_gamma, beta=None)
        else:
            # On last layer, we just need to perform the residual add. Adding into the residual
            # here is safe.
            residual.add_(hidden_states)

        return residual, hidden_states

    def _forward_unembed(self, hidden_states: torch.Tensor, ragged_batch_info: RaggedBatchWrapper) -> torch.Tensor:
        """
        Performs unembedding of the hidden states to logits. This will only sample the final
        token of each sequence.
        """
        logits = self.unembed(hidden_states,
                              self._non_transformer.word_unembed,
                              ragged_batch_info,
                              gamma=self._non_transformer.final_norm)

        if self.tp_size > 1:
            comm_buffer = empty_from(self._comm_logits, (self.tp_size, logits.shape[0], logits.shape[1]))
            full_logits = empty_from(self._return_logits, (logits.shape[0], self.vocab_size))

            dist.all_gather_into_tensor(comm_buffer, logits, group=self._base_mp_group)

            full_logits.copy_(comm_buffer.permute(1, 0, 2).reshape(logits.shape[0], self.vocab_size))

            return full_logits
        else:
            return logits

    def forward(self, wrapped_batch: RaggedBatchWrapper) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = self._forward_embed(wrapped_batch)
        # print(f"input ids: {wrapped_batch._input_ids_shadow}")

        residual, hidden_states = self.norm(residual, None, self._transformer[0].attn_norm_gamma, beta=None)
        # print("=============")
        # print(hidden_states)
        latent_buffer = torch.empty((self.num_layers, hidden_states.shape[0], hidden_states.shape[1]), device=get_accelerator().current_device(), dtype=hidden_states.dtype)

        for layer_idx in range(self.num_layers):
            latent_buffer[layer_idx].copy_(hidden_states)
            residual, hidden_states = self._forward_transformer_layer(layer_idx, residual, hidden_states,
                                                                      wrapped_batch)

        latent_cpu = latent_buffer.cpu()
        del latent_buffer
        return self._forward_unembed(residual, wrapped_batch), latent_cpu

    def restore_kv(self, wrapped_batch: RaggedBatchWrapper, latents: torch.Tensor):
        io_stream = Stream()
        compute_stream = torch.cuda.default_stream()
        io_done_events = [Event() for _ in range(len(latents))]
        temp_store_done_events = [Event() for _ in range(len(latents))]
        layer_buffer = torch.empty_like(latents[0], device=get_accelerator().current_device(), dtype=latents.dtype)
        for layer_idx, latent in enumerate(latents):
            with torch.cuda.stream(io_stream):
                if layer_idx > 0:
                    io_stream.wait_event(temp_store_done_events[layer_idx - 1])
                latent = latent.to(get_accelerator().current_device())
                io_done_events[layer_idx].record(io_stream)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(io_done_events[layer_idx])
                layer_buffer.copy_(latent)
                temp_store_done_events[layer_idx].record(compute_stream)
                cur_params = self._transformer[layer_idx]
                qkv = self.qkv(layer_buffer, cur_params.qkv_w, b=None)
                kv_cache = self.state_manager.get_cache(layer_idx)
                self.attn.restore_kv(qkv, kv_cache, wrapped_batch)

        io_stream.synchronize()
        compute_stream.synchronize()

    # def restore_kv(self, wrapped_batch: RaggedBatchWrapper, latents: torch.Tensor):
    #     for layer_idx, latent in enumerate(latents):
    #         latent = latent.to(get_accelerator().current_device())
    #         cur_params = self._transformer[layer_idx]
    #         qkv = self.qkv(latent, cur_params.qkv_w, b=None)
    #         kv_cache = self.state_manager.get_cache(layer_idx)
    #         self.attn.restore_kv(qkv, kv_cache, wrapped_batch)
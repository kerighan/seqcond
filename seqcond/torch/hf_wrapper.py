import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Tuple, Union, Any

from seqcond.config import ModelConfig
from seqcond.torch.model import SeqCondModel


class SeqCondHFConfig(PretrainedConfig):
    model_type = "seqcond"

    def __init__(
        self,
        d_model: int = 768,
        d_ff: int = 2304,
        num_layers: int = 12,
        vocab_size: int = 100300,
        maxlen: int = 768,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        seqcond_heads: int = 32,
        num_query_heads: int = 6,
        num_thetas: int = 4,
        num_anchor_heads: int = 0,
        conv_kernel_size: int = 4,
        expand_factor: float = 2.0,
        out_expand_factor: int = 3,
        seqcond_ratio: int = 5,
        tie_weights: bool = True,
        **kwargs,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.seqcond_heads = seqcond_heads
        self.num_query_heads = num_query_heads
        self.num_thetas = num_thetas
        self.num_anchor_heads = num_anchor_heads
        self.conv_kernel_size = conv_kernel_size
        self.expand_factor = expand_factor
        self.out_expand_factor = out_expand_factor
        self.seqcond_ratio = seqcond_ratio
        self.tie_weights = tie_weights

        # Aliases for Transformers compatibility
        self.num_hidden_layers = num_layers
        self.hidden_size = d_model

        super().__init__(**kwargs)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            model_type="seqcond",
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            maxlen=self.maxlen,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            qk_norm=self.qk_norm,
            qk_norm_eps=self.qk_norm_eps,
            seqcond_heads=self.seqcond_heads,
            num_query_heads=self.num_query_heads,
            num_thetas=self.num_thetas,
            num_anchor_heads=self.num_anchor_heads,
            conv_kernel_size=self.conv_kernel_size,
            expand_factor=self.expand_factor,
            out_expand_factor=self.out_expand_factor,
            seqcond_ratio=self.seqcond_ratio,
            tie_weights=self.tie_weights,
        )


class SeqCondForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SeqCondHFConfig
    _safetensors_name = "model.safetensors"
    _supports_cache_class = False  # Disable DynamicCache, use our custom states

    def __init__(self, config: SeqCondHFConfig):
        super().__init__(config)
        self.model_config = config.to_model_config()
        self.model = SeqCondModel(self.model_config)

        # Initialize weights / load checkpoint manually usually
        # But for HF usage, we might load_state_dict later

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.output_projection

    def set_output_embeddings(self, new_embeddings):
        self.model.output_projection = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # We assume input_ids is provided.
        # This implementation uses the sequential .step() method because
        # we don't have a parallel forward implementation yet.
        # This is slow but correct for evaluation.

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize states if not provided (past_key_values)
        # past_key_values can be None, our custom states list, or a DynamicCache
        # We need to handle DynamicCache by ignoring it and using our own states
        if past_key_values is None or not isinstance(past_key_values, list):
            # Prefill mode - Use parallel forward
            # states will be initialized inside model.forward
            logits, states = self.model.forward(input_ids)
            # logits is (B, L, V)
        else:
            # Generation mode - Sequential
            states = past_key_values

            logits_list = []

            # Iterate over sequence
            # Note: If past_key_values is provided, input_ids should be the new tokens only
            for t in range(seq_len):
                token_id = input_ids[:, t : t + 1]  # (B, 1)
                logits_step, states = self.model.step(token_id, states)
                logits_list.append(logits_step.unsqueeze(1))

            logits = torch.cat(logits_list, dim=1)  # (B, L, V)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits, states) + ((loss,) if loss is not None else ())
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=states,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Only use past_key_values if it's our custom list of states
        # DynamicCache or other cache types should be ignored
        if isinstance(past_key_values, list) and len(past_key_values) > 0:
            # We have valid cached states, only process the last token
            input_ids = input_ids[:, -1:]
        else:
            # No valid cache, reset past_key_values to None
            past_key_values = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }

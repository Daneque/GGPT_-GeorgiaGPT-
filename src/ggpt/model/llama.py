import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock
from .outputs import CausalLMOutput
from .rope import RotaryEmbedding


class LlamaLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.model_type != "llama":
            raise ValueError("LlamaLM requires model_type='llama'")
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embeddings.weight
        residual_std = config.initializer_range / math.sqrt(2 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, std=residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, std=residual_std)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long [B, T]")
        batch, length = input_ids.shape
        if batch == 0 or not 1 <= length <= self.config.max_seq_len:
            raise ValueError("require B > 0 and 1 <= T <= max_seq_len")
        if targets is not None:
            if targets.shape != input_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must be torch.long with the input shape")
            if targets.device != input_ids.device:
                raise ValueError("targets and inputs must be on the same device")
        cos, sin = self.rope(length, input_ids.device)
        x = self.token_embeddings(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            # No shift here. A fully ignored batch has differentiable zero loss.
            total = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                    targets.reshape(-1), ignore_index=-100, reduction="sum")
            loss = total / (targets != -100).sum().clamp_min(1)
        return CausalLMOutput(logits=logits, loss=loss)

from dataclasses import dataclass, field
from typing import Type
from torch import nn

@dataclass
class MoEArgs():
    """
    MoE input arguments class.
    """
    num_experts : int = field(default=4)
    num_experts_per_token : int = field(default=2)

@dataclass
class GPTConfig:
    """
    Base class for GPT models.
    """
    # required
    vocab_size: int

    # optional
    block_size: int = 32
    batch_size: int = 8
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    bias: bool = True
    ff_class: Type[nn.Module] | None = None
    moe: MoEArgs | None = None
import torch
from torch import nn
import torch.nn.functional as F

from src.config import GPTConfig, MoEArgs

class DenseFFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        hidden_dim = 4 * config.n_embd

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Expert(nn.Module):
    """
    An expert MLP instance from within a MoE.
    """

    def __init__(self, config:GPTConfig) -> None:
        """
        Initiates expert MLP given dimensions/hidden dimensions.
        """
        super().__init__()

        hidden_dim = 4 * config.n_embd // config.moe.num_experts_per_token

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)


class Gate(nn.Module):
    """
    MoE gating network MLP.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.proj = nn.Linear(config.n_embd, config.moe.num_experts)

    def forward(self, x):
        return self.proj(x)


class MoELayer(nn.Module):
    """
    Mixture of experts FeedForward Layer
    """

    def __init__(self, experts: list[nn.Module], gate: nn.Module, moe_args: MoEArgs):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # top-k probs
        topk_probs, topk_indices = torch.topk(
            torch.softmax(self.gate(x), dim=-1),    # gate probs
            k=self.args.num_experts_per_token,
            dim=-1
        )  # [batch_size, n_active], [batch_size, n_active] 

        # simple sequential looping over experts, route each batch
        output = torch.zeros_like(x)  # [batch_size, n_embd]

        for expert_id, expert in enumerate(self.experts):
            mask = (topk_indices == expert_id)

            if not mask.any():
                continue

            # row (batch) and col (expert) indices where this expert was selected
            batch_indices, topk_slot = mask.nonzero(as_tuple=True) # [non-0], [non-0]

            # get expert outputs, weigh by probs
            output[batch_indices] += (
                expert(x[batch_indices])   # route correct inputs [non-0, n_embd] -> get expert outputs [non-0, n_embd]
                * 
                topk_probs[batch_indices, topk_slot].unsqueeze(-1) # select probs [non-0] -> restore 2° dim [non-0, 1]
            )

        return output

class MoEFFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.moe = MoELayer(  
            experts=[Expert(config) for _ in range(config.moe.num_experts)],
            gate=Gate(config),
            moe_args=config.moe
        )

    def forward(self, x):
        # flatten seq_len
        B,S,N = x.shape
        x = x.reshape(B*S,N)
        
        outputs = self.moe(x)

        # restore previous shape
        return outputs.reshape(B,S,N)
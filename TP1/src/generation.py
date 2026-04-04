import torch
from torch import nn
import torch.nn.functional as F
from collections.abc import Callable
from pydantic import BaseModel, conint, confloat

from src.config import GPTConfig


class SamplingParams(BaseModel):
    greedy: bool = False
    top_k: conint(gt=0) | None = None
    top_p: confloat(gt=0.0, le=1.0) | None = None
    temperature: confloat(gt=0.0) = 1.0


def sample_logits(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:

    # apply temperature
    if params.temperature != 1.0:
        logits = logits / params.temperature

    # greedy -> return argmax
    if params.greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = F.softmax(logits, dim=-1)

    # top_k
    if params.top_k:
        top_k = min(params.top_k, probs.size(-1))
        values, _ = torch.topk(probs, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        probs[probs < min_values] = 0.0

    # top_p
    if params.top_p:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mask but keep at least 1
        sorted_mask = cumulative_probs > params.top_p
        sorted_mask[:, 0] = False
        sorted_probs[sorted_mask] = 0.0

        # zero everything inplace -> scatter non-0s from sorted_probs
        probs.zero_().scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    
    # post-filtering renorm
    probs /= probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # sample
    return torch.multinomial(probs, num_samples=1)


        
@torch.no_grad()
def generate(
    prompt: str,
    model: nn.Module, device: str, config: GPTConfig,
    encode: Callable[[str], list[int]], decode: Callable[[list[int]], str],
    params: SamplingParams,
    max_new_tokens: int = 100, use_cache: bool = True):

    model.eval()
    idx = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)
    kv_cache = None

    for _ in range(max_new_tokens):
        if use_cache and kv_cache is not None:
            idx_cond = idx[:, -1:]
        else:
            idx_cond = idx[:, -config.block_size:]

        out = model(idx_cond, kv_cache=kv_cache) if use_cache else model(idx_cond)

        if isinstance(out, tuple):
            logits, kv_cache = out
        else:
            logits = out
            kv_cache = None

        logits = logits[:, -1, :]
        next_token = sample_logits(logits, params=params)
        idx = torch.cat((idx, next_token), dim=1)

    return decode(idx[0].tolist())
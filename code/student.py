"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
import torch
from model import GPT
from torch import nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()

        hidden = round(8 * width / 3)

        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        gate = F.silu(self.gate(x))
        value = self.value(x)

        return self.down(gate * value)


#两个改进位置
class DropoutGPT(GPT):
    def __init__(self, config):
        super().__init__(config) 

        dropout = config.get('dropout', 0.1)

        for block in self.blocks:
            block.mlp = nn.Sequential(
                block.mlp,
                nn.Dropout(dropout)
            )



class SwiGLUGPT(GPT):
    def __init__(self, config):
        super().__init__(config)

        dropout = config.get("dropout", 0.0)

        for block in self.blocks:
            block.mlp = SwiGLU(config["width"])
            block.mlp.apply(self.initialize)

            if dropout > 0:
                block.mlp = nn.Sequential(block.mlp,nn.Dropout(dropout))
            else:
                block.mlp = block.mlp



# RoPE + GELU + dropout
class RoPEBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads != 0 or (width // heads) % 2 != 0:
            raise ValueError('RoPE requires width divisible by heads and an even head dimension.')
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, width // heads, 2).float() / (width // heads)))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def rotate(self, x):
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            self.rotate(q), self.rotate(k), v, is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class RoPEGPT(GPT):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        dropout = config.get('dropout', 0.1)
        self.token = nn.Embedding(config['vocab'], width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            RoPEBlock(width, config['heads'], dropout) for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    def features(self, ids):
        x = self.dropout(self.token(ids))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)



def build_model(config):
    variant = config.get("variant", "dropout")
    if variant == "dropout":
        return DropoutGPT(config)
    elif variant == "swiglu":
        return SwiGLUGPT(config)
    elif variant == "rope":
        return RoPEGPT(config)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

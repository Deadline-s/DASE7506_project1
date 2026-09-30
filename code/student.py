"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
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



def build_model(config):
    variant = config.get("variant", "dropout")
    if variant == "dropout":
        return DropoutGPT(config)
    elif variant == "swiglu":
        return SwiGLUGPT(config)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

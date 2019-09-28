import torch.nn as nn
import torch
import torch.nn.functional as F

from models.tatc import TATC
from models.res_block import ResStack
from models.selayer import SELayer


class EteWave(nn.Module):
    def __init__(self, n_class, out_ch=128, n_stack=3):
        super().__init__()
        self.n_class = n_class

        self.cv0 = nn.Conv1d(3, 64, kernel_size=1)
        self.se0 = SELayer(64)
        self.cv1 = nn.Conv1d(64, 128, kernel_size=1)
        self.se1 = SELayer(128)
        self.block0 = ResStack(128, out_ch)
        blocks = []
        for i in range(1, n_stack):
            blocks.append(ResStack(out_ch, out_ch))
        self.blocks = nn.ModuleList(blocks)
        self.conv1 = nn.Conv1d(out_ch, 2048, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(2048, n_class, kernel_size=3, padding=1)

        self.tatc = TATC(n_class)

    def forward(self, x):
        """
        x must be shape (n_batch, 3, seq_len)
        """
        out = self.se0(F.relu(self.cv0(x)))
        out = self.se1(F.relu(self.cv1(out)))
        res_out, sum_skip_out = self.block0(out)
        for b in self.blocks:
            res_out, skip_out = b(res_out)
            sum_skip_out += skip_out
        res_out += sum_skip_out.sum(dim=0)

        out = F.relu(self.conv1(res_out))
        out = self.conv2(out)

        return out.transpose(2,1)

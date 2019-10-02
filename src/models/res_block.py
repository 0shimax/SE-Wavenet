import torch.nn as nn
import torch
import torch.nn.functional as F
from models.selayer import SELayer


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, dilation=1, padding=0):
        super().__init__()
        self.res_dcnv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                                   stride=1, padding=padding, dilation=dilation)
        self.skip_dcnv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                                   stride=1, padding=padding, dilation=dilation)
        self.res_cnv  = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.skip_cnv = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.se = SELayer(out_ch)

    def forward(self, x):
        res_out = torch.tanh(self.res_dcnv(x))
        skip_out = F.relu(self.skip_dcnv(x))

        out = res_out * skip_out

        res_out = self.res_cnv(out)
        res_out = self.se(res_out)
        res_out += x
        skip_out = self.skip_cnv(out)
        return res_out, skip_out


class ResStack(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        self.ls_block = []

        n_dilation = 2**0
        padding = n_dilation
        self.ls_block.append(ResBlock(in_ch, out_ch, kernel_size=kernel_size,
                             dilation=n_dilation, padding=padding))
        for i in range(1, 10):
            n_dilation = 2**i
            padding = n_dilation
            self.ls_block.append(ResBlock(out_ch, out_ch, kernel_size=kernel_size,
                                          dilation=n_dilation, padding=padding))
        self.ls_block = nn.ModuleList(self.ls_block)

    def forward(self, x):
        l_skip = []
        res_out, skip_out = self.ls_block[0](x)
        l_skip.append(skip_out)
        for i in range(1, len(self.ls_block)):
            res_out, skip_out = self.ls_block[i](res_out)
            l_skip.append(skip_out)
        return res_out, torch.stack(l_skip)

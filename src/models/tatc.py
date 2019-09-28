import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

from models.res_block import ResStack
from models.selayer import SELayer


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TATC(nn.Module):
    def __init__(self, n_class, hidden_size=256, out_ch=128, n_stack=3, n_emb_hidden=195):
        super().__init__()
        self.n_class = n_class
        self.hidden_size = hidden_size

        self.cv0 = nn.Conv1d(3, 64, kernel_size=1)
        self.se0 = SELayer(64)
        self.cv1 = nn.Conv1d(64, out_ch, kernel_size=1)
        self.se1 = SELayer(out_ch)

        # self.conv1 = nn.Conv1d(64, out_ch, kernel_size=6)
        self.block0 = ResStack(out_ch, out_ch)
        blocks = []
        for i in range(1, n_stack):
            blocks.append(ResStack(out_ch, out_ch))
        self.blocks = nn.ModuleList(blocks)

        self.linear1 = nn.Linear(2*out_ch, 1024)
        self.linear2 = nn.Linear(1024, 2)

        self.drop2 = nn.Dropout(0.5)

    def select_data_per_labels(self, raw_data, inf_labels, in_ch=3, device="cpu"):
        """
        input:
            raw_data: (n_batch, 3, seq_len)
        """
        n_batch, in_ch, seq_len = raw_data.shape
        inf_labels = inf_labels.reshape([n_batch, seq_len])
        raw_data = raw_data.reshape([n_batch, seq_len, in_ch])

        data_per_inf_labels = []
        for i_batch in range(n_batch):
            b_inf_labels = inf_labels[i_batch]
            b_raw_data = raw_data[i_batch]
            data_per_inf_labels_per_batch = []
            for i in range(1, self.n_class):
                mask = torch.where(b_inf_labels==i, torch.ByteTensor([1]).to(device), torch.ByteTensor([0]).to(device))
                mask = mask.type(torch.bool)
                masked = torch.masked_select(b_raw_data, mask.repeat(1, in_ch).reshape([seq_len, in_ch]))
                masked = masked.reshape(-1, in_ch) if masked.shape[0]!=0 else torch.zeros([1, in_ch]).to(device)
                data_per_inf_labels_per_batch.append(masked)
            data_per_inf_labels.append(data_per_inf_labels_per_batch)
        self.data_per_inf_labels = data_per_inf_labels

    def _initialize_gru(self):
        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

    def forward(self):
        # self.ifo_pool.reset()

        l_out = []
        n_batch = len(self.data_per_inf_labels)
        for i_batch in range(n_batch):
            for i_class, x in enumerate(self.data_per_inf_labels[i_batch]):
                seq_len, feat_dim = x.shape

                x = x.transpose(1,0).unsqueeze(0)

                out = self.se0(F.relu(self.cv0(x)))
                out = self.se1(F.relu(self.cv1(out)))

                res_out, sum_skip_out = self.block0(out)
                for b in self.blocks:
                    res_out, skip_out = b(res_out)
                    sum_skip_out += skip_out
                res_out += sum_skip_out.sum(dim=0)

                out = res_out # .squeeze(3).unsqueeze(1)
                out_avg = F.avg_pool1d(out, out.shape[-1])
                out_max = F.max_pool1d(out, out.shape[-1])
                out = torch.cat([out_avg, out_max], dim=-1).reshape(1, -1)

                out = F.relu(self.linear1(out))
                out = self.drop2(out)
                out = self.linear2(out)
                l_out.append(out)
        return torch.stack(l_out).squeeze(1)

from pathlib import Path
import random
random.seed(555)
import numpy as np
np.random.seed(555)
import pandas as pd
import torch
from torch.utils.data import Dataset

from features.arrangement_time import get_values_per_second
# from features.diff_dim_seq import add_diff_features
from data.test_labels import gt_lack_labels


class ActivDataset(Dataset):
    # seq_len is fixed length of time series
    def __init__(self, actigraph_data_file_names, root_dir,
                 seq_len=100, time_step=1, subset=False, transform=None,
                 is_train=False, add_noise=False, test_in_train=False):
        super().__init__()
        self.actigraph_data_file_names = actigraph_data_file_names
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.time_step = time_step
        self.is_train = is_train
        self.add_noise = add_noise
        self.test_in_train = test_in_train

    def __len__(self):
        return len(self.actigraph_data_file_names)

    def __getitem__(self, idx):
        activ_file_name = self.actigraph_data_file_names[idx]
        data_file_path = Path(self.root_dir, activ_file_name)
        raw_data = pd.read_csv(data_file_path)
        features, labels = get_values_per_second(raw_data, self.time_step)

        if self.is_train:
            assert labels.max()==5, "Index {} is invalid labeled file.".format(idx)

        padded_features, padded_labels, lack_labels = \
            self._get_padded_data(features, labels, idx)
        # padded_features = add_diff_features(padded_features)
        return (torch.FloatTensor(padded_features.transpose()),
                torch.LongTensor(padded_labels),
                torch.LongTensor(lack_labels))

    def _get_padded_data(self, features, labels, data_idx):
        def create_lack_labels(n_all_label_idxs, n_input_len, labels,
                               rand_idxs, _all_label_idxs):
            n_diff = n_all_label_idxs - n_input_len
            lack_labels = np.zeros(labels.max())
            if n_diff==0:
                return lack_labels

            sum_block_len = 0
            # label_0:5, label_1:0,label_2:1,label_3:2,label_4:3,label_5:4,
            for i in rand_idxs[::-1]:
                diff_from_lack = n_diff - sum_block_len
                sum_block_len += len(_all_label_idxs[i])
                if i!=5 and diff_from_lack / len(_all_label_idxs[i]) > .5:
                    lack_labels[i] = 1

                if sum_block_len >= n_diff:
                    break
            return lack_labels

        if self.is_train:
            padded_features = np.random.uniform(-.3, .3, [self.seq_len, 3]).astype(np.float32)
            padded_labels = np.ones((self.seq_len), dtype=np.int32) * -1

            zero_idx = self._extract_one_zero_seq_idx(labels)
            non_zero_idx, extra_lack_labels = self._extract_non_zero_seq_idx(labels)
            _all_label_idxs = non_zero_idx + zero_idx

            rand_idxs = np.random.permutation(len(_all_label_idxs))
            all_label_idxs = [idx for rand_idx in rand_idxs for idx in _all_label_idxs[rand_idx]]
            n_all_label_idxs = len(all_label_idxs)
            n_input_len = min(self.seq_len, n_all_label_idxs)
            if self.add_noise:
                padded_features[:n_input_len] += features[all_label_idxs][:self.seq_len]
            else:
                padded_features[:n_input_len] = features[all_label_idxs][:self.seq_len]
            padded_labels[:n_input_len] = labels[all_label_idxs][:self.seq_len]
            lack_labels = create_lack_labels(n_all_label_idxs, n_input_len, labels, rand_idxs, _all_label_idxs)
            lack_labels = [1 if lb + ellb >= 1 else 0 for lb, ellb in zip(lack_labels, extra_lack_labels)]
            return padded_features, padded_labels, lack_labels
        else:
            if self.test_in_train:
                lack_labels = np.zeros(labels.max())
            else:
                lack_labels = np.array(gt_lack_labels[data_idx])
            return features, labels, lack_labels

    def _extract_one_zero_seq_idx(self, labels):
        pre = None
        seq = []
        sub_seq = []
        zero_idx = np.where(labels==0)[0]
        for x in zero_idx:
            if pre is None:
                pre = x
                sub_seq.append(x)
                continue

            if x==zero_idx[-1]:
                sub_seq.append(x)
                seq.append(sub_seq)
            elif x==pre+1:
                sub_seq.append(x)
                pre = x
            else:
                seq.append(sub_seq)
                sub_seq = []
                pre = x
        return [seq[random.randrange(len(seq))]]

    def _normalize(self, features):
        for i in range(features.shape[1]):
            features[:, i] = pd.Series(features[:, i]).ewm(span=3).mean()
        return features

    def __extract_on_zero_seq_idx(self, labels):
        return [np.where(labels == l)[0] for l in range(1, labels.max()+1)]

    def _extract_non_zero_seq_idx(self, labels):
        seq = []
        lack_labels = []
        for l in range(1, labels.max()+1):
            lr, lack_label = self._create_terminated_label()
            lack_labels.append(lack_label)

            seq_per_class = np.where(labels == l)[0]
            if lr < 1. and random.uniform(0,1) > 0.5:
                seq_per_class = seq_per_class[:int(len(seq_per_class)*lr)]
            elif lr < 1. and random.uniform(0,1) <= 0.5:
                seq_per_class = seq_per_class[int(len(seq_per_class)*(1 - lr)):]
            seq.append(seq_per_class)
        return seq, lack_labels

    def _create_terminated_label(self, remain_ratio=[.5, .333, 1., 1., 1.]):
        ratio_len = len(remain_ratio)
        lr = remain_ratio[random.randrange(ratio_len)]
        if lr < 1.:
            return lr, 1
        else:
            return lr, 0


def loader(dataset, batch_size,  shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader

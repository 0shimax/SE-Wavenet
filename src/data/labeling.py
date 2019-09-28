import pandas as pd
import numpy as np
from pathlib import Path

from test_labels import seq_labels


def add_labels_to_csv(data_root_path, work_logs, pointer_path, d_type="train"):
    if d_type=="train":
        raw_data_path = Path(data_root_path, "raw", pointer_path)
    elif d_type=="test":
        raw_data_path = Path(data_root_path, "raw", d_type, pointer_path)
        
    with open(raw_data_path) as f:
        data_file_names = [line.rstrip() for line in f]

    for i, data_f in enumerate(data_file_names):
        out_path = Path(data_root_path, d_type, data_f.replace(".csv", "_labbeled.csv"))
        if out_path.exists():
            continue

        if d_type=="train":
            read_file_path = Path(data_root_path, "raw", data_f)
        elif d_type=="test":
            read_file_path = Path(data_root_path, "raw", d_type, data_f)
            
        df_data = pd.read_csv(read_file_path)
        df_data = binary_labeling(df_data)
        df_data = interpolate_binary_labeling_for_non_zero(df_data)
        df_data = interpolate_binary_labeling_for_zero(df_data)
        if d_type == "train":
            df_data = multi_labeling(df_data, work_logs)
        elif d_type == "test":
            df_data = multi_labeling(df_data, seq_labels[i])
        df_data.to_csv(out_path, index=False)


def binary_labeling(df_data, threshold=0.3):
    labels = []
    pre_labels = np.array([None]*df_data.shape[0])
    work_log_idx = 0
    for i, (t, ax, ay, az, at) in enumerate(df_data.values):
        x_flag = -1 * threshold < ax < threshold
        y_flag = -1 * threshold < ay < threshold
        z_flag = -1 * threshold < az < threshold
        xy_flag = x_flag & y_flag
        xz_flag = x_flag & z_flag
        yz_flag = y_flag & z_flag

        if xy_flag or xz_flag or yz_flag:
            labels.append(0)
        else:
            labels.append(-1)
    return df_data.assign(label=labels)


def interpolate_binary_labeling_for_non_zero(df_data, zero_len_th=200):
    n_raw, _ = df_data.shape
    original_labels = df_data.label.values

    skip_to_idx = None
    for i in range(len(original_labels)):
        if skip_to_idx and i < skip_to_idx:
            continue

        next_non_zero_idx = None
        if original_labels[i] != 0:
            remain = n_raw - i - 1
            z_len = zero_len_th if remain >= zero_len_th else remain
            for z_idx in range(z_len):
                label = original_labels[i+z_idx+1]
                if label != 0:
                    next_non_zero_idx = i + z_idx + 1
                    skip_to_idx = i + z_idx + 1
                    original_labels[i:next_non_zero_idx+1] = -1
                    break
    df_data.label = original_labels
    return df_data


def interpolate_binary_labeling_for_zero(df_data, non_zero_len_th=50):
    n_raw, _ = df_data.shape
    original_labels = df_data.label.values

    skip_to_idx = None
    for i in range(len(original_labels)):
        if skip_to_idx and i < skip_to_idx:
            continue

        next_zero_idx = None
        if original_labels[i] == 0:
            remain = n_raw - i - 1
            nz_len = non_zero_len_th if remain >= non_zero_len_th else remain
            for z_idx in range(nz_len):
                label = original_labels[i+z_idx+1]
                if label == 0:
                    next_zero_idx = i + z_idx + 1
                    skip_to_idx = i + z_idx + 1
                    original_labels[i:next_zero_idx+1] = 0
                    break
    df_data.label = original_labels
    return df_data


def multi_labeling(df_data, work_logs):
    original_labels = df_data.label.values

    work_logs_idx = -1
    pre_label = 0
    for i in range(len(original_labels)):
        if pre_label == 0 and original_labels[i] != 0:
            work_logs_idx += 1
            original_labels[i] = work_logs[work_logs_idx]
        elif original_labels[i] != 0:
            original_labels[i] = work_logs[work_logs_idx]
        elif original_labels[i] == 0 and work_logs_idx == len(work_logs) - 1:
            original_labels[i:] = 0
            break
        pre_label = original_labels[i]
    df_data.label = original_labels
    return df_data


if __name__ == "__main__":
    d_type = "test"
    print("d_type:", d_type)
    
    if d_type == "train":
        data_root_path = "/mnt/tracker_data"
        work_logs = [1, 2, 3, 4, 5]
        pointer_path = "train_data_file_pointer"
        add_labels_to_csv(data_root_path, work_logs, pointer_path, d_type="train")
    elif d_type == "test":
        data_root_path = "/mnt/tracker_data"
        work_logs = None
        pointer_path = "test_data_file_pointer"
        add_labels_to_csv(data_root_path, work_logs, pointer_path, d_type="test")

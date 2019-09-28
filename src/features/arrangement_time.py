import numpy as np
import pandas as pd

def get_values_per_second(df_raw, time_step=1):
    """
    df_raw: has argments of 'time', 'ax','ay', 'az', 'aT', 'label'
    """
    max_time = int(df_raw.time.max())
    results = np.zeros((int(max_time*1/time_step)+1, 3), dtype=np.float32)
    labels = np.zeros(int(max_time*1/time_step)+1, dtype=np.uint8)

    for i, t in enumerate(np.arange(1, max_time+1*time_step, time_step)):
        result = df_raw[df_raw.time>=t].iloc[0].values.astype(np.float32)

        results[i] = result[1:-2]
        labels[i] = result[-1]

    return results, labels

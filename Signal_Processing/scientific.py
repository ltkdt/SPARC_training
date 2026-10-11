import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv(os.path.join(os.getcwd(), "Signal_Processing", "csv_data.csv"))

# 2. Lấy về số hàng và số cột của DataFrame
n_rows, n_cols = df.shape

print("Number of rows:", n_rows)
print("Number of columns:", n_cols)

# 3. In ra header của file
print("\nColumn names:")
print(df.columns.tolist())

# 4. In 5 row đầu tiên 
print("\nFirst 5 rows:")
print(df.head())

fs = 200 
duration = len(df) / fs # 4.065

# print(duration)


df.columns = [f"SC_{i+1}" for i in range(df.shape[1])] # Gán tên cột cho dễ gọi
print(df.head())
# Giá trị biên độ tất cả subcarrier của packet 1
packet_1 = df.iloc[0, :]
# Subcarrier đầu tiên của tất cả packet
sc_1 = df.iloc[:, 0]
# 100 packet đầu tiên, 10 subcarrier đầu tiên
subset = df.iloc[0:100, 0:10]
print("subset shape:", subset.shape)
# Chọn subcarrier 1 theo tên cột
sc_1 = df.loc[:, "SC_1"]
# chọn nhiều subcarrier theo tên cột (trên thực tế cũng có thể làm được bằng cách dùng iloc với index cần tìm)
selected = df.loc[:, ["SC_1", "SC_5", "SC_10"]]
# Chọn 100 packet đầu tiên, 10 subcarrier đầu tiên theo tên cột
subset = df.loc[0:99, "SC_1":"SC_10"]

print(selected.shape)  # (813, 3)
print(subset.shape)    # (100, 10)

csi = df.to_numpy(dtype=np.float64)

segment = csi[200:400, :]
selected = csi[:, [0, 5, 10, 15]]



# Chọn 800 packet đầu tiên, tất cả subcarrier
csi_800 = csi[:800, :]
# Chia thành 8 window, mỗi window có 100 packet, tất cả subcarrier
#  --> Cấu trúc tensor 3 chiều (8, 100, 52)
windows = csi_800.reshape(8, 100, 52)

'''
Vectorization example: multiply all elements of csi by 2 and store in result
'''
result = np.empty_like(csi)
# Normal method:
for i in range(800):
    for j in range(52):
        result[i, j] = csi[i, j] * 2
# Vectorize:
result = csi * 2


'''
Broadcasting example
'''
# Với từng subcarrier, tính mean và std của tất cả packet
mean = np.mean(csi, axis=0)
std = np.std(csi, axis=0)

print(mean.shape)  # (52,)
print(std.shape)   # (52,)

# Chuẩn hóa dữ liệu: với từng subcarrier, trừ đi mean và chia cho std (Z-score)
csi_normalized = (csi - mean) / (std + 1e-8)

print(csi_normalized.shape)  # (813, 52)

result = np.empty_like(csi)


import numpy as np
import matplotlib.pyplot as plt

def plot_csi_subcarriers(df, fs=200):
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Time axis
    time = np.arange(len(df)) / fs

    # Select 4 subcarriers
    subcarriers = ["SC_1", "SC_10", "SC_20", "SC_30"]

    for ax, sc in zip(axes.flat, subcarriers):
        ax.plot(time, df[sc], label=sc)

        ax.set_title(f"CSI Amplitude - {sc}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    plt.savefig(os.path.join("Signal_Processing", "csi_subcarriers.png"))

plot_csi_subcarriers(df)
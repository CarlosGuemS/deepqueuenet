import os
import shutil
import sys
import pandas as pd
from itertools import permutations
from heapq import merge
from concurrent.futures import ProcessPoolExecutor

DS_PATH = "/home/data/Testbed_datasets/one_router/one_router_1cbr_1mb_1TG/"
sys.path.append(DS_PATH)
from datanetAPI import DatanetAPI


def process_sample(packet_info_matrix, split, sample_id):
    # Process sequences
    processed_sequences = list()
    for ii, jj in permutations(range(5), 2):
        if packet_info_matrix[ii][jj]:
            for packet_sequence in packet_info_matrix[ii][jj][0]:
                processed_sequences.append(
                    [
                        (pkt[0], pkt[1], ii, jj, pkt[2])
                        for pkt in packet_sequence
                        if len(pkt) == 3
                    ]
                )
    # Merge sequence
    merged_sequence = list(merge(*processed_sequences, key=lambda x: x[0]))
    # Transform ts from ns to s, set up initial time as 0
    merged_sequence = [
        ((ts - merged_sequence[0][0]) / 1e9, size, ii, jj, delay)
        for ts, size, ii, jj, delay in merged_sequence
    ]

    # Store as csv
    pd.DataFrame(
        {
            "timestamp (sec)": [ts for ts, _, _, _, _ in merged_sequence],
            "pkt len (byte)": [size for _, size, _, _, _ in merged_sequence],
            "priority": [0 for _ in merged_sequence],
            "src": [ii for _, _, ii, _, _ in merged_sequence],
            "dst": [jj for _, _, _, jj, _ in merged_sequence],
            "time_in_sys": [delay for _, _, _, _, delay in merged_sequence],
        }
    ).to_csv(f"../data/5-port router/{split}/5port_sample_{sample_id}.csv", index=False)


# Create folder
if os.path.exists("../data/5-port router"):
    print("Folder already exists, removing it...")
    shutil.rmtree("../data/5-port router")
os.mkdir("../data/5-port router")
os.mkdir("../data/5-port router/_train")
os.mkdir("../data/5-port router/_test")
print("Folder created")

ds = DatanetAPI(DS_PATH, shuffle=True)
# Get ds_length
ds_length = sum([1 for _ in iter(ds)])

# Split into test and train
train_length = int(0.9 * ds_length)

# Process samples
with ProcessPoolExecutor(15) as executor:
    for raw_id, sample in enumerate(iter(ds)):
        split = "_train" if raw_id < train_length else "_test"
        executor.submit(
            process_sample,
            sample.get_pkts_info_object(),
            split,
            sample.get_sample_id()[1],
        )
# for raw_id, sample in enumerate(iter(ds)):
#     split = "_train" if raw_id < train_length else "_test"
#     print(sample.get_sample_id()[1])
#     process_sample(
#         sample.get_pkts_info_object(),
#         split,
#         sample.get_sample_id()[1],
#     )
#     break
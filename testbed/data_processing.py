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


def process_sample_total_agg(packet_info_matrix, split, sample_id):
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
    ).to_csv(
        f"../data/5-port router/full_device/_traces/{split}/5port_sample_{sample_id}.csv",
        index=False,
    )


def process_sample_port_agg(packet_info_matrix, split, sample_id, exit_agg=False):
    # Process sequences
    # if exit_agg is true, use exit port
    # if exit_agg is false, use entry port
    processed_sequences = [list() for _ in range(5)]
    for ii, jj in permutations(range(5), 2):
        if packet_info_matrix[ii][jj]:
            for packet_sequence in packet_info_matrix[ii][jj][0]:
                idx = jj if exit_agg else ii
                processed_sequences[idx].append(
                    [
                        (pkt[0], pkt[1], ii, jj, pkt[2])
                        for pkt in packet_sequence
                        if len(pkt) == 3
                    ]
                )
    # Merge sequence
    merged_sequences = [
        list(merge(*processed_sequences[jj], key=lambda x: x[0])) for jj in range(5)
    ]
    # Transform ts from ns to s, set up initial time as 0
    global_initial_time = min(
        merged_sequence[0][0] for merged_sequence in merged_sequences
    )
    merged_sequences = [
        [
            ((ts - global_initial_time) / 1e9, size, ii, jj, delay)
            for ts, size, ii, jj, delay in merged_sequence
        ]
        for merged_sequence in merged_sequences
    ]

    # Store as separate csv's
    for jj, merged_sequence in enumerate(merged_sequences):
        pd.DataFrame(
            {
                "timestamp (sec)": [ts for ts, _, _, _, _ in merged_sequence],
                "pkt len (byte)": [size for _, size, _, _, _ in merged_sequence],
                "priority": [0 for _ in merged_sequence],
                "src": [ii for _, _, ii, _, _ in merged_sequence],
                "dst": [jj for _, _, _, jj, _ in merged_sequence],
                "time_in_sys": [delay for _, _, _, _, delay in merged_sequence],
            }
        ).to_csv(
            "../data/5-port router/{}_port/_traces/{}/5port_sample{}_port{}.csv".format(
                "exit" if exit_agg else "entry", split, sample_id, jj
            ),
            index=False,
        )


# Create folder
if os.path.exists("../data/5-port router"):
    print("Folder already exists, removing it...")
    shutil.rmtree("../data/5-port router")
os.makedirs("../data/5-port router/exit_port/_traces/_train")
os.makedirs("../data/5-port router/exit_port/_traces/_test")
os.makedirs("../data/5-port router/entry_port/_traces/_train")
os.makedirs("../data/5-port router/entry_port/_traces/_test")
os.makedirs("../data/5-port router/full_device/_traces/_train")
os.makedirs("../data/5-port router/full_device/_traces/_test")
print("Folder created")

ds = DatanetAPI(DS_PATH, shuffle=True)
# Get ds_length
ds_length = sum([1 for _ in iter(ds)])

# Split into test and train
train_length = int(0.02 * ds_length)
test_length = train_length + int(0.02 * ds_length)

# Process samples
with ProcessPoolExecutor(15) as executor:
    for raw_id, sample in enumerate(iter(ds)):
        if raw_id < train_length:
            split = "_train"
        elif raw_id < test_length:
            split = "_test"
        else:
            break
        pkts_info = sample.get_pkts_info_object()
        true_id = sample.get_sample_id()[1]
        executor.submit(
            process_sample_total_agg,
            pkts_info,
            split,
            true_id,
        )
        executor.submit(
            process_sample_port_agg,
            pkts_info,
            split,
            true_id,
        )
        executor.submit(
            process_sample_port_agg,
            pkts_info,
            split,
            true_id,
            True,
        )
# for raw_id, sample in enumerate(iter(ds)):
#     split = "_train" if raw_id < train_length else "_test"
#     print(sample.get_sample_id()[1])
#     process_sample_port_agg(
#         sample.get_pkts_info_object(),
#         split,
#         sample.get_sample_id()[1],
#     )
#     break

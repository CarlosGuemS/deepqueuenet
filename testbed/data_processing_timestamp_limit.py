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


def process_sample_total_agg(packet_info_matrix, split, sample_id, timestamp_lim):
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
    clean_merged_sequence = []
    for ts, size, ii, jj, delay in merged_sequence:
        clean_ts = (ts - merged_sequence[0][0]) / 1e9
        if clean_ts > timestamp_lim:
            break
        clean_merged_sequence.append((clean_ts, size, ii, jj, delay))

    # Store as csv
    timestamp_lim_str = f"{int(timestamp_lim*1000)}ms"
    pd.DataFrame(
        {
            "timestamp (sec)": [ts for ts, _, _, _, _ in clean_merged_sequence],
            "pkt len (byte)": [size for _, size, _, _, _ in clean_merged_sequence],
            "priority": [0 for _ in clean_merged_sequence],
            "src": [ii for _, _, ii, _, _ in clean_merged_sequence],
            "dst": [jj for _, _, _, jj, _ in clean_merged_sequence],
            "time_in_sys": [delay for _, _, _, _, delay in clean_merged_sequence],
        }
    ).to_csv(
        f"../data/5-port router/full_device/{timestamp_lim_str}"
        + f"/_traces/{split}/5port_sample_{sample_id}.csv",
        index=False,
    )


def process_sample_port_agg(
    packet_info_matrix,
    split,
    sample_id,
    timestamp_lim,
    exit_agg=False,
):
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
    clean_merged_sequences = [list() for _ in range(5)]
    for clean_merged_sequence, merged_sequence in zip(
        clean_merged_sequences, merged_sequences
    ):
        for ts, size, ii, jj, delay in merged_sequence:
            clean_ts = (ts - global_initial_time) / 1e9
            if clean_ts > timestamp_lim:
                break
            clean_merged_sequence.append((clean_ts, size, ii, jj, delay))

    # Store as separate csv's
    timestamp_lim_str = f"{int(timestamp_lim*1000)}ms"
    for jj, clean_merged_sequence in enumerate(clean_merged_sequences):
        pd.DataFrame(
            {
                "timestamp (sec)": [ts for ts, _, _, _, _ in clean_merged_sequence],
                "pkt len (byte)": [size for _, size, _, _, _ in clean_merged_sequence],
                "priority": [0 for _ in clean_merged_sequence],
                "src": [ii for _, _, ii, _, _ in clean_merged_sequence],
                "dst": [jj for _, _, _, jj, _ in clean_merged_sequence],
                "time_in_sys": [delay for _, _, _, _, delay in clean_merged_sequence],
            }
        ).to_csv(
            f"../data/5-port router/{'exit' if exit_agg else 'entry'}_port_limited/"
            + f"/{timestamp_lim_str}/_traces/{split}/"
            + f"5port_sample{sample_id}_port{jj}.csv",
            index=False,
        )


assert not os.path.exists("../data/5-port router/"), "Non empty folder already exists, aborting"
     
ds = DatanetAPI(DS_PATH, shuffle=True)
# Get ds_length
ds_length = sum([1 for _ in iter(ds)])

# Split into test and train
train_length = 1262
test_length = train_length + 8

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
        for timestamp in [0.50, 0.1, 0.5, 1]:
            executor.submit(
                process_sample_total_agg,
                pkts_info,
                split,
                true_id,
                timestamp,
            )
            executor.submit(
                process_sample_port_agg,
                pkts_info,
                split,
                true_id,
                timestamp,
            )
            executor.submit(
                process_sample_port_agg,
                pkts_info,
                split,
                true_id,
                timestamp,
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

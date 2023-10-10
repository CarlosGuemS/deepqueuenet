# Script that takes a command line argument and logs the memory usage of the process every second.
# Usage: python mem_usage_wrapper.py <outputfile> <command>
import sys
import subprocess
import psutil
from time import sleep

# Output file
output_file = sys.argv[1]
# Command to run
cmd = sys.argv[2:]

with open(output_file, "w") as ff:

    # Run the command
    pp = subprocess.Popen(cmd)
    process = psutil.Process(pp.pid)

    while pp.poll() is None:
        # Get the memory usage
        ff.write(f"{process.memory_info().rss}\n")
        ff.flush()
        # Sleep for a second
        sleep(5)
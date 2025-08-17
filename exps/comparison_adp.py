import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--discount", type=float)
parser.add_argument("--precision", type=float)
args = parser.parse_args()

assert (
    args.model is not None and args.precision is not None and args.discount is not None
), "--model name --discount 0.99 --precision 1e-2 par exemple"

state_dims = [100, 5000, 25000]  # , 100000, 250000, 1000000]
processes = []

for state in state_dims:
    command = "python adp.py --model {} --state {} --discount {} --precision {}".format(
        args.model, state, args.discount, args.precision
    )
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

# Wait for all subprocesses to complete
for process in processes:
    process.communicate()

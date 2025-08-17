import threading
import subprocess

levels = range(2)
epsilon = 1e-2

model_names = [
    # "barto",
    # "block",
    # "chained",
    # "dam",
    # "garnet",
    # "impatience",
    # "inventory",
    # "maze_backtrack",
    # "maze_prims",
    # "maze_wilson",
    "mountain",
    # "replacement",
    "rooms",
    # "sutton",
    "tandem",
    # "toy",
    "inv_control",
]


def run_command(level, model_name):
    command = f"python plot_discount_impact.py --level {level} --model {model_name} --epsilon {epsilon}"
    subprocess.run(command, shell=True)


threads = []

# Iterate over parameters and launch subprocesses in separate threads
for level in levels:
    for model_name in model_names:
        thread = threading.Thread(target=run_command, args=(level, model_name))
        thread.start()
        threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()
